// FP8 E4M3 path-B: scale-postponed.
// Load FP8 byte, decode mantissa+exp to raw FP value WITHOUT the d_block
// multiply, cast to half (lossless for E4M3 magnitudes since FP16 has more
// precision). Post-mma SMEM round-trip applies d_block per (n, k_block).

#pragma once

#include "tc_grid.h"
#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::fp8_b {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float e8m0_to_f32_dev(uint8_t e) {
    return __int_as_float(((int) e) << 23);
}

__device__ __forceinline__ float e4m3fn_to_f32_dev_raw(uint8_t b) {
    bool neg = (b & 0x80) != 0;
    int  E   = (b >> 3) & 0xF;
    int  M   = b & 0x7;
    float v;
    if (E == 0) {
        v = ldexpf((float) M / 8.0f, -6);
    } else if (E == 15 && M == 7) {
        v = 0.0f;
    } else {
        v = ldexpf(1.0f + (float) M / 8.0f, E - 7);
    }
    return neg ? -v : v;
}

template <int BM_, int BN_, int BK_, int WARPS_>
__global__ void mm_f8_e4m3_b128_post_scale(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    static_assert(BN / WARPS == 16, "BN/WARPS must equal 16");
    static_assert(BK % 16 == 0, "BK constraints");

    constexpr int kBlkBytes = 1 + QK_F8;

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + BM * BK;
    float  * sC = reinterpret_cast<float *>(sB + BK * BN);

    FragC c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int blocks_per_row = K / QK_F8;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles = K / BK;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k0 = kt * BK;
        for (int idx = tid; idx < BM * BK; idx += threads) {
            int mm = idx / BK, kk = idx % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float v = (gm < M && gk < K) ? A[(size_t) gm * K + gk] : 0.0f;
            sA[mm * BK + kk] = __float2half(v);
        }
        for (int idx = tid; idx < BN * BK; idx += threads) {
            int nn = idx / BK, kk = idx % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v = __float2half(0.0f);
            if (gn < N && gk < K) {
                int blk = gk / QK_F8;
                int in_blk = gk - blk * QK_F8;
                const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                uint8_t q = row[blk * kBlkBytes + 1 + in_blk];
                v = __float2half(e4m3fn_to_f32_dev_raw(q));
            }
            sB[kk + nn * BK] = v;
        }
        __syncthreads();

        // For FP8, block size = QK_F8 = 128 > BK=32 or 64. So multiple BK-tiles
        // share the same block scale within a (n) column. We still do per-16-K-slice
        // mma with per-warp-column scale application, but for slices INSIDE one block
        // the scale is identical. We accept the redundant work in this pass.
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a; wmma::load_matrix_sync(a, sA + kk, BK);
            int n_off = warp * 16;
            FragB b; wmma::load_matrix_sync(b, sB + kk + n_off * BK, BK);
            FragC partial; wmma::fill_fragment(partial, 0.0f);
            wmma::mma_sync(partial, a, b, partial);

            float * tmp = sC + (size_t) warp * BM * 16;
            wmma::store_matrix_sync(tmp, partial, 16, wmma::mem_row_major);
            __syncwarp();

            int blk_idx_global = (k0 + kk) / QK_F8;

            for (int idx = lane; idx < BM * 16; idx += 32) {
                int mm = idx / 16, nn = idx % 16;
                int gn = tile_n + n_off + nn;
                int gm = tile_m + mm;
                if (gm < M && gn < N) {
                    const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                    uint8_t e = row[blk_idx_global * kBlkBytes];
                    float d = e8m0_to_f32_dev(e);
                    tmp[idx] *= d;
                } else {
                    tmp[idx] = 0.0f;
                }
            }
            __syncwarp();
            FragC scaled; wmma::load_matrix_sync(scaled, tmp, 16, wmma::mem_row_major);
            #pragma unroll
            for (int e = 0; e < c_frag.num_elements; ++e) c_frag.x[e] += scaled.x[e];
        }
        __syncthreads();
    }

    float * tile_c = sC + (size_t) warp * BM * 16;
    wmma::store_matrix_sync(tile_c, c_frag, 16, wmma::mem_row_major);
    __syncwarp();
    int n_off = warp * 16;
    for (int idx = lane; idx < BM * 16; idx += 32) {
        int mm = idx / 16, nn = idx % 16;
        int gm = tile_m + mm, gn = tile_n + n_off + nn;
        if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
    }
}

}  // namespace tc_grid::kernels::fp8_b

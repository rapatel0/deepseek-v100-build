// MXFP4 path-B: scale-postponed.
// Load nibble, decode to raw signed magnitude (kvals * sign * 0.5), WITHOUT
// the per-block d_block multiply. Store as FP16 in SMEM. After mma_sync,
// apply d_block per (n, k_block) via SMEM round-trip on the FP32 partial
// fragment, identical pattern to INT path-B.
//
// Win: 1 multiplication per element saved during dequant (the d_block factor),
//      plus the FP16 cast of d*kval*0.5 -> half is replaced by an exact
//      kval-table FP16 lookup (kvals are all exactly representable in FP16).
//
// Trade-off: per-K-block SMEM round-trip overhead. Same as INT path-B.

#pragma once

#include "tc_grid.h"
#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::fp4_b {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float e8m0_to_f32_dev(uint8_t e) {
    return __int_as_float(((int) e) << 23);
}

template <int BM_, int BN_, int BK_, int WARPS_>
__global__ void mm_mxfp4_post_scale(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    static_assert(BN / WARPS == 16, "BN/WARPS must equal 16");
    static_assert(BK % 16 == 0 && BK % QK_MXFP4 == 0, "BK constraints");

    // FP4 magnitude table. All values exactly representable in FP16, so the
    // __float2half_rn cast is lossless. constexpr so no dynamic init.
    constexpr float kvals_f[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    constexpr int kBlkBytes = 1 + (QK_MXFP4 / 2);

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

    const int blocks_per_row = K / QK_MXFP4;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles = K / BK;

    // Per-block d_block scales for this CTA's BN columns -- pre-cache to
    // avoid recomputing the e8m0_to_f32 each round-trip.
    // For simplicity we just re-read from W_blocks in the round-trip loop.

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k0 = kt * BK;
        for (int idx = tid; idx < BM * BK; idx += threads) {
            int mm = idx / BK, kk = idx % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float v = (gm < M && gk < K) ? A[(size_t) gm * K + gk] : 0.0f;
            sA[mm * BK + kk] = __float2half(v);
        }
        // Load B as RAW kval*0.5 (no d_block applied)
        for (int idx = tid; idx < BN * BK; idx += threads) {
            int nn = idx / BK, kk = idx % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v = __float2half(0.0f);
            if (gn < N && gk < K) {
                int blk = gk / QK_MXFP4;
                int in_blk = gk - blk * QK_MXFP4;
                const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                const uint8_t * bptr = row + blk * kBlkBytes;
                uint8_t code = bptr[1 + (in_blk >> 1)];
                uint8_t nibble = ((in_blk & 1) == 0) ? (code & 0xF) : (code >> 4);
                // Raw value = kvals[nibble & 7] * sign * 0.5; we fold the 0.5
                // into the per-block post-scale (multiply d_block * 0.5).
                float m_f = kvals_f[nibble & 7];
                if (nibble & 8) m_f = -m_f;
                v = __float2half_rn(m_f);
            }
            sB[kk + nn * BK] = v;
        }
        __syncthreads();

        // Per 16-K slice: partial mma, apply (d_block * 0.5) post-mma per (n, blk).
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

            int blk_idx_global = (k0 + kk) / QK_MXFP4;
            // Each lane handles one (mm, nn) of the 16x16 sub-tile.
            for (int idx = lane; idx < BM * 16; idx += 32) {
                int mm = idx / 16, nn = idx % 16;
                int gn = tile_n + n_off + nn;
                int gm = tile_m + mm;
                if (gm < M && gn < N) {
                    const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                    uint8_t e = row[blk_idx_global * kBlkBytes];
                    float d = e8m0_to_f32_dev(e) * 0.5f;
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

}  // namespace tc_grid::kernels::fp4_b

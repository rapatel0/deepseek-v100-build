// INT4 path-B: mantissa-bitshift FP16 packing.
//
// INT4 values q are in [-7, 7] (we clip in quantizer; -8 unused). Shift to
// unsigned domain by adding 8 -> [1, 15], 4 bits. Pack into low 4 bits of FP16
// mantissa with anchor exponent 25 (high bits 0x6400). The resulting half has
//   value = 1024 + (q + 8) = 1032 + q   for q in [-7, 7]
// (exact since 1032 + q in [1025, 1039], all within mantissa LSB resolution
// of FP16 with anchor 2^10).
//
// After mma_sync: acc[m,n] = sum_k a[m,k] * (1032 + q[n,k])
//                          = 1032 * row_sum_a_slice[m] + true_dot[m,n]
//
// Per-block FP16 scale applied in the SMEM-roundtrip post-multiply pass,
// identical structure to int8 path-B. We unpack two nibbles per byte during
// the SMEM load (one byte = 2 INT4 values along K).

#pragma once

#include "tc_grid.h"
#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int4_b {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ half int4_to_half_packed(int8_t q) {
    // Pack signed nibble q in [-7, 7] -> half value (1032 + q).
    uint16_t bits = (uint16_t) 0x6400u | (uint16_t)((int) q + 8);
    return __ushort_as_half(bits);
}

template <int BM_, int BN_, int BK_, int WARPS_>
__global__ void mm_int4_bitshift(
        const uint8_t * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    static_assert(BN / WARPS == 16, "BN/WARPS must equal 16");
    static_assert(BK % 16 == 0, "BK multiple of 16");
    static_assert(BK % QK_INT4 == 0, "BK multiple of QK_INT4");

    constexpr float ANCHOR_BIAS = 1032.0f;

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

    const int K_pad_bytes = K / 2;
    const int blocks_per_row = K / QK_INT4;
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
                int byte_idx = gk / 2;
                uint8_t b = W_qs[(size_t) gn * K_pad_bytes + byte_idx];
                int8_t n;
                if ((gk & 1) == 0) n = (int8_t)((b & 0xF) << 4) >> 4;
                else               n = (int8_t)(b & 0xF0)        >> 4;
                v = int4_to_half_packed(n);
            }
            sB[kk + nn * BK] = v;
        }
        __syncthreads();

        // ITEM 5: per-block scaling (one SMEM round-trip per QK_INT4 block).
        constexpr int SLICES_PER_BLOCK = QK_INT4 / 16;  // 2 for QK_INT4=32
        #pragma unroll
        for (int blk_in_bk = 0; blk_in_bk < BK / QK_INT4; ++blk_in_bk) {
            FragC partial; wmma::fill_fragment(partial, 0.0f);
            int n_off = warp * 16;
            #pragma unroll
            for (int s = 0; s < SLICES_PER_BLOCK; ++s) {
                int kk = blk_in_bk * QK_INT4 + s * 16;
                FragA a; wmma::load_matrix_sync(a, sA + kk, BK);
                FragB b; wmma::load_matrix_sync(b, sB + kk + n_off * BK, BK);
                wmma::mma_sync(partial, a, b, partial);
            }
            float * tmp = sC + (size_t) warp * BM * 16;
            wmma::store_matrix_sync(tmp, partial, 16, wmma::mem_row_major);
            __syncwarp();
            int blk_idx_global = (k0 + blk_in_bk * QK_INT4) / QK_INT4;
            for (int idx = lane; idx < BM * 16; idx += 32) {
                int mm = idx / 16, nn = idx % 16;
                int gm = tile_m + mm, gn = tile_n + n_off + nn;
                if (gm < M && gn < N) {
                    float s = __half2float(W_scales[(size_t) gn * blocks_per_row + blk_idx_global]);
                    float rs = 0.0f;
                    #pragma unroll
                    for (int p = 0; p < QK_INT4; ++p) rs += __half2float(sA[mm * BK + blk_in_bk * QK_INT4 + p]);
                    tmp[idx] = (tmp[idx] - ANCHOR_BIAS * rs) * s;
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

}  // namespace tc_grid::kernels::int4_b

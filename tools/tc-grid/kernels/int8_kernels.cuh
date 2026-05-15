// INT8 tensor-core matmul kernels for V100 (sm_70).
//
// PATH-A (LUT):       int8 -> (float)q * scale -> __float2half. One FP multiply
//                     per element in the dequant; scale applied before WMMA.
//
// PATH-B (BITSHIFT):  Pack int8 q + 128 into low 8 bits of FP16 mantissa with
//                     anchor exponent 25 (binary 11001 = 0x6400 high bits).
//                     The resulting half represents (1152 + q) exactly.
//                     After mma_sync the accumulator holds
//                       acc[m,n] = sum_k a[m,k] * (1152 + q[n,k])
//                                = 1152 * row_sum_a[m] + true_dot[m,n]
//                     We precompute row_sum_a[m] = sum_k a[m,k] once per
//                     CTA (or pass in as kernel arg).  Per-block FP16 scale
//                     applied via SMEM round-trip per 16-K slice (one mma_sync
//                     per slice, scale columns of the partial fragment, then
//                     accumulate).  This costs 1 SMEM round-trip per (n,m,
//                     16-K-slice) but eliminates the per-element FP multiply
//                     in the dequant.

#pragma once

#include "tc_grid.h"

#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8 {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// ============================================================ PATH-A LUT ===
template <int BM_, int BN_, int BK_, int WARPS_>
__global__ void mm_int8_lut(
        const int8_t  * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    static_assert(BN / WARPS == 16, "BN/WARPS must equal 16");
    static_assert(BK % 16 == 0,    "BK multiple of 16");
    static_assert(BK % QK_INT8 == 0, "BK multiple of QK_INT8");

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
                int blocks_per_row = K / QK_INT8;
                int blk = gk / QK_INT8;
                float s = __half2float(W_scales[(size_t) gn * blocks_per_row + blk]);
                int8_t q = W_qs[(size_t) gn * K + gk];
                v = __float2half((float) q * s);
            }
            sB[kk + nn * BK] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a; wmma::load_matrix_sync(a, sA + kk, BK);
            int n_off = warp * 16;
            FragB b; wmma::load_matrix_sync(b, sB + kk + n_off * BK, BK);
            wmma::mma_sync(c_frag, a, b, c_frag);
        }
        __syncthreads();
    }

    float * tile_c = sC + warp * BM * 16;
    wmma::store_matrix_sync(tile_c, c_frag, 16, wmma::mem_row_major);
    __syncwarp();
    int n_off = warp * 16;
    for (int idx = lane; idx < BM * 16; idx += 32) {
        int mm = idx / 16, nn = idx % 16;
        int gm = tile_m + mm, gn = tile_n + n_off + nn;
        if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
    }
}

// ======================================================= PATH-B BITSHIFT ===
// Anchor exponent constant for FP16 mantissa packing of (q + 128):
//   bits = 0x6400 | ((q + 128) & 0xFF)
//   value = 1024 + (q + 128) = 1152 + q   (for q in [-128, 127])
__device__ __forceinline__ half int8_to_half_packed(int8_t q) {
    uint16_t bits = (uint16_t) 0x6400u | (uint16_t)((int) q + 128);
    return __ushort_as_half(bits);
}

template <int BM_, int BN_, int BK_, int WARPS_>
__global__ void mm_int8_bitshift(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    static_assert(BN / WARPS == 16, "BN/WARPS must equal 16");
    static_assert(BK % 16 == 0, "BK multiple of 16");
    static_assert(BK % QK_INT8 == 0, "BK multiple of QK_INT8");

    constexpr float ANCHOR_BIAS = 1152.0f;  // 1024 + 128

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
    // After sC[WARPS*BM*16 floats]: BM floats for per-slice row_sum_a.
    float  * sRow = sC + (size_t) WARPS * BM * 16;

    FragC c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int k_tiles = K / BK;
    constexpr int BLOCKS_PER_BK = BK / QK_INT8;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k0 = kt * BK;
        // Load A (cast to half) and compute per-(m, slice) running sums
        // for the 1152-anchor correction. We split BK into K_SLICE=16 pieces
        // matching the inner mma_sync chunk, so each slice gets its own bias.
        for (int idx = tid; idx < BM * BK; idx += threads) {
            int mm = idx / BK, kk = idx % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float v = (gm < M && gk < K) ? A[(size_t) gm * K + gk] : 0.0f;
            sA[mm * BK + kk] = __float2half(v);
        }
        // Load B as bit-packed halves (no scale yet).
        for (int idx = tid; idx < BN * BK; idx += threads) {
            int nn = idx / BK, kk = idx % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v = __float2half(0.0f);
            if (gn < N && gk < K) {
                int8_t q = W_qs[(size_t) gn * K + gk];
                v = int8_to_half_packed(q);
            }
            sB[kk + nn * BK] = v;
        }
        __syncthreads();

        const int blocks_per_row = K / QK_INT8;
        // ITEM 5: per-block (not per-slice) scaling.
        // Accumulate all (BK/16) WMMA slices that share the same QK_INT8 block
        // into ONE partial fragment, then apply scale + bias correction once
        // per block instead of once per 16-K slice. For BK=32 and QK_INT8=32
        // we have 2 slices per block → 2× fewer SMEM round-trips.
        constexpr int SLICES_PER_BLOCK = QK_INT8 / 16;  // 2 for QK_INT8=32
        #pragma unroll
        for (int blk_in_bk = 0; blk_in_bk < BK / QK_INT8; ++blk_in_bk) {
            FragC partial; wmma::fill_fragment(partial, 0.0f);
            int n_off = warp * 16;
            // Accumulate the SLICES_PER_BLOCK 16-K slices of this block.
            #pragma unroll
            for (int s = 0; s < SLICES_PER_BLOCK; ++s) {
                int kk = blk_in_bk * QK_INT8 + s * 16;
                FragA a; wmma::load_matrix_sync(a, sA + kk, BK);
                FragB b; wmma::load_matrix_sync(b, sB + kk + n_off * BK, BK);
                wmma::mma_sync(partial, a, b, partial);
            }
            float * tmp = sC + (size_t) warp * BM * 16;
            wmma::store_matrix_sync(tmp, partial, 16, wmma::mem_row_major);
            __syncwarp();
            int blk_idx_global = (k0 + blk_in_bk * QK_INT8) / QK_INT8;
            for (int idx = lane; idx < BM * 16; idx += 32) {
                int mm = idx / 16, nn = idx % 16;
                int gm = tile_m + mm, gn = tile_n + n_off + nn;
                if (gm < M && gn < N) {
                    float s = __half2float(W_scales[(size_t) gn * blocks_per_row + blk_idx_global]);
                    // Row sum over the ENTIRE block (QK_INT8 elements).
                    float rs = 0.0f;
                    #pragma unroll
                    for (int p = 0; p < QK_INT8; ++p) rs += __half2float(sA[mm * BK + blk_in_bk * QK_INT8 + p]);
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

}  // namespace tc_grid::kernels::int8

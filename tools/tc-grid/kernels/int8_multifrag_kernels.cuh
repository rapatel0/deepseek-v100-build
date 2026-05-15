// INT8 LUT, multi-N-fragment per warp.
//
// Each warp owns FRAG_N fragments of 16 cols each (so each warp covers
// FRAG_N*16 cols of N output). One a_frag is loaded per K-slice and is
// shared across FRAG_N mma_syncs targeting FRAG_N different b_frags.
// This amortises the a_frag load and packs FRAG_N* more tensor-core
// math per warp per K-slice.
//
// CTA covers (BM, BN) = (16, WARPS * FRAG_N * 16) of output.
// SMEM: sA[BM*BK], sB[BK*BN], sC[WARPS*BM*16] (reused per warp for the final
// store; per-warp output is built up via FRAG_N register c_frags).

#pragma once

#include "tc_grid.h"

#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8_mf {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_N_>
__global__ void mm_int8_lut_mf(
        const int8_t  * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_N = FRAG_N_;
    static_assert(BN == WARPS * FRAG_N * 16, "BN must equal WARPS * FRAG_N * 16");
    static_assert(BK % 16 == 0 && BK % QK_INT8 == 0, "BK constraints");
    static_assert(BM == 16, "BM must be 16 (single a_frag per warp)");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);            // [BM, BK]
    __half * sB = sA + BM * BK;                                    // [BK, BN] col-major
    float  * sC = reinterpret_cast<float *>(sB + BK * BN);         // [WARPS * BM * 16] reused

    FragC c_frag[FRAG_N];
    #pragma unroll
    for (int f = 0; f < FRAG_N; ++f) wmma::fill_fragment(c_frag[f], 0.0f);

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

        // Per 16-K slice: load 1 a_frag, FRAG_N b_frags, do FRAG_N mma_syncs.
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a; wmma::load_matrix_sync(a, sA + kk, BK);
            int n_off_warp = warp * FRAG_N * 16;
            #pragma unroll
            for (int f = 0; f < FRAG_N; ++f) {
                FragB b;
                wmma::load_matrix_sync(b, sB + kk + (n_off_warp + f * 16) * BK, BK);
                wmma::mma_sync(c_frag[f], a, b, c_frag[f]);
            }
        }
        __syncthreads();
    }

    // Store FRAG_N fragments per warp.
    #pragma unroll
    for (int f = 0; f < FRAG_N; ++f) {
        float * tile_c = sC + (size_t) warp * BM * 16;  // each warp reuses its slot
        wmma::store_matrix_sync(tile_c, c_frag[f], 16, wmma::mem_row_major);
        __syncwarp();
        int n_off = warp * FRAG_N * 16 + f * 16;
        for (int idx = lane; idx < BM * 16; idx += 32) {
            int mm = idx / 16, nn = idx % 16;
            int gm = tile_m + mm, gn = tile_n + n_off + nn;
            if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
        }
        __syncwarp();
    }
}

}  // namespace tc_grid::kernels::int8_mf

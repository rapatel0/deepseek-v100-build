// v7 = v3 base + Tier B3 (3-stage triple-buffered SMEM pipeline).
//
// v3 has 2 SMEM K-tile buffers (double-buffer). v7 uses 3 buffers, allowing
// 2 K-tiles to be in flight while compute runs on the third. Hides ~1 extra
// K-tile of gmem latency.
//
// SMEM cost: 1.5× of v3's double-buffer. Only fits at BM ≤ 64 BN ≤ 128 BK=32
// padded (≈ 42 KB, under V100's 48 KB limit).
//
// Pipeline:
//   prologue: load tile 0 → buf[0], load tile 1 → buf[1], sync
//   loop kt = 0..k_tiles-1:
//     if kt+2 < k_tiles: issue load tile kt+2 → buf[(kt+2) % 3]
//     mma_sync on buf[kt % 3]
//     __syncthreads()  // wait for tile kt+1 to be ready in buf[(kt+1) % 3]
//
// Bit-equivalent to v3 (same numerical sequence).

#pragma once

#include "tc_grid.h"
#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8_v7 {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__global__ void mm_int8_lut_v7(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int BK_PAD = BK + 8;
    constexpr int N_STAGES = 3;
    static_assert(BM == FRAG_M * 16, "BM == FRAG_M * 16");
    static_assert(N_PER_WARP == FRAG_N * 16, "N_PER_WARP == FRAG_N * 16");
    static_assert(BK % 16 == 0 && BK % QK_INT8 == 0, "BK constraints");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);                 // [3][BM][BK]
    __half * sB = sA + N_STAGES * BM * BK;                              // [3][BK_PAD][BN]
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK_PAD * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    auto load_tile = [&](int kt, int buf_idx) {
        if (kt >= k_tiles) return;
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
        const int k0 = kt * BK;
        constexpr int kA_chunks = (BM * BK) / 4;
        for (int c_ = tid; c_ < kA_chunks; c_ += threads) {
            int idx0 = c_ * 4;
            int mm = idx0 / BK, kk = idx0 % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float4 v = make_float4(0, 0, 0, 0);
            if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        constexpr int kB_chunks = (BN * BK) / 16;
        for (int c_ = tid; c_ < kB_chunks; c_ += threads) {
            int idx0 = c_ * 16;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            int8_t qs[16] = {0};
            if (gn < N && gk + 15 < K) {
                ::int4 vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                *(::int4 *)&qs[0] = vq;
            }
            float s = (gn < N) ? __half2float(__ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8))) : 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                half v = __float2half((float) qs[i] * s);
                sB_b[kk + i + nn * BK_PAD] = v;
            }
        }
    };

    // Prologue: load tiles 0 and 1.
    load_tile(0, 0);
    load_tile(1, 1);
    __syncthreads();

    // Main loop. Compute on buf[kt%3], pre-load buf[(kt+2)%3].
    for (int kt = 0; kt < k_tiles; ++kt) {
        // Issue load for tile kt+2 if it exists.
        load_tile(kt + 2, (kt + 2) % N_STAGES);

        __half * sA_c = sA_buf(kt % N_STAGES);
        __half * sB_c = sB_buf(kt % N_STAGES);
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a[FRAG_M];
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; ++fm)
                wmma::load_matrix_sync(a[fm], &sA_c[fm * 16 * BK + kk], BK);
            FragB b[FRAG_N];
            #pragma unroll
            for (int fn = 0; fn < FRAG_N; ++fn)
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; ++fm)
                #pragma unroll
                for (int fn = 0; fn < FRAG_N; ++fn)
                    wmma::mma_sync(c[fm][fn], a[fm], b[fn], c[fm][fn]);
        }
        __syncthreads();
    }

    // Store. Reuse first buffer slot's sA as scratch.
    float * sC_scratch = reinterpret_cast<float *>(sA);
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm) {
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn) {
            float * tile_c = sC_scratch + warp * 16 * 16;
            wmma::store_matrix_sync(tile_c, c[fm][fn], 16, wmma::mem_row_major);
            __syncwarp();
            int n_off = n_off_warp + fn * 16;
            int m_off = fm * 16;
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                int mm = idx / 16, nn = idx % 16;
                int gm = tile_m + m_off + mm, gn = tile_n + n_off + nn;
                if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
            }
            __syncwarp();
        }
    }
}

}  // namespace int8_v7

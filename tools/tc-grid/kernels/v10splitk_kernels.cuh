// v10s = v10 row-major B + SplitK accumulation across blockIdx.z.
//
// Mirrors v10 except:
//   - Extra template param `int KS_` (K-split factor, compile-time).
//   - K-loop bounds keyed off blockIdx.z; each CTA processes K / (BK * KS) tiles.
//   - Epilogue uses atomicAdd into C when KS > 1, store when KS == 1. The KS==1
//     path is bit-identical to v10 (sanity check).
//
// Caller contract: when KS > 1, C must be pre-zeroed (cudaMemsetAsync) BEFORE
// launching. Grid dim is (N/BN, M/BM, KS).
//
// Why ship this (per SPRINT-017 Wave 1 A0): turbomind ships SplitK=true in every
// sm_70 config. The infrastructure unlocks tile shapes where per-CTA grid is
// small enough that SplitK boosts SM occupancy, regardless of M=2048's
// already-saturated grid. Reusable for v11 + INT4/FP4 variants.

#pragma once

#include "tc_grid.h"
#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8_v10s {

__device__ __forceinline__ void prefetch_l2(const void * ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_, int KS_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int8_lut_v10s(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int KS = KS_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int BN_PAD = BN + 8;
    static_assert(BM == FRAG_M * 16, "BM == FRAG_M * 16");
    static_assert(N_PER_WARP == FRAG_N * 16, "N_PER_WARP == FRAG_N * 16");
    static_assert(BK % 16 == 0 && BK % QK_INT8 == 0, "BK constraints");
    static_assert(BN_PAD % 8 == 0, "WMMA ldm must be multiple of 8");
    static_assert(KS >= 1, "KS must be >= 1");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int k_split = blockIdx.z;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK * BN_PAD; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int k_tiles_total = K / BK;
    // K-split: each blockIdx.z handles a contiguous chunk of k_tiles_total.
    // Distribute evenly; remainder goes to early splits.
    const int kt_base = (k_tiles_total / KS) * k_split + min(k_split, k_tiles_total % KS);
    const int kt_lim  = kt_base + (k_tiles_total / KS) + ((k_split < (k_tiles_total % KS)) ? 1 : 0);
    const int blocks_per_row = K / QK_INT8;

    auto load_tile = [&](int kt, int buf_idx, int prefetch_kt) {
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
        const int k0 = kt * BK;
        if (prefetch_kt < kt_lim && tid == 0) {
            const int pk0 = prefetch_kt * BK;
            prefetch_l2(&A[(size_t)(tile_m) * K + pk0]);
            prefetch_l2(&W_qs[(size_t)(tile_n) * K + pk0]);
            prefetch_l2(&W_scales[(size_t)(tile_n) * blocks_per_row + (pk0 / QK_INT8)]);
        }
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
            half s_h = (gn < N) ? __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8)) : __float2half(0.0f);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                half v = __hmul(__short2half_rn((short) qs[i]), s_h);
                sB_b[(kk + i) * BN_PAD + nn] = v;
            }
        }
    };

    // Guard: if this k_split has zero tiles (KS > k_tiles_total), bail early.
    if (kt_base >= kt_lim) return;

    load_tile(kt_base, 0, kt_base + 1);
    __syncthreads();
    int buf = 0;
    for (int kt = kt_base; kt < kt_lim - 1; ++kt) {
        load_tile(kt + 1, 1 - buf, kt + 2);
        __half * sA_c = sA_buf(buf);
        __half * sB_c = sB_buf(buf);
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a[FRAG_M];
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; ++fm)
                wmma::load_matrix_sync(a[fm], &sA_c[fm * 16 * BK + kk], BK);
            FragB b[FRAG_N];
            #pragma unroll
            for (int fn = 0; fn < FRAG_N; ++fn)
                wmma::load_matrix_sync(b[fn], &sB_c[kk * BN_PAD + (n_off_warp + fn * 16)], BN_PAD);
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; ++fm)
                #pragma unroll
                for (int fn = 0; fn < FRAG_N; ++fn)
                    wmma::mma_sync(c[fm][fn], a[fm], b[fn], c[fm][fn]);
        }
        __syncthreads();
        buf = 1 - buf;
    }
    {
        __half * sA_c = sA_buf(buf);
        __half * sB_c = sB_buf(buf);
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a[FRAG_M];
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; ++fm)
                wmma::load_matrix_sync(a[fm], &sA_c[fm * 16 * BK + kk], BK);
            FragB b[FRAG_N];
            #pragma unroll
            for (int fn = 0; fn < FRAG_N; ++fn)
                wmma::load_matrix_sync(b[fn], &sB_c[kk * BN_PAD + (n_off_warp + fn * 16)], BN_PAD);
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; ++fm)
                #pragma unroll
                for (int fn = 0; fn < FRAG_N; ++fn)
                    wmma::mma_sync(c[fm][fn], a[fm], b[fn], c[fm][fn]);
        }
    }
    __syncthreads();
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
                if (gm < M && gn < N) {
                    if (KS == 1) {
                        C[(size_t) gm * N + gn] = tile_c[idx];
                    } else {
                        atomicAdd(&C[(size_t) gm * N + gn], tile_c[idx]);
                    }
                }
            }
            __syncwarp();
        }
    }
}

}  // namespace int8_v10s

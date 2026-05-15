// v4 = v3 + Tier S (items S1+S2+S3).
//
// S1 — L2 prefetch hints: inline PTX `prefetch.global.L2` issued one K-tile
//      AHEAD of the actual `ld.global`. The double-buffer in v3 already
//      issues tile k+1 while computing tile k; v4 adds an L2 prefetch for
//      tile k+2 (or k+1's NEXT chunk) so the actual load hits L2 cache
//      instead of HBM.
// S2 — __launch_bounds__: declared per kernel to constrain nvcc register
//      count for target occupancy (2 CTAs / SM, total 256 active warps).
// S3 — __ldg: weights (W_qs, W_scales, W_blocks) are read-only across all
//      threads/CTAs. `__ldg()` routes them through the read-only data
//      cache (4 KB / SM) which has separate bandwidth from L1.
//
// These are pure scheduling/hint optimizations -- numerical output MUST be
// bit-identical to v3. Correctness check: per-cell max_abs / p99 / rel_err
// against the FP32 reference must MATCH v3's values exactly (down to last
// digit at the same shape).

#pragma once

#include "tc_grid.h"
#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8_v4 {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void prefetch_l2(const void * ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int8_lut_v4(
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
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;
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

    auto load_tile = [&](int kt, int buf_idx, int prefetch_kt) {
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
        const int k0 = kt * BK;

        // S1: prefetch one K-tile ahead. Issue prefetches for the row
        // anchors of the FUTURE tile so its data is warm in L2 by the time
        // we issue its actual load.
        if (prefetch_kt < k_tiles) {
            const int pk0 = prefetch_kt * BK;
            // 1 prefetch per warp per CTA covers the row anchors we'll need.
            if (tid == 0) {
                prefetch_l2(&A[(size_t)(tile_m) * K + pk0]);
                prefetch_l2(&W_qs[(size_t)(tile_n) * K + pk0]);
                prefetch_l2(&W_scales[(size_t)(tile_n) * blocks_per_row + (pk0 / QK_INT8)]);
            }
        }

        // S3: __ldg for read-only weight reads.
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
                // S3: __ldg for weight bytes. Read 16 bytes via int4 vector
                // through RO cache.
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

    // Prologue: load tile 0, prefetch tile 1.
    load_tile(0, 0, 1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        // Load tile kt+1, prefetch tile kt+2.
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
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
                if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
            }
            __syncwarp();
        }
    }
}

}  // namespace int8_v4

// =============================== INT4 / MXFP4 / F8 v4 LUT — same pattern ===
// (Identical kernel architecture as v3 + Tier S items applied to the
//  format-specific dequant/load section.)

namespace tc_grid::kernels::int4_v4 {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void prefetch_l2(const void * ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int4_lut_v4(
        const uint8_t * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int BK_PAD = BK + 8;

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK_PAD * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int K_pad_bytes = K / 2;
    const int blocks_per_row = K / QK_INT4;
    const int k_tiles = K / BK;

    auto load_tile = [&](int kt, int buf_idx, int prefetch_kt) {
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
        const int k0 = kt * BK;
        if (prefetch_kt < k_tiles && tid == 0) {
            const int pk0 = prefetch_kt * BK;
            prefetch_l2(&A[(size_t)(tile_m) * K + pk0]);
            prefetch_l2(&W_qs[(size_t)(tile_n) * K_pad_bytes + pk0 / 2]);
            prefetch_l2(&W_scales[(size_t)(tile_n) * blocks_per_row + (pk0 / QK_INT4)]);
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
        for (int idx = tid; idx < BN * BK / 2; idx += threads) {
            int idx0 = idx * 2;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v0 = __float2half(0.0f), v1 = __float2half(0.0f);
            if (gn < N && gk + 1 < K) {
                uint8_t b = __ldg(W_qs + (size_t) gn * K_pad_bytes + gk / 2);
                int8_t n_lo = (int8_t)((b & 0xF) << 4) >> 4;
                int8_t n_hi = (int8_t)(b & 0xF0)        >> 4;
                float s = __half2float(__ldg(W_scales + (size_t) gn * blocks_per_row + gk / QK_INT4));
                v0 = __float2half((float) n_lo * s);
                v1 = __float2half((float) n_hi * s);
            }
            sB_b[kk     + nn * BK_PAD] = v0;
            sB_b[kk + 1 + nn * BK_PAD] = v1;
        }
    };

    load_tile(0, 0, 1);
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
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
                if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
            }
            __syncwarp();
        }
    }
}

}  // namespace int4_v4

namespace tc_grid::kernels::fp4_v4 {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void prefetch_l2(const void * ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}
__device__ __forceinline__ float e8m0_to_f32_dev(uint8_t e) {
    return __int_as_float(((int) e) << 23);
}

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_mxfp4_lut_v4(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int BK_PAD = BK + 8;
    constexpr int kBlkBytes = 1 + (QK_MXFP4 / 2);

    static const uint16_t kvals_h16[8] = {
        0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    };

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK_PAD * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int blocks_per_row = K / QK_MXFP4;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles = K / BK;

    auto load_tile = [&](int kt, int buf_idx, int prefetch_kt) {
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
        const int k0 = kt * BK;
        if (prefetch_kt < k_tiles && tid == 0) {
            const int pk0 = prefetch_kt * BK;
            prefetch_l2(&A[(size_t)(tile_m) * K + pk0]);
            int pblk = pk0 / QK_MXFP4;
            prefetch_l2(&W_blocks[(size_t)(tile_n) * W_row_bytes + pblk * kBlkBytes]);
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
        for (int idx = tid; idx < BN * BK / 8; idx += threads) {
            int idx0 = idx * 8;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half hh[8] = {(half) 0, (half) 0, (half) 0, (half) 0,
                          (half) 0, (half) 0, (half) 0, (half) 0};
            if (gn < N && gk + 7 < K) {
                int blk = gk / QK_MXFP4;
                int in_blk = gk - blk * QK_MXFP4;
                const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                const uint8_t * bptr = row + blk * kBlkBytes;
                uint8_t e = __ldg(bptr);
                float d_half = e8m0_to_f32_dev(e) * 0.5f;
                half d = __float2half(d_half);
                uint32_t code4;
                __builtin_memcpy(&code4, &bptr[1 + (in_blk >> 1)], 4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint8_t code = (code4 >> (i * 8)) & 0xFFu;
                    uint8_t n_lo = code & 0xF;
                    uint8_t n_hi = code >> 4;
                    half v_lo = __ushort_as_half(kvals_h16[n_lo & 7]);
                    half v_hi = __ushort_as_half(kvals_h16[n_hi & 7]);
                    if (n_lo & 8) v_lo = __hneg(v_lo);
                    if (n_hi & 8) v_hi = __hneg(v_hi);
                    hh[i * 2 + 0] = __hmul(v_lo, d);
                    hh[i * 2 + 1] = __hmul(v_hi, d);
                }
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) sB_b[kk + i + nn * BK_PAD] = hh[i];
        }
    };

    load_tile(0, 0, 1);
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
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
                if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
            }
            __syncwarp();
        }
    }
}

}  // namespace fp4_v4

namespace tc_grid::kernels::fp8_v4 {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void prefetch_l2(const void * ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}
__device__ __forceinline__ float e8m0_to_f32_dev(uint8_t e) {
    return __int_as_float(((int) e) << 23);
}
__device__ __forceinline__ float e4m3fn_to_f32(uint8_t b) {
    bool neg = (b & 0x80) != 0;
    int E = (b >> 3) & 0xF;
    int M = b & 0x7;
    float v;
    if (E == 0) v = ldexpf((float) M / 8.0f, -6);
    else if (E == 15 && M == 7) v = 0.0f;
    else v = ldexpf(1.0f + (float) M / 8.0f, E - 7);
    return neg ? -v : v;
}

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_f8_e4m3_b128_lut_v4(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int BK_PAD = BK + 8;
    constexpr int kBlkBytes = 1 + QK_F8;

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK_PAD * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int blocks_per_row = K / QK_F8;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles = K / BK;

    auto load_tile = [&](int kt, int buf_idx, int prefetch_kt) {
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
        const int k0 = kt * BK;
        if (prefetch_kt < k_tiles && tid == 0) {
            const int pk0 = prefetch_kt * BK;
            prefetch_l2(&A[(size_t)(tile_m) * K + pk0]);
            int pblk = pk0 / QK_F8;
            prefetch_l2(&W_blocks[(size_t)(tile_n) * W_row_bytes + pblk * kBlkBytes]);
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
        for (int idx = tid; idx < BN * BK / 4; idx += threads) {
            int idx0 = idx * 4;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half hh[4] = {(half) 0, (half) 0, (half) 0, (half) 0};
            if (gn < N && gk + 3 < K) {
                int blk = gk / QK_F8;
                int in_blk = gk - blk * QK_F8;
                const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                const uint8_t * bptr = row + blk * kBlkBytes;
                uint8_t e = __ldg(bptr);
                float d_f = e8m0_to_f32_dev(e);
                uint32_t bytes4;
                __builtin_memcpy(&bytes4, &bptr[1 + in_blk], 4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint8_t q = (bytes4 >> (i * 8)) & 0xFFu;
                    hh[i] = __float2half(d_f * e4m3fn_to_f32(q));
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) sB_b[kk + i + nn * BK_PAD] = hh[i];
        }
    };

    load_tile(0, 0, 1);
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK_PAD], BK_PAD);
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
                if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
            }
            __syncwarp();
        }
    }
}

}  // namespace fp8_v4

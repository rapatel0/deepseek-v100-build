// v3s = v3 base + SplitK (k-direction CTAs with atomicAdd into C).
//
// Three format variants share the same SplitK pattern; only the dequant
// front-end differs (nibble unpack for INT4, PRMT-based decode for MXFP4,
// E4M3 decode for F8). Each is a thin wrapper around the corresponding
// v3 kernel structure.
//
// Caller contract: when KS > 1, C must be pre-zeroed (cudaMemsetAsync)
// BEFORE launching. Grid dim is (N/BN, M/BM, KS).
//
// KS=1 path is bit-identical to v3 (no atomic, same k-loop bounds).
//
// Per SPRINT-017 mechanical-port phase: v10s SplitK delivered +84% at M=64
// for INT8. Same infrastructure benefit expected for INT4/MXFP4/F8 small-M.

#pragma once

#include "tc_grid.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================ INT4 v3s ===
namespace tc_grid::kernels::int4_v3s {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_, int KS_>
__global__ void mm_int4_lut_v3s(
        const uint8_t * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int KS = KS_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int BK_PAD = BK + 8;
    static_assert(BM == FRAG_M * 16);
    static_assert(N_PER_WARP == FRAG_N * 16);
    static_assert(KS >= 1);

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
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK_PAD * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int K_pad_bytes = K / 2;
    const int blocks_per_row = K / QK_INT4;
    const int k_tiles_total = K / BK;
    const int kt_base = (k_tiles_total / KS) * k_split + min(k_split, k_tiles_total % KS);
    const int kt_lim  = kt_base + (k_tiles_total / KS) + ((k_split < (k_tiles_total % KS)) ? 1 : 0);

    auto load_tile = [&](int kt, int buf_idx) {
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
        for (int idx = tid; idx < BN * BK / 2; idx += threads) {
            int idx0 = idx * 2;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v0 = __float2half(0.0f), v1 = __float2half(0.0f);
            if (gn < N && gk + 1 < K) {
                uint8_t b = W_qs[(size_t) gn * K_pad_bytes + gk / 2];
                int8_t n_lo = (int8_t)((b & 0xF) << 4) >> 4;
                int8_t n_hi = (int8_t)(b & 0xF0)        >> 4;
                float s = __half2float(W_scales[(size_t) gn * blocks_per_row + gk / QK_INT4]);
                v0 = __float2half((float) n_lo * s);
                v1 = __float2half((float) n_hi * s);
            }
            sB_b[kk     + nn * BK_PAD] = v0;
            sB_b[kk + 1 + nn * BK_PAD] = v1;
        }
    };

    if (kt_base >= kt_lim) return;

    load_tile(kt_base, 0);
    __syncthreads();
    int buf = 0;
    for (int kt = kt_base; kt < kt_lim - 1; ++kt) {
        load_tile(kt + 1, 1 - buf);
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
                if (gm < M && gn < N) {
                    if (KS == 1) C[(size_t) gm * N + gn] = tile_c[idx];
                    else         atomicAdd(&C[(size_t) gm * N + gn], tile_c[idx]);
                }
            }
            __syncwarp();
        }
    }
}

}  // namespace int4_v3s

// =========================================================== MXFP4 v3s ===
namespace tc_grid::kernels::fp4_v3s {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float e8m0_to_f32_dev_s(uint8_t e) {
    return __int_as_float(((int) e) << 23);
}

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_, int KS_>
__global__ void mm_mxfp4_lut_v3s(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int KS = KS_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int BK_PAD = BK + 8;
    constexpr int kBlkBytes = 1 + (QK_MXFP4 / 2);
    static_assert(BM == FRAG_M * 16);
    static_assert(N_PER_WARP == FRAG_N * 16);
    static_assert(KS >= 1);

    static const uint16_t kvals_h16[8] = {
        0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    };

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
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK_PAD * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int blocks_per_row = K / QK_MXFP4;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles_total = K / BK;
    const int kt_base = (k_tiles_total / KS) * k_split + min(k_split, k_tiles_total % KS);
    const int kt_lim  = kt_base + (k_tiles_total / KS) + ((k_split < (k_tiles_total % KS)) ? 1 : 0);

    auto load_tile = [&](int kt, int buf_idx) {
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
                uint8_t e = bptr[0];
                float d_half = e8m0_to_f32_dev_s(e) * 0.5f;
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

    if (kt_base >= kt_lim) return;

    load_tile(kt_base, 0);
    __syncthreads();
    int buf = 0;
    for (int kt = kt_base; kt < kt_lim - 1; ++kt) {
        load_tile(kt + 1, 1 - buf);
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
                if (gm < M && gn < N) {
                    if (KS == 1) C[(size_t) gm * N + gn] = tile_c[idx];
                    else         atomicAdd(&C[(size_t) gm * N + gn], tile_c[idx]);
                }
            }
            __syncwarp();
        }
    }
}

}  // namespace fp4_v3s

// ============================================================== F8 v3s ===
namespace tc_grid::kernels::fp8_v3s {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float e8m0_to_f32_dev_fs(uint8_t e) {
    return __int_as_float(((int) e) << 23);
}
__device__ __forceinline__ float e4m3fn_to_f32_s(uint8_t b) {
    bool neg = (b & 0x80) != 0;
    int E = (b >> 3) & 0xF;
    int Mant = b & 0x7;
    float v;
    if (E == 0)                    v = ldexpf((float) Mant / 8.0f, -6);
    else if (E == 15 && Mant == 7) v = 0.0f;
    else                           v = ldexpf(1.0f + (float) Mant / 8.0f, E - 7);
    return neg ? -v : v;
}

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_, int KS_>
__global__ void mm_f8_e4m3_b128_lut_v3s(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int KS = KS_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int BK_PAD = BK + 8;
    constexpr int kBlkBytes = 1 + QK_F8;
    static_assert(BM == FRAG_M * 16);
    static_assert(N_PER_WARP == FRAG_N * 16);
    static_assert(KS >= 1);

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
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK_PAD * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int blocks_per_row = K / QK_F8;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles_total = K / BK;
    const int kt_base = (k_tiles_total / KS) * k_split + min(k_split, k_tiles_total % KS);
    const int kt_lim  = kt_base + (k_tiles_total / KS) + ((k_split < (k_tiles_total % KS)) ? 1 : 0);

    auto load_tile = [&](int kt, int buf_idx) {
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
                uint8_t e = bptr[0];
                float d_f = e8m0_to_f32_dev_fs(e);
                uint32_t bytes4;
                __builtin_memcpy(&bytes4, &bptr[1 + in_blk], 4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint8_t q = (bytes4 >> (i * 8)) & 0xFFu;
                    hh[i] = __float2half(d_f * e4m3fn_to_f32_s(q));
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) sB_b[kk + i + nn * BK_PAD] = hh[i];
        }
    };

    if (kt_base >= kt_lim) return;

    load_tile(kt_base, 0);
    __syncthreads();
    int buf = 0;
    for (int kt = kt_base; kt < kt_lim - 1; ++kt) {
        load_tile(kt + 1, 1 - buf);
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
                if (gm < M && gn < N) {
                    if (KS == 1) C[(size_t) gm * N + gn] = tile_c[idx];
                    else         atomicAdd(&C[(size_t) gm * N + gn], tile_c[idx]);
                }
            }
            __syncwarp();
        }
    }
}

}  // namespace fp8_v3s

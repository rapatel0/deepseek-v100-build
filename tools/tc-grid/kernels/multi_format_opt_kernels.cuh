// v2-style optimized LUT kernels for INT4, MXFP4, F8 E4M3 B128.
// Same kernel architecture as int8_opt::mm_int8_lut_v2:
//   - BM=128 (or 64) multi-A-fragment per warp
//   - Double-buffered SMEM K-tile pipeline
//   - uint4 vectorized A loads (16 bytes → 8 halves)
//   - Format-specific B load + dequant in SMEM
//
// Each kernel follows the same skeleton; the only delta is the SMEM B load
// section.

#pragma once

#include "tc_grid.h"
#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int4_opt {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__global__ void mm_int4_lut_v2(
        const uint8_t * __restrict__ W_qs,        // 2 nibbles/byte, K/2 bytes per row
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    static_assert(BM == FRAG_M * 16, "BM == FRAG_M * 16");
    static_assert(N_PER_WARP == FRAG_N * 16, "N_PER_WARP == FRAG_N * 16");
    static_assert(BK % 16 == 0, "BK % 16");
    static_assert(BK % QK_INT4 == 0, "BK % QK_INT4");

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
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int K_pad_bytes = K / 2;
    const int blocks_per_row = K / QK_INT4;
    const int k_tiles = K / BK;

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
            half2 h01 = __floats2half2_rn(v.x, v.y);
            half2 h23 = __floats2half2_rn(v.z, v.w);
            *(half2 *) &sA_b[mm * BK + kk    ] = h01;
            *(half2 *) &sA_b[mm * BK + kk + 2] = h23;
        }
        // B load: each thread handles 2 nibbles (1 byte) at a time.
        for (int idx = tid; idx < BN * BK / 2; idx += threads) {
            int idx0 = idx * 2;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v0 = __float2half(0.0f), v1 = __float2half(0.0f);
            if (gn < N && gk + 1 < K) {
                int byte_idx = gk / 2;
                uint8_t b = W_qs[(size_t) gn * K_pad_bytes + byte_idx];
                int8_t n_lo = (int8_t)((b & 0xF) << 4) >> 4;
                int8_t n_hi = (int8_t)(b & 0xF0)        >> 4;
                int blk = gk / QK_INT4;
                float s = __half2float(W_scales[(size_t) gn * blocks_per_row + blk]);
                v0 = __float2half((float) n_lo * s);
                v1 = __float2half((float) n_hi * s);
            }
            sB_b[kk     + nn * BK] = v0;
            sB_b[kk + 1 + nn * BK] = v1;
        }
    };

    load_tile(0, 0);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK], BK);
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK], BK);
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

}  // namespace tc_grid::kernels::int4_opt

namespace tc_grid::kernels::fp4_opt {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float e8m0_to_f32_dev_fp4(uint8_t e) {
    return __int_as_float(((int) e) << 23);
}

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__global__ void mm_mxfp4_lut_v2(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    static_assert(BM == FRAG_M * 16, "BM == FRAG_M * 16");
    static_assert(N_PER_WARP == FRAG_N * 16, "N_PER_WARP == FRAG_N * 16");
    static_assert(BK % QK_MXFP4 == 0, "BK % QK_MXFP4");

    constexpr float kvals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    constexpr int kBlkBytes = 1 + (QK_MXFP4 / 2);

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
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int blocks_per_row = K / QK_MXFP4;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles = K / BK;

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
            half2 h01 = __floats2half2_rn(v.x, v.y);
            half2 h23 = __floats2half2_rn(v.z, v.w);
            *(half2 *) &sA_b[mm * BK + kk    ] = h01;
            *(half2 *) &sA_b[mm * BK + kk + 2] = h23;
        }
        // B: each thread handles 2 nibbles (one byte) at a time.
        for (int idx = tid; idx < BN * BK / 2; idx += threads) {
            int idx0 = idx * 2;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v0 = __float2half(0.0f), v1 = __float2half(0.0f);
            if (gn < N && gk + 1 < K) {
                int blk = gk / QK_MXFP4;
                int in_blk = gk - blk * QK_MXFP4;
                const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                const uint8_t * bptr = row + blk * kBlkBytes;
                uint8_t e = bptr[0];
                uint8_t code = bptr[1 + (in_blk >> 1)];
                uint8_t n_lo = code & 0xF;
                uint8_t n_hi = code >> 4;
                float d = e8m0_to_f32_dev_fp4(e) * 0.5f;
                v0 = __float2half(kvals[n_lo & 7] * ((n_lo & 8) ? -1.0f : 1.0f) * d);
                v1 = __float2half(kvals[n_hi & 7] * ((n_hi & 8) ? -1.0f : 1.0f) * d);
            }
            sB_b[kk     + nn * BK] = v0;
            sB_b[kk + 1 + nn * BK] = v1;
        }
    };

    load_tile(0, 0);
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK], BK);
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK], BK);
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

}  // namespace tc_grid::kernels::fp4_opt

namespace tc_grid::kernels::fp8_opt {

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float e8m0_to_f32_dev_fp8(uint8_t e) {
    return __int_as_float(((int) e) << 23);
}

__device__ __forceinline__ float e4m3fn_to_f32_dev_v2(uint8_t b) {
    bool neg = (b & 0x80) != 0;
    int  E   = (b >> 3) & 0xF;
    int  M   = b & 0x7;
    float v;
    if (E == 0) v = ldexpf((float) M / 8.0f, -6);
    else if (E == 15 && M == 7) v = 0.0f;
    else v = ldexpf(1.0f + (float) M / 8.0f, E - 7);
    return neg ? -v : v;
}

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__global__ void mm_f8_e4m3_b128_lut_v2(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    static_assert(BM == FRAG_M * 16, "BM == FRAG_M * 16");
    static_assert(N_PER_WARP == FRAG_N * 16, "N_PER_WARP == FRAG_N * 16");

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
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int blocks_per_row = K / QK_F8;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles = K / BK;

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
            half2 h01 = __floats2half2_rn(v.x, v.y);
            half2 h23 = __floats2half2_rn(v.z, v.w);
            *(half2 *) &sA_b[mm * BK + kk    ] = h01;
            *(half2 *) &sA_b[mm * BK + kk + 2] = h23;
        }
        for (int idx = tid; idx < BN * BK; idx += threads) {
            int nn = idx / BK, kk = idx % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v = __float2half(0.0f);
            if (gn < N && gk < K) {
                int blk = gk / QK_F8;
                int in_blk = gk - blk * QK_F8;
                const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                const uint8_t * bptr = row + blk * kBlkBytes;
                uint8_t e = bptr[0];
                uint8_t q = bptr[1 + in_blk];
                float d = e8m0_to_f32_dev_fp8(e);
                v = __float2half(d * e4m3fn_to_f32_dev_v2(q));
            }
            sB_b[kk + nn * BK] = v;
        }
    };

    load_tile(0, 0);
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK], BK);
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
                wmma::load_matrix_sync(b[fn], &sB_c[kk + (n_off_warp + fn * 16) * BK], BK);
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

}  // namespace tc_grid::kernels::fp8_opt

// INT8 LUT optimized kernel variants for V100 tensor-core saturation.
//
// Naming convention: mm_int8_lut_v{N} where N is the cumulative optimization
// stack applied:
//   v1: BM=128 multi-A-fragment per warp. Each warp owns FRAG_M=8 A-fragments
//       (full BM=128 rows) and FRAG_N N-fragments. Per K-slice: load 8 a_frags
//       + FRAG_N b_frags, do FRAG_M*FRAG_N mma_syncs. A-fragment LOAD is fully
//       amortized across FRAG_N mma_syncs, B-fragment LOAD across FRAG_M.
//   v2: + double-buffered SMEM K-tile pipelining (item 2). 2x SMEM, overlap
//       next gmem load with current mma_sync.
//   v3: + uint4-vectorized SMEM loads/stores (item 3).
//   v4: + XOR-swizzled B layout to break SMEM bank conflicts (item 4).

#pragma once

#include "tc_grid.h"

#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8_opt {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// ============================================================ v1: BM=128 ===

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__global__ void mm_int8_lut_v1(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    static_assert(BM == FRAG_M * 16, "BM == FRAG_M * 16");
    static_assert(N_PER_WARP == FRAG_N * 16, "N_PER_WARP == FRAG_N * 16");
    static_assert(BK % 16 == 0, "BK % 16");
    static_assert(BK % QK_INT8 == 0, "BK % QK_INT8");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + BM * BK;

    // FRAG_M × FRAG_N register accumulators. For BM=128, FRAG_M=8 ⇒ 16 frags
    // per warp at FRAG_N=2. Each FragC has 8 float elements ⇒ 128 floats per
    // warp = 4 floats per thread. Fine for register budget.
    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k0 = kt * BK;
        // Load A [BM, BK] (scalar; v3 will vectorize)
        for (int idx = tid; idx < BM * BK; idx += threads) {
            int mm = idx / BK, kk = idx % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float v = (gm < M && gk < K) ? A[(size_t) gm * K + gk] : 0.0f;
            sA[mm * BK + kk] = __float2half(v);
        }
        // Load B [BN, BK] col-major (apply scale during load)
        for (int idx = tid; idx < BN * BK; idx += threads) {
            int nn = idx / BK, kk = idx % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v = __float2half(0.0f);
            if (gn < N && gk < K) {
                int blk = gk / QK_INT8;
                float s = __half2float(W_scales[(size_t) gn * blocks_per_row + blk]);
                int8_t q = W_qs[(size_t) gn * K + gk];
                v = __float2half((float) q * s);
            }
            sB[kk + nn * BK] = v;
        }
        __syncthreads();

        // Per K-slice: load FRAG_M a_frags, FRAG_N b_frags, do FRAG_M*FRAG_N mma_syncs.
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a[FRAG_M];
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; ++fm) {
                wmma::load_matrix_sync(a[fm], &sA[fm * 16 * BK + kk], BK);
            }
            FragB b[FRAG_N];
            #pragma unroll
            for (int fn = 0; fn < FRAG_N; ++fn) {
                wmma::load_matrix_sync(b[fn], &sB[kk + (n_off_warp + fn * 16) * BK], BK);
            }
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; ++fm) {
                #pragma unroll
                for (int fn = 0; fn < FRAG_N; ++fn) {
                    wmma::mma_sync(c[fm][fn], a[fm], b[fn], c[fm][fn]);
                }
            }
        }
        __syncthreads();
    }

    // Store FRAG_M*FRAG_N fragments to C. Reuse sA as float scratch
    // (K-loop done, SMEM contents no longer needed).
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

// =================================== v2: + double-buffer + uint4 loads ===
//
// Adds items 2 (software pipelining) and 3 (uint4 vectorized SMEM stores)
// on top of v1's BM=128 multi-A-fragment.
//
// SMEM layout: sA[2][BM][BK], sB[2][BK][BN] (double-buffered).
// While compute uses buf[i%2], the next K-tile's gmem loads target buf[(i+1)%2].
//
// uint4 path: each thread loads 16 bytes (4 floats from A, or 16 int8 quants
// from W) per transaction. Converted to 8 halves and stored as one uint4 to
// SMEM. ~4× fewer SMEM transactions than scalar.

template <int BM_, int BN_, int BK_, int WARPS_, int FRAG_M_, int FRAG_N_>
__global__ void mm_int8_lut_v2(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int FRAG_M = FRAG_M_, FRAG_N = FRAG_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    static_assert(BM == FRAG_M * 16, "BM == FRAG_M * 16");
    static_assert(N_PER_WARP == FRAG_N * 16, "N_PER_WARP == FRAG_N * 16");
    static_assert(BK % 16 == 0, "BK % 16");
    static_assert(BK % QK_INT8 == 0, "BK % QK_INT8");
    // For uint4 path: BM * BK must be a multiple of 8 (8 halves per uint4).
    static_assert((BM * BK) % 8 == 0, "BM*BK must be uint4-aligned");
    static_assert((BN * BK) % 8 == 0, "BN*BK must be uint4-aligned");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;          // double-buffered

    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BK * BN; };

    FragC c[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; ++fm)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; ++fn)
            wmma::fill_fragment(c[fm][fn], 0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    // Load A[0]/B[0] (scalar with uint4 hint).
    auto load_tile = [&](int kt, int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
        const int k0 = kt * BK;

        // ---- A tile [BM, BK]: load 4 floats per uint4, cast to 4 halves, store as 8-byte half2 pair (no native uint4 half store on V100).
        //      For BM*BK halves: total elements / 4 = (BM*BK)/4 uint128 spans.
        //      Each thread handles `(BM*BK)/4 / threads` chunks of 4 elements.
        constexpr int kA_elem = BM * BK;
        constexpr int kA_chunks = kA_elem / 4;  // each chunk = 4 elements
        // We iterate chunks; per chunk load 4 floats from A (16 bytes) and store as 2 half2 to SMEM (8 bytes).
        for (int c = tid; c < kA_chunks; c += threads) {
            int idx0 = c * 4;
            int mm = idx0 / BK;
            int kk = idx0 % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float4 v = make_float4(0, 0, 0, 0);
            if (gm < M && gk + 3 < K) {
                v = *(const float4 *) &A[(size_t) gm * K + gk];
            } else {
                if (gm < M) {
                    if (gk     < K) v.x = A[(size_t) gm * K + gk];
                    if (gk + 1 < K) v.y = A[(size_t) gm * K + gk + 1];
                    if (gk + 2 < K) v.z = A[(size_t) gm * K + gk + 2];
                    if (gk + 3 < K) v.w = A[(size_t) gm * K + gk + 3];
                }
            }
            half2 h01 = __floats2half2_rn(v.x, v.y);
            half2 h23 = __floats2half2_rn(v.z, v.w);
            *(half2 *) &sA_b[mm * BK + kk    ] = h01;
            *(half2 *) &sA_b[mm * BK + kk + 2] = h23;
        }
        // ---- B tile [BN, BK] (col-major in SMEM): scalar load int8, scale, cast.
        //      Vectorize int8 load with int4 (16 bytes = 16 quants).
        constexpr int kB_elem = BN * BK;
        constexpr int kB_chunks = kB_elem / 16;  // 16 int8 per int4 vector
        for (int c = tid; c < kB_chunks; c += threads) {
            int idx0 = c * 16;
            int nn = idx0 / BK;
            int kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            // Load 16 int8 quants
            int8_t qs[16] = {0};
            if (gn < N && gk + 15 < K) {
                *(int4 *)&qs[0] = *(const int4 *) &W_qs[(size_t) gn * K + gk];
            } else if (gn < N) {
                for (int i = 0; i < 16; ++i) {
                    if (gk + i < K) qs[i] = W_qs[(size_t) gn * K + gk + i];
                }
            }
            // For BK=32 and stride 16, the 16 quants span 1 QK_INT8 block (since 32 / 16 = 2 chunks per block).
            // Scale shared across all 16 quants.
            float s = 1.0f;
            if (gn < N) {
                int blk = gk / QK_INT8;
                s = __half2float(W_scales[(size_t) gn * blocks_per_row + blk]);
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                half v = __float2half((float) qs[i] * s);
                sB_b[kk + i + nn * BK] = v;
            }
        }
    };

    // Prologue: load tile 0.
    load_tile(0, 0);
    __syncthreads();

    // Main loop with double-buffer.
    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        // Issue load for next tile into the OTHER buffer.
        // Note: no async cp on V100; this is just reordered gmem→smem loads
        // before the compute, then a __syncthreads() at the end.
        load_tile(kt + 1, 1 - buf);

        // Compute on current tile from buf.
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

    // Epilogue: compute final tile from buf.
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

    // Store FRAG_M*FRAG_N fragments. Reuse sA region as scratch.
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

}  // namespace tc_grid::kernels::int8_opt

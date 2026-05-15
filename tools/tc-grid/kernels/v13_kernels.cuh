// SPRINT-021 P1 — v13_rf: register-file dequant kernel.
//
// Diagnosis from SPRINT-021 P0 (REPORT-15):
//   v12_ms3 INT8 HMMA active = 31.7% at M=2048 N=K=7168 (vs 56% for
//   turbomind FP8 at same shape, same MMA family). Bottleneck is
//   instruction-issue serialization caused by:
//
//     Stage 2 (STS): PRMT-dequant INT8→FP16 in registers, but RESULT
//                    is stored to SMEM as FP16 (2 B/wt SMEM traffic).
//     Stage 3 (mma): LDS dequanted FP16 from SMEM into b_frags.
//
//   The dequant-result goes through SMEM. v12 writes 2 B/wt to SMEM and
//   reads 2 B/wt from SMEM. The round-trip adds dependency latency that
//   ncu shows as smsp__average_warp_latency_per_inst_issued = 7.63 cyc
//   (vs 3.59 for turbomind).
//
// v13_rf fix: store RAW INT8 to SMEM (1 B/wt), LDS 8 INT8 bytes per lane
// in mainloop, PRMT-dequant in registers between LDS and mma. This
// matches what turbomind's Transform_HMMA_SIMT_B does inside its
// SmemCopyAtom_Pack_v3.
//
// Changes vs v12_ms3:
//   * sB layout changes: INT8 instead of FP16, halving SMEM B traffic
//   * sS new: per-buffer scale array (BN halves) so mainloop can load
//     per-N scales for the on-the-fly PRMT
//   * Stage 2 stores raw INT8 + scales (no PRMT)
//   * Stage 3 LDS INT8 + scale + PRMT-dequant → b_frags → mma
//
// Prototype shape: BM=128 BN=128 BK=16 WARPS=4 ATOMS_M=16 ATOMS_N=1 —
// matches v12_ms3 champion to allow head-to-head ncu comparison.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "mma_sm70.cuh"

namespace tc_grid {
namespace kernels {
namespace int8_v13 {

#ifndef QK_INT8
#define QK_INT8 32
#endif

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

using ::tc_grid::mma_sm70::mma_m8n8k4_row_col_acc_f16;

// PRMT-based INT8→FP16 dequant (bias trick), same as v12 — kept here
// so v13 is self-contained and registers are sized identically.
__device__ __forceinline__ void prmt_dequant_4_int8(
        uint32_t qs_u32, half2& out_lo, half2& out_hi, half2 scale_h2) {
    const uint32_t qs_u = qs_u32 ^ 0x80808080;
    uint32_t r_lo, r_hi;
    asm("prmt.b32 %0, %1, 0x64646464, 0x4140;" : "=r"(r_lo) : "r"(qs_u));
    asm("prmt.b32 %0, %1, 0x64646464, 0x4342;" : "=r"(r_hi) : "r"(qs_u));
    const half  c1152   = __float2half(1152.0f);
    const half2 c1152h2 = __halves2half2(c1152, c1152);
    out_lo = __hmul2(__hsub2(*reinterpret_cast<half2*>(&r_lo), c1152h2), scale_h2);
    out_hi = __hmul2(__hsub2(*reinterpret_cast<half2*>(&r_hi), c1152h2), scale_h2);
}

template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int8_lut_v13_rf(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int K_ITERS = BK / ATOM_K;
    // INT8 row stride padding. Each row of sB is BK INT8s; we pad to
    // BK+16 to give 16-byte aligned strides and defeat 4-way conflicts
    // at BN=128 (256-byte alias every row).
    constexpr int BK_INT8_PAD = BK + 16;
    constexpr int BN_PAD = BN + 8;
    static_assert(BM == ATOMS_M * ATOM_M, "BM == ATOMS_M * 8");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "N_PER_WARP == ATOMS_N * 32");
    static_assert(BK == 16, "v13_rf prototype: BK must be 16 (one-scale-per-tile assumption)");
    static_assert(BK <= QK_INT8, "v13_rf assumes BK <= QK_INT8");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;
    static_assert(BK == 16, "");
    static_assert(B_PER_THR * threads >= BN, "need to cover all BN scales");

    // SMEM layout (one buffer each, doubled below):
    //   sA[BM][BK]      half   = BM*BK*2  bytes per buf
    //   sB[BN][BK_INT8_PAD] int8  = BN*BK_INT8_PAD bytes per buf
    //   sS[BN]          half   = BN*2 bytes per buf
    // Plus a final epilogue tile reusing sA storage as sC.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    int8_t * sB = reinterpret_cast<int8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_INT8_PAD);
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> int8_t * { return sB + (size_t) b * BN * BK_INT8_PAD; };
    auto sS_buf = [&](int b) -> __half * { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    // Persistent per-thread rmem buffers across the 3-stage pipeline.
    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];
    int    B_idx0[B_PER_THR];

    // ---- Stage 1: LDG gmem K-tile -> per-thread rmem (same as v12_ms3) ----
    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) {
                    v = *(const float4 *) &A[(size_t) gm * K + gk];
                }
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            ::int4 vq; vq.x = vq.y = vq.z = vq.w = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                idx0 = c_ * 16;
                const int nn = idx0 / BK, kk = idx0 % BK;
                const int gn = tile_n + nn, gk = k0 + kk;
                if (gn < N && gk + 15 < K) {
                    vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                }
                if (gn < N) {
                    s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8));
                }
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    // ---- Stage 2: STS rmem -> SMEM (NO dequant; raw INT8 + scale) ----
    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        int8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b = sS_buf(buf_idx);
        // A: FP32 → FP16 as before
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        // B: write raw INT8 16-bytes-at-a-time (one int4 per thread per chunk).
        // S: write scale at sS_b[nn] for this thread's chunk.
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int idx0 = B_idx0[p];
            if (idx0 < 0) continue;
            const int nn = idx0 / BK, kk = idx0 % BK;
            // Store 16 INT8s as one int4 store.
            *(::int4 *) &sB_b[nn * BK_INT8_PAD + kk] = B_rmem[p];
            // Each thread for this chunk corresponds to one N. Store scale.
            // (BK == 16 → kB_chunks == BN → one chunk per N → one writer per N.)
            sS_b[nn] = S_rmem[p];
        }
    };

    // ---- Stage 3: mma over current SMEM buffer (PRMT dequant in mainloop) ----
    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        int8_t * sB_c = sB_buf(buf);
        __half * sS_c = sS_buf(buf);

        // Scales are constant across K_ITERS within a CTA-K-tile (BK<=QK_INT8).
        // Load each lane's per-N scales ONCE per mainloop entry.
        // Lane L of warp W operates on N's in [an*ATOM_N + n_off_warp + bL_n,
        // an*ATOM_N + n_off_warp + bL_n + 8) for each atom-N. The 8 N's
        // may not be contiguous — bL_n maps lanes to a 4-N stride pattern.
        // Conservative: load the 8 scales lane-wise (8 halves = 16 bytes = 1 LDS).
        // Simpler still: since bL_n's 8 N's land on 8 contiguous positions per
        // f16-acc layout (per V12-DESIGN.md §2: lane holds 1×8 n-strip),
        // we can do a single uint128 load if cL_n_f16 maps consecutively. But
        // that's for OUTPUT. For INPUT, the b_frag lanes use bL_n which spans
        // 4 N's then jumps by 16. Use scalar loads — only 8 per lane per
        // mainloop entry, tiny cost.
        half scales_per_atom[ATOMS_N][8];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                // Match the b_frag lane→N mapping used during LDS below.
                // For 884 B operand: 8 K-direction values per lane; the
                // N-direction varies per lane via bL_n. Within a single
                // lane, all 8 K-direction values are the SAME N. So one
                // scale per lane, NOT 8! Override: scales_per_atom has
                // only one meaningful entry per atom-N.
                (void) j;
                break;
            }
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales_per_atom[an][0] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ++ki) {
            const int k_base = ki * ATOM_K;

            // A frags: identical to v12 (FP16 LDS).
            half a_frags[ATOMS_M][8];
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                const int m = am * ATOM_M + aL_m;
                *reinterpret_cast<uint4*>(&a_frags[am][0]) =
                    *reinterpret_cast<const uint4*>(&sA_c[m * BK + k_base]);
            }

            // B frags: LDS 8 INT8 (one uint64_t) per lane, then PRMT-dequant
            // in registers. The 884 B operand wants 8 halves per lane for
            // the 8 K-direction values at the lane's N.
            half b_frags[ATOMS_N][8];
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                const int n = an * ATOM_N + n_off_warp + bL_n;
                // 8 INT8 bytes at sB_c[n * BK_INT8_PAD + k_base..+7]
                uint64_t qb = *reinterpret_cast<const uint64_t*>(
                    &sB_c[n * BK_INT8_PAD + k_base]);
                uint32_t qs_lo = (uint32_t) (qb & 0xFFFFFFFFu);
                uint32_t qs_hi = (uint32_t) (qb >> 32);
                const half s = scales_per_atom[an][0];
                const half2 s_h2 = __halves2half2(s, s);
                half2 v0, v1, v2, v3;
                prmt_dequant_4_int8(qs_lo, v0, v1, s_h2);
                prmt_dequant_4_int8(qs_hi, v2, v3, s_h2);
                *(half2*) &b_frags[an][0] = v0;
                *(half2*) &b_frags[an][2] = v1;
                *(half2*) &b_frags[an][4] = v2;
                *(half2*) &b_frags[an][6] = v3;
            }

            // mma (same as v12_ms3).
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                #pragma unroll
                for (int an = 0; an < ATOMS_N; ++an) {
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][0], &b_frags[an][0]);
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][4], &b_frags[an][4]);
                }
            }
        }
    };

    // ---- Prelude: tiles 0 and 1, mirroring v12_ms3 ----
    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    // ---- SMEM round-trip epilogue (identical to v12_ms3) ----
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "even split for half2 STG");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2*>(&sC[row * BN_PAD + col]);
        const float2 f  = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ============================================================================
// v13_rf_v2 — K-iter LDS fusion. NEGATIVE RESULT (measured 40.5 TF vs
// v13_rf 44.4 TF, -8.6%). Forcing all LDS before any mma killed the
// implicit LDS/mma overlap that the original ki-staggered loop allowed.
// Kept as documented negative; do not dispatch in champion table.
// ============================================================================
template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int8_lut_v13_rf_v2(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int BK_INT8_PAD = BK + 16;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "v13_rf_v2: BK=16 fused-K path");
    static_assert(BM == ATOMS_M * ATOM_M, "");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    int8_t * sB = reinterpret_cast<int8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_INT8_PAD);
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> int8_t * { return sB + (size_t) b * BN * BK_INT8_PAD; };
    auto sS_buf = [&](int b) -> __half * { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];
    int    B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) {
                    v = *(const float4 *) &A[(size_t) gm * K + gk];
                }
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            ::int4 vq; vq.x = vq.y = vq.z = vq.w = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                idx0 = c_ * 16;
                const int nn = idx0 / BK, kk = idx0 % BK;
                const int gn = tile_n + nn, gk = k0 + kk;
                if (gn < N && gk + 15 < K) {
                    vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                }
                if (gn < N) {
                    s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8));
                }
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        int8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int idx0 = B_idx0[p];
            if (idx0 < 0) continue;
            const int nn = idx0 / BK, kk = idx0 % BK;
            *(::int4 *) &sB_b[nn * BK_INT8_PAD + kk] = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    // ---- Fused-K mainloop: single uint4 B LDS + 4 PRMT + 4 mma per atom ----
    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        int8_t * sB_c = sB_buf(buf);
        __half * sS_c = sS_buf(buf);

        // Per-atom scale (one per warp's atom-N, constant across mainloop).
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }

        // A frags: full BK=16 per lane = 16 halves = uint4 + uint4.
        half a_frags[ATOMS_M][16];
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + aL_m;
            *reinterpret_cast<uint4*>(&a_frags[am][0]) =
                *reinterpret_cast<const uint4*>(&sA_c[m * BK + 0]);
            *reinterpret_cast<uint4*>(&a_frags[am][8]) =
                *reinterpret_cast<const uint4*>(&sA_c[m * BK + 8]);
        }

        // B frags: single uint4 (16 INT8) LDS per lane per atom, expanded
        // via 4× PRMT into 16 halves covering ALL K's.
        half b_frags[ATOMS_N][16];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            uint32_t qs[4];
            *(::int4 *) &qs[0] = *reinterpret_cast<const ::int4*>(
                &sB_c[n * BK_INT8_PAD + 0]);
            const half2 s_h2 = __halves2half2(scales[an], scales[an]);
            half2 t0, t1, t2, t3, t4, t5, t6, t7;
            prmt_dequant_4_int8(qs[0], t0, t1, s_h2);
            prmt_dequant_4_int8(qs[1], t2, t3, s_h2);
            prmt_dequant_4_int8(qs[2], t4, t5, s_h2);
            prmt_dequant_4_int8(qs[3], t6, t7, s_h2);
            *(half2*) &b_frags[an][ 0] = t0;
            *(half2*) &b_frags[an][ 2] = t1;
            *(half2*) &b_frags[an][ 4] = t2;
            *(half2*) &b_frags[an][ 6] = t3;
            *(half2*) &b_frags[an][ 8] = t4;
            *(half2*) &b_frags[an][10] = t5;
            *(half2*) &b_frags[an][12] = t6;
            *(half2*) &b_frags[an][14] = t7;
        }

        // 4 back-to-back m8n8k4 atoms per (am, an): K=0..3, 4..7, 8..11, 12..15.
        // Compiler can interleave the 4 mma's with their A frag loads above.
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 0], &b_frags[an][ 0]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 4], &b_frags[an][ 4]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 8], &b_frags[an][ 8]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][12], &b_frags[an][12]);
            }
        }
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2*>(&sC[row * BN_PAD + col]);
        const float2 f  = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ============================================================================
// v13_rf_v3 — explicit software pipeline within mainloop. Pre-load
// b_frags + a_frags for ki+1 before issuing mma calls for ki. Allows
// the compiler to schedule LDS issue concurrent with mma execution
// (HMMA pipe and LSU pipe are independent). Mirrors turbomind's
// `cute::copy` + mma overlap pattern within a single CTA-K-tile.
// ============================================================================
// v13_rf_v4 stub: declared inline after v13_rf_v3. (forward marker)
template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int8_lut_v13_rf_v3(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int K_ITERS = BK / ATOM_K;
    constexpr int BK_INT8_PAD = BK + 16;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "");
    static_assert(K_ITERS == 2, "v13_rf_v3: K_ITERS=2 hand-pipelined");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    int8_t * sB = reinterpret_cast<int8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_INT8_PAD);
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> int8_t * { return sB + (size_t) b * BN * BK_INT8_PAD; };
    auto sS_buf = [&](int b) -> __half * { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];
    int    B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            ::int4 vq; vq.x = vq.y = vq.z = vq.w = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                idx0 = c_ * 16;
                const int nn = idx0 / BK, kk = idx0 % BK;
                const int gn = tile_n + nn, gk = k0 + kk;
                if (gn < N && gk + 15 < K) vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                if (gn < N) s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8));
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        int8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int idx0 = B_idx0[p];
            if (idx0 < 0) continue;
            const int nn = idx0 / BK, kk = idx0 % BK;
            *(::int4 *) &sB_b[nn * BK_INT8_PAD + kk] = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    // Helper: dequant 8 INT8 (uint64_t) into 8 halves with PRMT.
    auto dequant_8 = [&](uint64_t qb, half2 s_h2, half * out8) {
        uint32_t qs_lo = (uint32_t)(qb & 0xFFFFFFFFu);
        uint32_t qs_hi = (uint32_t)(qb >> 32);
        half2 v0, v1, v2, v3;
        prmt_dequant_4_int8(qs_lo, v0, v1, s_h2);
        prmt_dequant_4_int8(qs_hi, v2, v3, s_h2);
        *(half2*) &out8[0] = v0;
        *(half2*) &out8[2] = v1;
        *(half2*) &out8[4] = v2;
        *(half2*) &out8[6] = v3;
    };

    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        int8_t * sB_c = sB_buf(buf);
        __half * sS_c = sS_buf(buf);

        // Pre-load per-atom scales once per mainloop entry.
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }

        // Two buffers for software-pipelined a/b frags: ki ⇄ ki+1.
        half a_frags[2][ATOMS_M][8];
        half b_frags[2][ATOMS_N][8];

        // Prologue: load frags for ki=0.
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + aL_m;
            *reinterpret_cast<uint4*>(&a_frags[0][am][0]) =
                *reinterpret_cast<const uint4*>(&sA_c[m * BK + 0]);
        }
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            uint64_t qb = *reinterpret_cast<const uint64_t*>(&sB_c[n * BK_INT8_PAD + 0]);
            const half2 s_h2 = __halves2half2(scales[an], scales[an]);
            dequant_8(qb, s_h2, &b_frags[0][an][0]);
        }

        // Steady state: issue LDS+PRMT for ki+1 BEFORE mma of ki. The
        // compiler can schedule LDS/PRMT (LSU+ALU) concurrent with mma
        // (HMMA pipe), hiding the LDS latency under the dependent mma
        // serialization. K_ITERS=2 → 1 steady-state iteration only.
        // Drain iteration handles ki=K_ITERS-1=1 without preload.
        const int ki = 0;
        const int k_next = (ki + 1) * ATOM_K;
        // Issue ki+1 frag loads first.
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + aL_m;
            *reinterpret_cast<uint4*>(&a_frags[1][am][0]) =
                *reinterpret_cast<const uint4*>(&sA_c[m * BK + k_next]);
        }
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            uint64_t qb = *reinterpret_cast<const uint64_t*>(&sB_c[n * BK_INT8_PAD + k_next]);
            const half2 s_h2 = __halves2half2(scales[an], scales[an]);
            dequant_8(qb, s_h2, &b_frags[1][an][0]);
        }
        // mma for ki=0.
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[0][am][0], &b_frags[0][an][0]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[0][am][4], &b_frags[0][an][4]);
            }
        }
        // mma for ki=1 (frags already loaded above).
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[1][am][0], &b_frags[1][an][0]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[1][am][4], &b_frags[1][an][4]);
            }
        }
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2*>(&sC[row * BN_PAD + col]);
        const float2 f  = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ============================================================================
// v13_rf_v4 — v13_rf with launch_bounds(*, 1) instead of (*, 2). Gives
// the compiler up to 256 regs/thread (vs ~128 cap at occupancy=2). That
// matches turbomind's 244 reg/thread footprint. Trades 50% occupancy
// for headroom to unroll/schedule. Bet: compute-bound kernels prefer
// regs to occupancy.
// ============================================================================
template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_>
__launch_bounds__(WARPS_ * 32, 1)
__global__ void mm_int8_lut_v13_rf_v4(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int K_ITERS = BK / ATOM_K;
    constexpr int BK_INT8_PAD = BK + 16;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    int8_t * sB = reinterpret_cast<int8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_INT8_PAD);
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> int8_t * { return sB + (size_t) b * BN * BK_INT8_PAD; };
    auto sS_buf = [&](int b) -> __half * { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];
    int    B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            ::int4 vq; vq.x = vq.y = vq.z = vq.w = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                idx0 = c_ * 16;
                const int nn = idx0 / BK, kk = idx0 % BK;
                const int gn = tile_n + nn, gk = k0 + kk;
                if (gn < N && gk + 15 < K) vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                if (gn < N) s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8));
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        int8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int idx0 = B_idx0[p];
            if (idx0 < 0) continue;
            const int nn = idx0 / BK, kk = idx0 % BK;
            *(::int4 *) &sB_b[nn * BK_INT8_PAD + kk] = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        int8_t * sB_c = sB_buf(buf);
        __half * sS_c = sS_buf(buf);
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_warp + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ++ki) {
            const int k_base = ki * ATOM_K;
            half a_frags[ATOMS_M][8];
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                const int m = am * ATOM_M + aL_m;
                *reinterpret_cast<uint4*>(&a_frags[am][0]) =
                    *reinterpret_cast<const uint4*>(&sA_c[m * BK + k_base]);
            }
            half b_frags[ATOMS_N][8];
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                const int n = an * ATOM_N + n_off_warp + bL_n;
                uint64_t qb = *reinterpret_cast<const uint64_t*>(&sB_c[n * BK_INT8_PAD + k_base]);
                uint32_t qs_lo = (uint32_t)(qb & 0xFFFFFFFFu);
                uint32_t qs_hi = (uint32_t)(qb >> 32);
                const half2 s_h2 = __halves2half2(scales[an], scales[an]);
                half2 v0, v1, v2, v3;
                prmt_dequant_4_int8(qs_lo, v0, v1, s_h2);
                prmt_dequant_4_int8(qs_hi, v2, v3, s_h2);
                *(half2*) &b_frags[an][0] = v0;
                *(half2*) &b_frags[an][2] = v1;
                *(half2*) &b_frags[an][4] = v2;
                *(half2*) &b_frags[an][6] = v3;
            }
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                #pragma unroll
                for (int an = 0; an < ATOMS_N; ++an) {
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][0], &b_frags[an][0]);
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][4], &b_frags[an][4]);
                }
            }
        }
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2*>(&sC[row * BN_PAD + col]);
        const float2 f  = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ============================================================================
// v13_rf_v5 — 2×2 warp partition (like turbomind Blocked<2,2>). Splits
// 4 warps as 2-along-M × 2-along-N so each warp covers 64×64 = 8 atoms_M
// × 2 atoms_N. ATOMS_N=2 means each pair of mma calls SHARES the same A
// frag LDS — halves A-side LDS instruction count.
//
// Compare per K-iter LDS counts on the CTA-tile-K-step:
//   v13_rf_v4 (4×1, ATOMS_M=16, ATOMS_N=1): 16 A LDS + 1 B LDS = 17 LDS
//   v13_rf_v5 (2×2, ATOMS_M=8,  ATOMS_N=2):  8 A LDS + 2 B LDS = 10 LDS
// Same 16 mma calls per warp; 41% fewer LDS instructions.
// ============================================================================
// BK_PAD_ default of 16 keeps the original 32-byte row stride; pass 32
// or 48 at instantiation to expand the stride for bank-conflict tuning.
// Must remain a multiple of 16 to preserve uint4 store alignment in
// the Stage 2 STS.
template <int BM_, int BN_, int BK_, int BK_PAD_=16>
__launch_bounds__(128, 1)
__global__ void mm_int8_lut_v13_rf_v5(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = 4;
    constexpr int WARPS_M = 2, WARPS_N = 2;
    constexpr int BM_PER_WARP = BM / WARPS_M;   // 64
    constexpr int BN_PER_WARP = BN / WARPS_N;   // 64
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int ATOMS_M = BM_PER_WARP / ATOM_M;  // 8
    constexpr int ATOMS_N = BN_PER_WARP / ATOM_N;  // 2
    constexpr int K_ITERS = BK / ATOM_K;
    constexpr int BK_INT8_PAD = BK + BK_PAD_;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "");
    static_assert(BM % (WARPS_M * ATOM_M) == 0 && BN % (WARPS_N * ATOM_N) == 0, "BM/BN tile shape");
    static_assert(WARPS == 4, "v13_rf_v5: 4 warps (2x2 grid)");
    static_assert(BK_INT8_PAD % 4 == 0, "BK_INT8_PAD must be 4-byte aligned for uint LDS");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    // 2x2 warp partition: warp_m = warp/WARPS_N, warp_n = warp%WARPS_N
    const int warp_m   = warp / WARPS_N;
    const int warp_n   = warp % WARPS_N;
    const int m_off_w  = warp_m * BM_PER_WARP;   // 0 or 64
    const int n_off_w  = warp_n * BN_PER_WARP;   // 0 or 64

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    int8_t * sB = reinterpret_cast<int8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_INT8_PAD);
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> int8_t * { return sB + (size_t) b * BN * BK_INT8_PAD; };
    auto sS_buf = [&](int b) -> __half * { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];
    int    B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            ::int4 vq; vq.x = vq.y = vq.z = vq.w = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                idx0 = c_ * 16;
                const int nn = idx0 / BK, kk = idx0 % BK;
                const int gn = tile_n + nn, gk = k0 + kk;
                if (gn < N && gk + 15 < K) vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                if (gn < N) s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8));
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        int8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int idx0 = B_idx0[p];
            if (idx0 < 0) continue;
            const int nn = idx0 / BK, kk = idx0 % BK;
            *(::int4 *) &sB_b[nn * BK_INT8_PAD + kk] = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        int8_t * sB_c = sB_buf(buf);
        __half * sS_c = sS_buf(buf);

        // Scales for this warp's ATOMS_N=2 atoms.
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_w + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ++ki) {
            const int k_base = ki * ATOM_K;

            // A frags: only the M strip for this warp (ATOMS_M=8 atoms,
            // half as many as the 4x1 partition).
            half a_frags[ATOMS_M][8];
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                const int m = am * ATOM_M + m_off_w + aL_m;
                *reinterpret_cast<uint4*>(&a_frags[am][0]) =
                    *reinterpret_cast<const uint4*>(&sA_c[m * BK + k_base]);
            }

            // B frags: ATOMS_N=2 atoms — twice as many LDS but each
            // shares with multiple A frags below.
            half b_frags[ATOMS_N][8];
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                const int n = an * ATOM_N + n_off_w + bL_n;
                uint64_t qb = *reinterpret_cast<const uint64_t*>(&sB_c[n * BK_INT8_PAD + k_base]);
                uint32_t qs_lo = (uint32_t)(qb & 0xFFFFFFFFu);
                uint32_t qs_hi = (uint32_t)(qb >> 32);
                const half2 s_h2 = __halves2half2(scales[an], scales[an]);
                half2 v0, v1, v2, v3;
                prmt_dequant_4_int8(qs_lo, v0, v1, s_h2);
                prmt_dequant_4_int8(qs_hi, v2, v3, s_h2);
                *(half2*) &b_frags[an][0] = v0;
                *(half2*) &b_frags[an][2] = v1;
                *(half2*) &b_frags[an][4] = v2;
                *(half2*) &b_frags[an][6] = v3;
            }

            // mma: outer am loop, inner an loop → each A frag is reused
            // for ATOMS_N=2 mma calls (register reuse, scheduler-friendly).
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                #pragma unroll
                for (int an = 0; an < ATOMS_N; ++an) {
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][0], &b_frags[an][0]);
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][4], &b_frags[an][4]);
                }
            }
        }
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + m_off_w + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_w + cL_n_f16;
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2*>(&sC[row * BN_PAD + col]);
        const float2 f  = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ============================================================================
// v13_rf_v6 — v5 (2x2 warp partition) + K-iter LDS fusion. Each lane does
// ONE uint4 (16 INT8) LDS per atom-N covering BOTH ki=0 and ki=1 K's.
// 4× PRMT then produces 16 halves; 4 back-to-back mma's consume them.
//
// v13_rf_v2 tried this on the 4x1 partition and lost (-7.6%) because the
// 16-atom mma chain was too long for the compiler to schedule with all
// LDS up front. With v5's 2x2 partition (8x2=16 mma per warp), the K
// fusion may now win because A frag reuse hides the dependency.
// ============================================================================
template <int BM_, int BN_, int BK_, int BK_PAD_=16>
__launch_bounds__(128, 1)
__global__ void mm_int8_lut_v13_rf_v6(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = 4;
    constexpr int WARPS_M = 2, WARPS_N = 2;
    constexpr int BM_PER_WARP = BM / WARPS_M;
    constexpr int BN_PER_WARP = BN / WARPS_N;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int ATOMS_M = BM_PER_WARP / ATOM_M;
    constexpr int ATOMS_N = BN_PER_WARP / ATOM_N;
    constexpr int BK_INT8_PAD = BK + BK_PAD_;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "");
    static_assert(BM % (WARPS_M * ATOM_M) == 0 && BN % (WARPS_N * ATOM_N) == 0, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int warp_m   = warp / WARPS_N;
    const int warp_n   = warp % WARPS_N;
    const int m_off_w  = warp_m * BM_PER_WARP;
    const int n_off_w  = warp_n * BN_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    int8_t * sB = reinterpret_cast<int8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_INT8_PAD);
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> int8_t * { return sB + (size_t) b * BN * BK_INT8_PAD; };
    auto sS_buf = [&](int b) -> __half * { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];
    int    B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            ::int4 vq; vq.x = vq.y = vq.z = vq.w = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                idx0 = c_ * 16;
                const int nn = idx0 / BK, kk = idx0 % BK;
                const int gn = tile_n + nn, gk = k0 + kk;
                if (gn < N && gk + 15 < K) vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                if (gn < N) s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8));
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        int8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int idx0 = B_idx0[p];
            if (idx0 < 0) continue;
            const int nn = idx0 / BK, kk = idx0 % BK;
            *(::int4 *) &sB_b[nn * BK_INT8_PAD + kk] = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        int8_t * sB_c = sB_buf(buf);
        __half * sS_c = sS_buf(buf);

        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_w + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }

        // K-fused: a_frags[ATOMS_M][16] covers all 16 K's per lane.
        // 2 LDS per atom (uint4 + uint4) = 16 halves total.
        half a_frags[ATOMS_M][16];
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + m_off_w + aL_m;
            *reinterpret_cast<uint4*>(&a_frags[am][0]) =
                *reinterpret_cast<const uint4*>(&sA_c[m * BK + 0]);
            *reinterpret_cast<uint4*>(&a_frags[am][8]) =
                *reinterpret_cast<const uint4*>(&sA_c[m * BK + 8]);
        }

        // K-fused B: 1 uint4 LDS (16 INT8) per atom-N per lane.
        // 4 PRMT calls → 16 halves into b_frags[an][0..15].
        half b_frags[ATOMS_N][16];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_w + bL_n;
            uint32_t qs[4];
            *(::int4 *) &qs[0] = *reinterpret_cast<const ::int4*>(&sB_c[n * BK_INT8_PAD + 0]);
            const half2 s_h2 = __halves2half2(scales[an], scales[an]);
            half2 t0, t1, t2, t3, t4, t5, t6, t7;
            prmt_dequant_4_int8(qs[0], t0, t1, s_h2);
            prmt_dequant_4_int8(qs[1], t2, t3, s_h2);
            prmt_dequant_4_int8(qs[2], t4, t5, s_h2);
            prmt_dequant_4_int8(qs[3], t6, t7, s_h2);
            *(half2*) &b_frags[an][ 0] = t0;
            *(half2*) &b_frags[an][ 2] = t1;
            *(half2*) &b_frags[an][ 4] = t2;
            *(half2*) &b_frags[an][ 6] = t3;
            *(half2*) &b_frags[an][ 8] = t4;
            *(half2*) &b_frags[an][10] = t5;
            *(half2*) &b_frags[an][12] = t6;
            *(half2*) &b_frags[an][14] = t7;
        }

        // 4 K-direction mma's × ATOMS_M × ATOMS_N. With 2x2 partition and
        // ATOMS_M=8 ATOMS_N=2, that's 8*2*4 = 64 mma calls per warp per
        // CTA-K-tile (vs 32 in v5). Each (am,an) accumulator is hit 4
        // times in a row, sharing the lane's a_frag register state.
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 0], &b_frags[an][ 0]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 4], &b_frags[an][ 4]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 8], &b_frags[an][ 8]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][12], &b_frags[an][12]);
            }
        }
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + m_off_w + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_w + cL_n_f16;
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2*>(&sC[row * BN_PAD + col]);
        const float2 f  = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ============================================================================
// v13_rf_v7 — v5 + reordered mma loop. Instead of issuing two mma's on
// the SAME c_frag[am][an] back-to-back (which creates a HMMA → HMMA
// dependency chain on the same accumulator), interleave the K=0..3 and
// K=4..7 passes across ALL (am,an) pairs. The dependency gap between
// hits on the same accumulator grows from 0 to ATOMS_M*ATOMS_N-1
// independent mma's, letting the HMMA pipe stay busy.
// ============================================================================
template <int BM_, int BN_, int BK_, int BK_PAD_=16>
__launch_bounds__(128, 1)
__global__ void mm_int8_lut_v13_rf_v7(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = 4;
    constexpr int WARPS_M = 2, WARPS_N = 2;
    constexpr int BM_PER_WARP = BM / WARPS_M;
    constexpr int BN_PER_WARP = BN / WARPS_N;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int ATOMS_M = BM_PER_WARP / ATOM_M;
    constexpr int ATOMS_N = BN_PER_WARP / ATOM_N;
    constexpr int K_ITERS = BK / ATOM_K;
    constexpr int BK_INT8_PAD = BK + BK_PAD_;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int warp_m   = warp / WARPS_N;
    const int warp_n   = warp % WARPS_N;
    const int m_off_w  = warp_m * BM_PER_WARP;
    const int n_off_w  = warp_n * BN_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    int8_t * sB = reinterpret_cast<int8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_INT8_PAD);
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> int8_t * { return sB + (size_t) b * BN * BK_INT8_PAD; };
    auto sS_buf = [&](int b) -> __half * { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];
    int    B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            ::int4 vq; vq.x = vq.y = vq.z = vq.w = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                idx0 = c_ * 16;
                const int nn = idx0 / BK, kk = idx0 % BK;
                const int gn = tile_n + nn, gk = k0 + kk;
                if (gn < N && gk + 15 < K) vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                if (gn < N) s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8));
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        int8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int idx0 = B_idx0[p];
            if (idx0 < 0) continue;
            const int nn = idx0 / BK, kk = idx0 % BK;
            *(::int4 *) &sB_b[nn * BK_INT8_PAD + kk] = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        int8_t * sB_c = sB_buf(buf);
        __half * sS_c = sS_buf(buf);
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_w + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ++ki) {
            const int k_base = ki * ATOM_K;
            half a_frags[ATOMS_M][8];
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                const int m = am * ATOM_M + m_off_w + aL_m;
                *reinterpret_cast<uint4*>(&a_frags[am][0]) =
                    *reinterpret_cast<const uint4*>(&sA_c[m * BK + k_base]);
            }
            half b_frags[ATOMS_N][8];
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                const int n = an * ATOM_N + n_off_w + bL_n;
                uint64_t qb = *reinterpret_cast<const uint64_t*>(&sB_c[n * BK_INT8_PAD + k_base]);
                uint32_t qs_lo = (uint32_t)(qb & 0xFFFFFFFFu);
                uint32_t qs_hi = (uint32_t)(qb >> 32);
                const half2 s_h2 = __halves2half2(scales[an], scales[an]);
                half2 v0, v1, v2, v3;
                prmt_dequant_4_int8(qs_lo, v0, v1, s_h2);
                prmt_dequant_4_int8(qs_hi, v2, v3, s_h2);
                *(half2*) &b_frags[an][0] = v0;
                *(half2*) &b_frags[an][2] = v1;
                *(half2*) &b_frags[an][4] = v2;
                *(half2*) &b_frags[an][6] = v3;
            }
            // KEY DIFFERENCE vs v5: two separate sub-passes for K=0..3 then
            // K=4..7. Each pass hits each accumulator exactly once before
            // any accumulator gets a 2nd hit. Maximizes inter-mma
            // independence — 16 mma's deep between same-c-frag hits.
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                #pragma unroll
                for (int an = 0; an < ATOMS_N; ++an) {
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][0], &b_frags[an][0]);
                }
            }
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                #pragma unroll
                for (int an = 0; an < ATOMS_N; ++an) {
                    mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][4], &b_frags[an][4]);
                }
            }
        }
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + m_off_w + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_w + cL_n_f16;
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2*>(&sC[row * BN_PAD + col]);
        const float2 f  = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

// ============================================================================
// v13_rf_v8 — v6 with launch_bounds(*, 2) instead of (*, 1). Untested
// synergy: K-fused + 2x2 partition at HIGHER OCCUPANCY (2 CTAs/SM, 8
// warps/SM vs v6's 4 warps/SM). ncu showed sm__warps_active=12% across
// all v6 measurements; doubling that could help latency hiding even at
// the cost of fewer regs/thread.
// ============================================================================
template <int BM_, int BN_, int BK_, int BK_PAD_=16>
__launch_bounds__(128, 2)
__global__ void mm_int8_lut_v13_rf_v8(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = 4;
    constexpr int WARPS_M = 2, WARPS_N = 2;
    constexpr int BM_PER_WARP = BM / WARPS_M;
    constexpr int BN_PER_WARP = BN / WARPS_N;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int ATOMS_M = BM_PER_WARP / ATOM_M;
    constexpr int ATOMS_N = BN_PER_WARP / ATOM_N;
    constexpr int BK_INT8_PAD = BK + BK_PAD_;
    constexpr int BN_PAD = BN + 8;
    static_assert(BK == 16, "");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int warp_m   = warp / WARPS_N;
    const int warp_n   = warp % WARPS_N;
    const int m_off_w  = warp_m * BM_PER_WARP;
    const int n_off_w  = warp_n * BN_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;
    constexpr int A_PER_THR = (kA_chunks + threads - 1) / threads;
    constexpr int B_PER_THR = (kB_chunks + threads - 1) / threads;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    int8_t * sB = reinterpret_cast<int8_t *>(sA + 2 * BM * BK);
    __half * sS = reinterpret_cast<__half *>(sB + 2 * BN * BK_INT8_PAD);
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> int8_t * { return sB + (size_t) b * BN * BK_INT8_PAD; };
    auto sS_buf = [&](int b) -> __half * { return sS + (size_t) b * BN; };

    const int aL_m = (lane / 16) * 4 + (lane % 4);
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;

    half c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = __float2half(0.0f);

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];
    int    B_idx0[B_PER_THR];

    auto load_gmem_to_rmem = [&](int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            float4 v = make_float4(0, 0, 0, 0);
            int idx0 = -1;
            if (c_ < kA_chunks) {
                idx0 = c_ * 4;
                const int mm = idx0 / BK, kk = idx0 % BK;
                const int gm = tile_m + mm, gk = k0 + kk;
                if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            }
            A_rmem[p] = v;
            A_idx0[p] = idx0;
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int c_ = p * threads + tid;
            ::int4 vq; vq.x = vq.y = vq.z = vq.w = 0;
            half s_h = __float2half(0.0f);
            int idx0 = -1;
            if (c_ < kB_chunks) {
                idx0 = c_ * 16;
                const int nn = idx0 / BK, kk = idx0 % BK;
                const int gn = tile_n + nn, gk = k0 + kk;
                if (gn < N && gk + 15 < K) vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                if (gn < N) s_h = __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8));
            }
            B_rmem[p] = vq;
            S_rmem[p] = s_h;
            B_idx0[p] = idx0;
        }
    };

    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        int8_t * sB_b = sB_buf(buf_idx);
        __half * sS_b = sS_buf(buf_idx);
        #pragma unroll
        for (int p = 0; p < A_PER_THR; ++p) {
            const int idx0 = A_idx0[p];
            if (idx0 < 0) continue;
            const int mm = idx0 / BK, kk = idx0 % BK;
            const float4 v = A_rmem[p];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        #pragma unroll
        for (int p = 0; p < B_PER_THR; ++p) {
            const int idx0 = B_idx0[p];
            if (idx0 < 0) continue;
            const int nn = idx0 / BK, kk = idx0 % BK;
            *(::int4 *) &sB_b[nn * BK_INT8_PAD + kk] = B_rmem[p];
            sS_b[nn] = S_rmem[p];
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        int8_t * sB_c = sB_buf(buf);
        __half * sS_c = sS_buf(buf);
        half scales[ATOMS_N];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_w + bL_n;
            scales[an] = (n < BN) ? sS_c[n] : __float2half(0.0f);
        }
        // K-fused (same as v6).
        half a_frags[ATOMS_M][16];
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            const int m = am * ATOM_M + m_off_w + aL_m;
            *reinterpret_cast<uint4*>(&a_frags[am][0]) =
                *reinterpret_cast<const uint4*>(&sA_c[m * BK + 0]);
            *reinterpret_cast<uint4*>(&a_frags[am][8]) =
                *reinterpret_cast<const uint4*>(&sA_c[m * BK + 8]);
        }
        half b_frags[ATOMS_N][16];
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n = an * ATOM_N + n_off_w + bL_n;
            uint32_t qs[4];
            *(::int4 *) &qs[0] = *reinterpret_cast<const ::int4*>(&sB_c[n * BK_INT8_PAD + 0]);
            const half2 s_h2 = __halves2half2(scales[an], scales[an]);
            half2 t0, t1, t2, t3, t4, t5, t6, t7;
            prmt_dequant_4_int8(qs[0], t0, t1, s_h2);
            prmt_dequant_4_int8(qs[1], t2, t3, s_h2);
            prmt_dequant_4_int8(qs[2], t4, t5, s_h2);
            prmt_dequant_4_int8(qs[3], t6, t7, s_h2);
            *(half2*) &b_frags[an][ 0] = t0;
            *(half2*) &b_frags[an][ 2] = t1;
            *(half2*) &b_frags[an][ 4] = t2;
            *(half2*) &b_frags[an][ 6] = t3;
            *(half2*) &b_frags[an][ 8] = t4;
            *(half2*) &b_frags[an][10] = t5;
            *(half2*) &b_frags[an][12] = t6;
            *(half2*) &b_frags[an][14] = t7;
        }
        #pragma unroll
        for (int am = 0; am < ATOMS_M; ++am) {
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 0], &b_frags[an][ 0]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 4], &b_frags[an][ 4]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][ 8], &b_frags[an][ 8]);
                mma_m8n8k4_row_col_acc_f16(c_frag[am][an], &a_frags[am][12], &b_frags[an][12]);
            }
        }
    };

    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        store_rmem_to_smem(next_buf);
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    mainloop(buf);

    __syncthreads();
    __half * sC = sA;
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + m_off_w + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_w + cL_n_f16;
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;
        const int row    = h2_idx / (BN / 2);
        const int col_h2 = h2_idx % (BN / 2);
        const int col    = col_h2 * 2;
        const half2 v = *reinterpret_cast<const half2*>(&sC[row * BN_PAD + col]);
        const float2 f  = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (gm < M && gn + 1 < N) {
            C[(size_t) gm * N + gn + 0] = f.x;
            C[(size_t) gm * N + gn + 1] = f.y;
        } else if (gm < M && gn < N) {
            C[(size_t) gm * N + gn] = f.x;
        }
    }
}

}  // namespace int8_v13
}  // namespace kernels
}  // namespace tc_grid

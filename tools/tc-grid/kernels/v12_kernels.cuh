// v12 = v11 base + FP16 accumulator + SMEM round-trip epilogue.
//
// What changed vs v11:
//   - c_frag: float[ATOMS_M][ATOMS_N][8] → half[ATOMS_M][ATOMS_N][8].
//     The PTX m8n8k4 f16-acc variant has D = 4 × .f16x2 (8 halves) per
//     thread, SAME element count as f32-acc but half the register footprint
//     (16 bytes vs 32 bytes per atom-frag).
//   - mma atom: `mma_m8n8k4_row_col_acc` → `mma_m8n8k4_row_col_acc_f16`.
//   - Epilogue: SMEM round-trip (write c_frag halves to sC, __syncthreads,
//     cooperative half2→fp32 STG). The f16-acc lane→(m,n) mapping is
//     CONTIGUOUS 1×8 n-strip per lane (see V12-DESIGN.md §2), so each lane's
//     scatter is one `uint4` (16 bytes = 8 halves) store. Much simpler than
//     the f32-acc 4-pair scatter.
//   - SMEM allocation: sC reuses the sA+sB region after the last mma
//     `__syncthreads()`. Total dynamic SMEM = max(sA+sB, sC). For the champ
//     shape (BM=128, BN=128, BK=16): max(20 KB, 17.4 KB) = 20 KB; at
//     `launch_bounds(2)` → 40 KB per SM, well within 96 KB.
//
// Rationale: §6.1 of REPORT-12. v11's mainloop is gmem-latency-bound
// (21% long_scoreboard, 31% on the rebuild ncu pass); ncu shows the tensor
// pipe is idle ~71% of active time. Doubling tensor-pipe peak via f16-acc
// (62 → 125 TF) is the biggest single lever. The c_frag register cut also
// unlocks BM=192 (P4) and 3-stage register-budget headroom (P2).
//
// Acceptance: bit-correct vs v11 reference at every M ∈ {1, 8, 32, 64,
// 256, 1024, 2048, 4096} with sprint tolerance `rel ≤ 1e-3 ∧ p99 ≤ 0.05
// ∧ maxabs ≤ 0.1`. The fp16 accumulator's quantization floor will surface
// as small absolute deltas (~1e-3) on uniform_small inputs; sprint tol
// holds because `maxabs ≤ 0.1` backstops the rel-blowup at small magnitudes.

#pragma once

#include "tc_grid.h"
#include "mma_sm70.cuh"
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8_v12 {

__device__ __forceinline__ void prefetch_l2(const void * ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

using ::tc_grid::mma_sm70::mma_m8n8k4_row_col_acc_f16;

// PRMT-based INT8→FP16 dequant (bias trick), same as v11.
// Output: two half2 values covering 4 dequanted values.
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
__global__ void mm_int8_lut_v12(
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
    constexpr int BK_PAD = BK + 8;
    constexpr int BN_PAD = BN + 8;  // sC row stride, pads to defeat the 4-way
                                    // bank conflict at BN=128 (256-byte rows
                                    // alias bank 0 every row).
    static_assert(BM == ATOMS_M * ATOM_M, "BM == ATOMS_M * 8");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "N_PER_WARP == ATOMS_N * 32");
    static_assert(BK % ATOM_K == 0, "BK must be multiple of ATOM_K=8");
    static_assert(BK % QK_INT8 == 0 || QK_INT8 % BK == 0, "BK and QK_INT8 commensurate");
    static_assert(BK_PAD % 2 == 0, "alignment");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;

    // SMEM layout: sA[0 .. 2*BM*BK), sB[2*BM*BK .. 2*BM*BK + 2*BN*BK_PAD)
    // during mainloop. After the last __syncthreads(), the same buffer is
    // reinterpreted as sC[BM][BN_PAD]. We size the dynamic SMEM
    // (via launch_int8 smem_bytes) to max(mainloop, epilogue).
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BN * BK_PAD; };

    // 884-atom per-lane offsets for the INPUT operands (unchanged from v11).
    const int aL_m = (lane / 16) * 4 + (lane % 4);                       // A m
    const int bL_n = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);     // B n

    // f16-acc OUTPUT mapping (empirically derived in P1.1, V12-DESIGN.md §2):
    //   per (am, an) atom, lane L holds 8 contiguous n's at fixed m.
    const int cL_m_f16 = ((lane >> 4) << 2) | (lane & 3);     // ∈ {0..7}
    const int cL_n_f16 = ((lane >> 2) & 3) << 3;              // ∈ {0,8,16,24}

    // FragC = [ATOMS_M][ATOMS_N][8 halves per lane]
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

    // 2-stage gmem→smem fused loader (same as v11; 3-stage deferred to P2).
    auto load_tile = [&](int kt, int buf_idx, int prefetch_kt) {
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
        const int k0 = kt * BK;
        if (prefetch_kt < k_tiles && tid == 0) {
            const int pk0 = prefetch_kt * BK;
            prefetch_l2(&A[(size_t)(tile_m) * K + pk0]);
            prefetch_l2(&W_qs[(size_t)(tile_n) * K + pk0]);
            prefetch_l2(&W_scales[(size_t)(tile_n) * blocks_per_row + (pk0 / QK_INT8)]);
        }
        for (int c_ = tid; c_ < kA_chunks; c_ += threads) {
            int idx0 = c_ * 4;
            int mm = idx0 / BK, kk = idx0 % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float4 v = make_float4(0, 0, 0, 0);
            if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        for (int c_ = tid; c_ < kB_chunks; c_ += threads) {
            int idx0 = c_ * 16;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            uint32_t qs_u32[4] = {0, 0, 0, 0};
            if (gn < N && gk + 15 < K) {
                ::int4 vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                *(::int4 *)&qs_u32[0] = vq;
            }
            half s_h = (gn < N) ? __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8))
                                : __float2half(0.0f);
            const half2 s_h2 = __halves2half2(s_h, s_h);
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                half2 v_lo, v_hi;
                prmt_dequant_4_int8(qs_u32[g], v_lo, v_hi, s_h2);
                *(half2*)&sB_b[nn * BK_PAD + (kk + g * 4 + 0)] = v_lo;
                *(half2*)&sB_b[nn * BK_PAD + (kk + g * 4 + 2)] = v_hi;
            }
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        __half * sB_c = sB_buf(buf);
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
                *reinterpret_cast<uint4*>(&b_frags[an][0]) =
                    *reinterpret_cast<const uint4*>(&sB_c[n * BK_PAD + k_base]);
            }
            // Two back-to-back m8n8k4 calls (K=0..3 then K=4..7); f16 acc.
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

    load_tile(0, 0, 1);
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        load_tile(kt + 1, 1 - buf, kt + 2);
        mainloop(buf);
        __syncthreads();
        buf = 1 - buf;
    }
    mainloop(buf);

    __syncthreads();
    // ---- SMEM round-trip epilogue ----
    // sB and sA are dead. Reinterpret the same SMEM region as sC[BM][BN_PAD].
    __half * sC = sA;   // base of dynamic SMEM
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        const int m_in_cta = am * ATOM_M + cL_m_f16;
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int n_in_cta = an * ATOM_N + n_off_warp + cL_n_f16;
            // One uint4 store (16 bytes = 8 halves) per (am, an) per lane.
            *reinterpret_cast<uint4*>(&sC[m_in_cta * BN_PAD + n_in_cta]) =
                *reinterpret_cast<uint4*>(&c_frag[am][an][0]);
        }
    }
    __syncthreads();

    // Cooperative half2 → fp32 STG. Threads stride over sC in half2 chunks.
    // Each thread does (BM*BN / threads / 2) half2 conversions.
    constexpr int total_half2 = (BM * BN) / 2;
    constexpr int per_thr     = total_half2 / threads;
    static_assert((BM * BN) % (2 * threads) == 0, "even split for half2 STG");
    #pragma unroll
    for (int it = 0; it < per_thr; ++it) {
        const int h2_idx = it * threads + tid;       // half2 index in BM*BN
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
// v12_ms3 — 3-stage decoupled pipeline (SPRINT-019 P2 / §6.2).
//
// Mainloop schedule per K-tile (sprint REPORT-12 §6.2):
//   stage 1: LDG  gmem → rmem  (next K-tile's data into per-thread rmem)
//   stage 2: STS  rmem → SMEM  (previous tile's rmem flushed to SMEM, with
//                               PRMT INT8→FP16 dequant for B)
//   stage 3: mma  SMEM → tensor cores (current tile's compute)
//
// The compiler is expected to interleave the three across one K-iter so LDG
// (gmem latency) overlaps with mma (math + smem latency). sprint-017's 3-stage
// attempt on v11 cratered the BM=128 W=4 champion (-29%) because v11's 194-reg
// footprint left no headroom for the persistent rmem (~20 regs). v12's 133-reg
// footprint provides that headroom; sprint-019 §3.3 puts §6.2 after §6.1
// specifically for this reason.
//
// Per-thread rmem cost (champion BM=128 BN=128 BK=16, threads=128):
//   A_PER_THR = (BM*BK/4) / threads = 512/128 = 4  float4 = 16 floats = 16 regs
//   B_PER_THR = (BN*BK/16) / threads = 128/128 = 1  int4 = 4 i32 = 4 regs
//   S_PER_THR = B_PER_THR                        = 1  half ≈ 1 reg
//   total ≈ 21 regs persistent across K-iters
//
// 21 regs added to v12's 133 → ~154 regs/thread. 154 * 128 * 2 CTAs = 39424
// regs/SM < 65536 V100 limit. launch_bounds(2) preserved.

template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int8_lut_v12_ms3(
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
    constexpr int BK_PAD = BK + 8;
    constexpr int BN_PAD = BN + 8;
    static_assert(BM == ATOMS_M * ATOM_M, "BM == ATOMS_M * 8");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "N_PER_WARP == ATOMS_N * 32");
    static_assert(BK % ATOM_K == 0, "BK multiple of ATOM_K=8");

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
    __half * sB = sA + 2 * BM * BK;
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BN * BK_PAD; };

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

    // Persistent per-thread rmem buffers (one K-tile's worth).
    float4 A_rmem[A_PER_THR];
    ::int4 B_rmem[B_PER_THR];
    half   S_rmem[B_PER_THR];
    int    A_idx0[A_PER_THR];   // (mm, kk) compressed: idx0 = c_*4
    int    B_idx0[B_PER_THR];   // (nn, kk) compressed: idx0 = c_*16

    // ---- Stage 1: LDG gmem K-tile -> per-thread rmem ----
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

    // ---- Stage 2: STS rmem -> SMEM (with PRMT dequant for B) ----
    auto store_rmem_to_smem = [&](int buf_idx) {
        __half * sA_b = sA_buf(buf_idx);
        __half * sB_b = sB_buf(buf_idx);
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
            uint32_t qs_u32[4];
            *(::int4 *)&qs_u32[0] = B_rmem[p];
            const half2 s_h2 = __halves2half2(S_rmem[p], S_rmem[p]);
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                half2 v_lo, v_hi;
                prmt_dequant_4_int8(qs_u32[g], v_lo, v_hi, s_h2);
                *(half2*)&sB_b[nn * BK_PAD + (kk + g * 4 + 0)] = v_lo;
                *(half2*)&sB_b[nn * BK_PAD + (kk + g * 4 + 2)] = v_hi;
            }
        }
    };

    // ---- Stage 3: mma over the current SMEM buffer ----
    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        __half * sB_c = sB_buf(buf);
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
                *reinterpret_cast<uint4*>(&b_frags[an][0]) =
                    *reinterpret_cast<const uint4*>(&sB_c[n * BK_PAD + k_base]);
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

    // ---- Prelude: load tiles 0 and 1 into SMEM. K-tile 1 stays in rmem
    // until the first steady-state iter's STS. ----
    load_gmem_to_rmem(0);
    store_rmem_to_smem(0);
    if (k_tiles > 1) load_gmem_to_rmem(1);
    __syncthreads();

    // ---- Steady state: mma(kt) || STS(kt+1) || LDG(kt+2) ----
    // At iter kt: rmem holds tile kt+1, SMEM[kt%2] holds tile kt for mma.
    int buf = 0;
    for (int kt = 0; kt < k_tiles - 1; ++kt) {
        const int next_buf = 1 - buf;
        mainloop(buf);
        // After mma reads SMEM[buf], next iter will read SMEM[next_buf].
        // STS tile (kt+1) from rmem into SMEM[next_buf] now.
        store_rmem_to_smem(next_buf);
        // LDG tile (kt+2) into rmem (overwriting rmem that just got STS'd).
        if (kt + 2 < k_tiles) load_gmem_to_rmem(kt + 2);
        __syncthreads();
        buf = next_buf;
    }
    // ---- Drain: final mma on SMEM[buf]. ----
    mainloop(buf);

    __syncthreads();
    // ---- SMEM round-trip epilogue (identical to v12) ----
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
// v12s — v12 mainloop + SplitK accumulation across blockIdx.z (SPRINT-019 P3).
//
// Target: close the M=64 production gap. v11/v12 at M=64 hit ~7.5 TF
// because the (M/BM) × (N/BN) grid is small (1 row × 56 cols at BM=128 BN=128
// N=7168) which leaves SMs idle. SplitK multiplies the grid.z dimension by
// KS, so KS=8 at M=64 BN=64 BM=64 yields 1 × 112 × 8 = 896 CTAs — enough
// to fill all 80 V100 SMs with parallel work.
//
// Output contract (mirrors v10s):
//   - When KS == 1: regular store; bit-identical to v12.
//   - When KS  > 1: atomicAdd to gmem C. Caller MUST pre-zero C.
//
// fp32→fp16 epilogue kernel deferred — v11/v12/v12s all output fp32 to gmem
// (matching v10's contract), so no precision-cast epilogue is needed in the
// SplitK case. The sprint §3.2 "separate fp32→fp16 epilogue" is relevant
// only if/when we switch the C buffer to fp16; that's a separate change.

template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_, int KS_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int8_lut_v12s(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        const float  * __restrict__ A,
        float        * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int ATOMS_M = ATOMS_M_;
    constexpr int ATOMS_N = ATOMS_N_;
    constexpr int KS = KS_;
    constexpr int N_PER_WARP = BN / WARPS;
    constexpr int ATOM_M = 8, ATOM_N = 32, ATOM_K = 8;
    constexpr int K_ITERS = BK / ATOM_K;
    constexpr int BK_PAD = BK + 8;
    constexpr int BN_PAD = BN + 8;
    static_assert(BM == ATOMS_M * ATOM_M, "BM == ATOMS_M * 8");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "N_PER_WARP == ATOMS_N * 32");
    static_assert(BK % ATOM_K == 0, "BK multiple of ATOM_K=8");
    static_assert(KS >= 1, "KS >= 1");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int k_split = blockIdx.z;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks = (BM * BK) / 4;
    constexpr int kB_chunks = (BN * BK) / 16;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BN * BK_PAD; };

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

    const int k_tiles_total = K / BK;
    // Each k_split handles a contiguous chunk of k_tiles; remainder to early splits.
    const int kt_base = (k_tiles_total / KS) * k_split + min(k_split, k_tiles_total % KS);
    const int kt_lim  = kt_base + (k_tiles_total / KS) + ((k_split < (k_tiles_total % KS)) ? 1 : 0);
    const int blocks_per_row = K / QK_INT8;

    // Guard: empty range (KS > k_tiles_total)
    if (kt_base >= kt_lim) return;

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
        for (int c_ = tid; c_ < kA_chunks; c_ += threads) {
            int idx0 = c_ * 4;
            int mm = idx0 / BK, kk = idx0 % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float4 v = make_float4(0, 0, 0, 0);
            if (gm < M && gk + 3 < K) v = *(const float4 *) &A[(size_t) gm * K + gk];
            *(half2 *) &sA_b[mm * BK + kk    ] = __floats2half2_rn(v.x, v.y);
            *(half2 *) &sA_b[mm * BK + kk + 2] = __floats2half2_rn(v.z, v.w);
        }
        for (int c_ = tid; c_ < kB_chunks; c_ += threads) {
            int idx0 = c_ * 16;
            int nn = idx0 / BK, kk = idx0 % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            uint32_t qs_u32[4] = {0, 0, 0, 0};
            if (gn < N && gk + 15 < K) {
                ::int4 vq = __ldg((const ::int4 *) &W_qs[(size_t) gn * K + gk]);
                *(::int4 *)&qs_u32[0] = vq;
            }
            half s_h = (gn < N) ? __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8))
                                : __float2half(0.0f);
            const half2 s_h2 = __halves2half2(s_h, s_h);
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                half2 v_lo, v_hi;
                prmt_dequant_4_int8(qs_u32[g], v_lo, v_hi, s_h2);
                *(half2*)&sB_b[nn * BK_PAD + (kk + g * 4 + 0)] = v_lo;
                *(half2*)&sB_b[nn * BK_PAD + (kk + g * 4 + 2)] = v_hi;
            }
        }
    };

    auto mainloop = [&](int buf) {
        __half * sA_c = sA_buf(buf);
        __half * sB_c = sB_buf(buf);
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
                *reinterpret_cast<uint4*>(&b_frags[an][0]) =
                    *reinterpret_cast<const uint4*>(&sB_c[n * BK_PAD + k_base]);
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

    load_tile(kt_base, 0, kt_base + 1);
    __syncthreads();
    int buf = 0;
    for (int kt = kt_base; kt < kt_lim - 1; ++kt) {
        load_tile(kt + 1, 1 - buf, kt + 2);
        mainloop(buf);
        __syncthreads();
        buf = 1 - buf;
    }
    mainloop(buf);

    __syncthreads();
    // SMEM round-trip epilogue (same as v12). KS == 1 stores; KS > 1 atomicAdds.
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
        const float2 f = __half22float2(v);
        const int gm = tile_m + row;
        const int gn = tile_n + col;
        if (KS == 1) {
            if (gm < M && gn + 1 < N) {
                C[(size_t) gm * N + gn + 0] = f.x;
                C[(size_t) gm * N + gn + 1] = f.y;
            } else if (gm < M && gn < N) {
                C[(size_t) gm * N + gn] = f.x;
            }
        } else {
            if (gm < M && gn + 1 < N) {
                atomicAdd(&C[(size_t) gm * N + gn + 0], f.x);
                atomicAdd(&C[(size_t) gm * N + gn + 1], f.y);
            } else if (gm < M && gn < N) {
                atomicAdd(&C[(size_t) gm * N + gn], f.x);
            }
        }
    }
}

}  // namespace tc_grid::kernels::int8_v12

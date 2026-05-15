// v11 = v10 base + manual Lds + SM70_MMA_884 atoms (m8n32k8 per fma).
//
// Replaces v10's wmma path:
//   wmma::load_matrix_sync (A row-major, B row-major)
//   wmma::mma_sync         (m16n16k16 over half×half→float)
//   wmma::store_matrix_sync via SMEM scratchpad → gmem
//
// with the manual Lds path validated in tests/test_smem_to_frag_sm70.cu and
// tests/test_mma_884_tile_sm70.cu:
//   per-lane uint4 Lds from SMEM at offsets given by SmemCopy_MMA_884
//   SM70_MMA_884::fma = 2× mma.m8n8k4.row.col PTX, K=0..3 then K=4..7
//   per-lane scatter to gmem via thread_offset_C + static_offset_C
//
// Per-CTA tile: BM × BN, sliced as ATOMS_M m-atoms × ATOMS_N n-atoms-per-warp.
//   ATOMS_M = BM / 8         (884 atom is m=8)
//   ATOMS_N = N_PER_WARP / 32 (884 atom is n=32)
//   K_ITERS = BK / 8          (884 atom is k=8)
//
// SMEM layout:
//   sA: row-major [BM][BK]                    — unchanged from v10
//   sB: col-major [BN][BK_PAD] (n outer, k inner) — **CHANGED** from v10's row-major,
//       so that lane's 8-K-contiguous reads can use uint4. BK_PAD = BK + 2 halves
//       gives stride/2 odd (coprime with 32) → zero SMEM bank conflicts.
//
// Acceptance: bit-correct vs v10 reference at every M ∈ {64, 256, 1024, 2048, 4096}.
// Perf: no improvement expected for this step (same op count, same SMEM size).
//
// Co-Authored-By: Claude Opus 4.7 (1M context)

#pragma once

#include "tc_grid.h"
#include "mma_sm70.cuh"
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8_v11 {

__device__ __forceinline__ void prefetch_l2(const void * ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

using ::tc_grid::mma_sm70::mma_m8n8k4_row_col;
using ::tc_grid::mma_sm70::mma_m8n8k4_row_col_acc;

// Swizzle<3,3,3> from turbomind (core/layout.h). XORs offset bits 6,7,8 into
// bits 3,4,5. Preserves bits 0..2 (the 8-half/16-byte uint4-internal block),
// so uint4 loads/stores remain valid for any 8-half-aligned base offset.
// Self-inverse (XOR), so the same fn is used at store and load.
template <int Bits = 3, int Base = 3, int Shift = 3>
__device__ __forceinline__ int swizzle_offset(int offset) {
    constexpr int bit_mask = (1 << Bits) - 1;
    constexpr int yyy_mask = bit_mask << (Base + Shift);
    return offset ^ ((offset & yyy_mask) >> Shift);
}

// PRMT-based dequant: 4 signed INT8 → 4 FP16 (= 2 half2).
//
// Standard "bias trick" used in AWQ/GPTQ kernels for sm_70+:
//   1. XOR with 0x80 flips the sign bit, mapping signed [-128, 127] → unsigned [0, 255].
//   2. PRMT splices each unsigned int8 into FP16's mantissa low byte, with the high
//      byte set to 0x64 (= biased exponent for value 1024.0). The resulting FP16 is
//      1024 + u8 (since exp = 25, mantissa bits 0-7 carry the int8 value).
//   3. Subtract 1152.0 (= 1024 + 128) in FP16 to undo the bias-then-unsigned shift,
//      yielding the original signed int8 as a half.
//
// Replaces 4× __short2half_rn (each compiling to multiple SASS cvt/I2F ops) with
// 2 PRMT + 2 HFMA — savings of ~10–15 cycles per 4 int8s on Volta.
//
// `qs_u32` packs 4 int8 bytes (e.g., qs[0..3] in little-endian).
// Output: two half2 values covering the 4 dequanted values.
__device__ __forceinline__ void prmt_dequant_4_int8(
        uint32_t qs_u32, half2& out_lo, half2& out_hi, half2 scale_h2) {
    const uint32_t qs_u = qs_u32 ^ 0x80808080;
    uint32_t r_lo, r_hi;
    // ctrl 0x4140: out byte0 = src byte0, byte1 = src1 byte4 (= 0x64),
    //              out byte2 = src byte1, byte3 = src1 byte4 (= 0x64)
    asm("prmt.b32 %0, %1, 0x64646464, 0x4140;" : "=r"(r_lo) : "r"(qs_u));
    // ctrl 0x4342: out byte0 = src byte2, byte1 = 0x64, byte2 = src byte3, byte3 = 0x64
    asm("prmt.b32 %0, %1, 0x64646464, 0x4342;" : "=r"(r_hi) : "r"(qs_u));
    // Subtract bias (1152.0f) and scale in one __hfma2.
    // (FP16_1152 + i8) * scale - 1152*scale  ==>  i8 * scale.  Done as
    // __hfma2(*half2*, scale_h2, -1152 * scale_h2) using one FFMA each.
    const half  c1152   = __float2half(1152.0f);
    const half2 c1152h2 = __halves2half2(c1152, c1152);
    out_lo = __hmul2(__hsub2(*reinterpret_cast<half2*>(&r_lo), c1152h2), scale_h2);
    out_hi = __hmul2(__hsub2(*reinterpret_cast<half2*>(&r_hi), c1152h2), scale_h2);
}

template <int BM_, int BN_, int BK_, int WARPS_, int ATOMS_M_, int ATOMS_N_>
__launch_bounds__(WARPS_ * 32, 2)
__global__ void mm_int8_lut_v11(
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
    // BK_PAD = BK + 8 (multiple of 8 → uint4-aligned). Bank stride 4-way conflict
    // (stride/2 = BK/2 + 4 is even, GCD with 32 ≥ 2). Measured on V100 to beat
    // both (a) BK_PAD=BK with Swizzle<3,3,3> by ~1% at the BK=16 champion shape
    // and (b) BK_PAD=BK no-swizzle (8/16-way conflict) by 5-15%.
    constexpr int BK_PAD = BK + 8;
    static_assert(BM == ATOMS_M * ATOM_M, "BM == ATOMS_M * 8");
    static_assert(N_PER_WARP == ATOMS_N * ATOM_N, "N_PER_WARP == ATOMS_N * 32");
    static_assert(BK % ATOM_K == 0, "BK must be multiple of ATOM_K=8");
    static_assert(BK % QK_INT8 == 0 || QK_INT8 % BK == 0, "BK and QK_INT8 must be commensurate");
    static_assert(BK_PAD % 2 == 0, "alignment");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    constexpr int threads = WARPS * 32;
    const int n_off_warp = warp * N_PER_WARP;

    constexpr int kA_chunks   = (BM * BK) / 4;
    constexpr int kB_chunks   = (BN * BK) / 16;
    constexpr int A_PER_THR   = kA_chunks / threads + (kA_chunks % threads != 0);
    constexpr int B_PER_THR   = kB_chunks / threads + (kB_chunks % threads != 0);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + 2 * BM * BK;
    auto sA_buf = [&](int b) -> __half * { return sA + (size_t) b * BM * BK; };
    auto sB_buf = [&](int b) -> __half * { return sB + (size_t) b * BN * BK_PAD; };

    // 884-atom per-lane offsets (Step 2a-validated)
    const int aL_m  = (lane / 16) * 4 + (lane % 4);                            // m for A
    const int bL_n  = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4);          // n for B (relative to atom base)
    const int cL_m  = (lane & 1) + (lane / 16) * 4;                            // m base for C
    const int cL_n  = (lane & 2) + (lane & 12) * 2;                            // n base for C

    // FragC = [ATOMS_M][ATOMS_N][8 floats per lane]
    float c_frag[ATOMS_M][ATOMS_N][8];
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am)
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                c_frag[am][an][i] = 0.0f;

    const int k_tiles = K / BK;
    const int blocks_per_row = K / QK_INT8;

    // 2-stage gmem→smem fused loader. We tried a 3-stage decoupled
    // pipeline (LDG into per-thread rmem, then STS-from-rmem in a separate
    // phase). It hit a peak of 36.68 TF at M=2048 with 64x256x16_w8 (+4.6%)
    // but added register pressure that cratered other shapes — particularly
    // at M=4096 (128x128x16_w4 went 34.77 → 24.92 TF). The 3-stage pipeline
    // needs per-shape register-budget tuning to be a net win; left as a
    // future-work item.
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
            half s_h = (gn < N) ? __ldg(W_scales + (size_t) gn * blocks_per_row + (gk / QK_INT8)) : __float2half(0.0f);
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
            // Load FragA[ATOMS_M][8]: lane reads at sA[am*8 + aL_m, k_base..k_base+7]
            half a_frags[ATOMS_M][8];
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                const int m = am * ATOM_M + aL_m;
                *reinterpret_cast<uint4*>(&a_frags[am][0]) =
                    *reinterpret_cast<const uint4*>(&sA_c[m * BK + k_base]);
            }
            // Load FragB[ATOMS_N][8]: lane reads at sB[an*32 + n_off_warp + bL_n, k_base..k_base+7]
            half b_frags[ATOMS_N][8];
            #pragma unroll
            for (int an = 0; an < ATOMS_N; ++an) {
                const int n = an * ATOM_N + n_off_warp + bL_n;
                *reinterpret_cast<uint4*>(&b_frags[an][0]) =
                    *reinterpret_cast<const uint4*>(&sB_c[n * BK_PAD + k_base]);
            }
            // mma: two back-to-back m8n8k4 per atom (K=0..3 then K=4..7).
            // Atom-major (am, an inner) — compiler already interleaves mma1/mma2
            // across atoms when this is #pragma unroll'd. Tried explicit k-major
            // ordering (all mma1 then all mma2) and it regressed -1.9%, presumably
            // because it widened a_frags register lifetime. Keep atom-major.
            #pragma unroll
            for (int am = 0; am < ATOMS_M; ++am) {
                #pragma unroll
                for (int an = 0; an < ATOMS_N; ++an) {
                    mma_m8n8k4_row_col_acc(c_frag[am][an], &a_frags[am][0], &b_frags[an][0]);
                    mma_m8n8k4_row_col_acc(c_frag[am][an], &a_frags[am][4], &b_frags[an][4]);
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
    // Epilogue: scatter c_frag to gmem C[M, N].
    // Per-lane FragC[8] is 4 pairs at static_offset_C ∈ {(0,0),(2,0),(0,4),(2,4)}
    // relative to (cL_m, cL_n). Each pair is 2 floats at (m+dm, n+dn) and (m+dm, n+dn+1).
    #pragma unroll
    for (int am = 0; am < ATOMS_M; ++am) {
        #pragma unroll
        for (int an = 0; an < ATOMS_N; ++an) {
            const int m_atom = am * ATOM_M;
            const int n_atom = an * ATOM_N + n_off_warp;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int dm = (p & 1) * 2;
                const int dn = ((p >> 1) & 1) * 4;
                const int gm = tile_m + m_atom + cL_m + dm;
                const int gn = tile_n + n_atom + cL_n + dn;
                if (gm < M && gn + 1 < N) {
                    C[(size_t) gm * N + gn + 0] = c_frag[am][an][p * 2 + 0];
                    C[(size_t) gm * N + gn + 1] = c_frag[am][an][p * 2 + 1];
                }
            }
        }
    }
}

}  // namespace int8_v11

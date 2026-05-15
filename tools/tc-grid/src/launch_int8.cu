#include "tc_grid.h"
#include "int8_kernels.cuh"
#include "int8_multifrag_kernels.cuh"
#include "int8_opt_kernels.cuh"
#include "v3_kernels.cuh"
#include "v4_kernels.cuh"
#include "v5_kernels.cuh"
#include "v6_kernels.cuh"
#include "v7_kernels.cuh"
#include "v8_kernels.cuh"
#include "v9_kernels.cuh"
#include "v10_kernels.cuh"
#include "v10splitk_kernels.cuh"
#include "v11_kernels.cuh"
#include "v12_kernels.cuh"
#include "v13_kernels.cuh"
#ifdef TCGRID_HAS_CUTLASS
#include "cutlass_int8_kernels.cuh"
#endif

#include <cstdio>
#include <vector>

namespace tc_grid {

static const int N_TIMING_ITERS = 8;
static const int N_WARMUP_ITERS = 2;

// Need both qs and scales pointers; we receive a single d_W pointer that
// points at the int8 qs base, and use rows*K offset for scales.
// W layout from quantize(): int8 qs[rows*K] then half scales[rows*blocks_per_row].
static inline const int8_t * w_qs_ptr(const void * d_W) {
    return (const int8_t *) d_W;
}
static inline const __half * w_scales_ptr(const void * d_W, int N, int K) {
    return (const __half *)((const char *) d_W + (size_t) N * K);
}

LaunchResult launch_int8(const LaunchSpec & s, const void * d_W,
                         const float * d_act, float * d_dst, const float * d_ref) {
    LaunchResult R;
    R.ok = false;
    R.note = "";

    const int M = s.M, N = s.N, K = s.K;
    const int BM = s.BM, BN = s.BN, BK = s.BK, WARPS = s.warps;

    if (BN != WARPS * s.frag_n * 16) { R.note = "BN != WARPS * frag_n * 16"; return R; }
    if (BK % 16 != 0) { R.note = "BK constraints (multiple of 16)"; return R; }
    // QK_INT8=32 commensurate: BK is a multiple of 32 OR 32 is a multiple of BK.
    if (BK % QK_INT8 != 0 && QK_INT8 % BK != 0) { R.note = "BK constraints (commensurate w/ QK_INT8)"; return R; }
    // Most kernels (v0..v10) require BK % QK_INT8 == 0; v11, v12, v12_ms3, v12s support BK<QK_INT8.
    if (BK < QK_INT8 && s.version != 11 && s.version != 50 && s.version != 51 && s.version != 60 && !(s.version >= 80 && s.version <= 91)) {
        R.note = "BK < QK_INT8 only supported by v11/v12/v12_ms3/v12s/v13_rf"; return R;
    }
    if (BM < 16) { R.note = "BM < 16 unsupported (WMMA fragment 16x16)"; return R; }
    if (N  % BN != 0 || K % BK != 0) { R.note = "NK %tile != 0"; return R; }

    const int num_tile_x = N / BN;
    const int num_tile_y = (M + BM - 1) / BM;
    const int num_tiles_total = num_tile_x * num_tile_y;
    // v5 uses persistent CTAs: grid = min(160, num_tiles_total).
    // Other versions use 1 CTA per tile.
    const int persistent_blocks = 160;  // 2 CTAs/SM × 80 SMs on V100
    const dim3 grid_normal((unsigned) num_tile_x, (unsigned) num_tile_y, 1);
    const dim3 grid_persistent((unsigned)((num_tiles_total < persistent_blocks) ? num_tiles_total : persistent_blocks), 1, 1);
    // v10s (SplitK) uses grid.z = split_k factor (passed via s.split_k).
    const int v10s_ks = (s.version == 20) ? std::max(1, s.split_k) : 1;
    const dim3 grid_v10s((unsigned) num_tile_x, (unsigned) num_tile_y, (unsigned) v10s_ks);
    // v12s (SplitK on v12 base) — same grid.z convention as v10s.
    const int v12s_ks = (s.version == 60) ? std::max(1, s.split_k) : 1;
    const dim3 grid_v12s((unsigned) num_tile_x, (unsigned) num_tile_y, (unsigned) v12s_ks);
    const dim3 grid = (s.version == 5)  ? grid_persistent
                    : (s.version == 20) ? grid_v10s
                    : (s.version == 60) ? grid_v12s
                    : grid_normal;
    const dim3 block((unsigned)(WARPS * 32), 1, 1);

    // CUTLASS P1 state: FP16 staging buffers (pre-dequanted, persists across reruns).
    // Declared here so the timing loop's RERUN_CUTLASS can see them. Initialized
    // to nullptr (value-initialized; safe across the goto). Freed after timing.
    struct CutlassState {
        __half * W_fp16 = nullptr;
        __half * A_fp16 = nullptr;
        int M = 0, N = 0, K = 0;
    } cs;

    const size_t smem_bytes_opt_v1 =
        (size_t)(BM * BK) * sizeof(__half)
      + (size_t)(BK * BN) * sizeof(__half);
    const size_t smem_bytes_opt_v2 = 2 * smem_bytes_opt_v1;
    const int BK_PAD = BK + 8;
    const size_t smem_bytes_opt_v3 = 2 * (
        (size_t)(BM * BK) * sizeof(__half)
      + (size_t)(BK_PAD * BN) * sizeof(__half));
    const size_t smem_bytes_opt_v4 = smem_bytes_opt_v3;  // same layout
    const size_t smem_bytes_opt_v5 = smem_bytes_opt_v3;  // same layout, persistent grid
    // v6: single-buffer with BK=64 padded. SMEM = BM*BK + BK_PAD*BN (single buf).
    const size_t smem_bytes_opt_v6 =
        (size_t)(BM * BK) * sizeof(__half)
      + (size_t)(BK_PAD * BN) * sizeof(__half);
    // v7: triple-buffer (3 SMEM stages).
    const size_t smem_bytes_opt_v7 = 3 * (
        (size_t)(BM * BK) * sizeof(__half)
      + (size_t)(BK_PAD * BN) * sizeof(__half));
    // v8: v3 layout + FP16 accumulator. SMEM unchanged from v3.
    const size_t smem_bytes_opt_v8 = smem_bytes_opt_v3;
    // v9: v3 layout + chunked FP16/FP32 mixed accumulator. SMEM unchanged from v3.
    const size_t smem_bytes_opt_v9 = smem_bytes_opt_v3;
    // v10: row-major B SMEM layout (BN_PAD inner stride instead of BK_PAD).
    const int BN_PAD_v10 = BN + 8;
    const size_t smem_bytes_opt_v10 = 2 * (
        (size_t)(BM * BK) * sizeof(__half)
      + (size_t)(BK * BN_PAD_v10) * sizeof(__half));
    // v11: col-major B SMEM (n outer, k inner). BK_PAD = BK + 8 so the stride is
    // a multiple of 8 halves (uint4-aligned for any n). Bank conflict is 4-way
    // (stride/2=20, GCD(20,32)=4), same as v3-class kernels.
    const int BK_PAD_v11 = BK + 8;
    const size_t smem_bytes_opt_v11 = 2 * (
        (size_t)(BM * BK) * sizeof(__half)
      + (size_t)(BN * BK_PAD_v11) * sizeof(__half));
    // v12 dyn SMEM = max(mainloop sA+sB, epilogue sC[BM][BN+8]).
    // sC reuses the sA+sB region after the last __syncthreads().
    const size_t smem_bytes_v12_mainloop = smem_bytes_opt_v11;        // same layout
    const size_t smem_bytes_v12_epilogue = (size_t)(BM * (BN + 8)) * sizeof(__half);
    const size_t smem_bytes_opt_v12 = (smem_bytes_v12_mainloop > smem_bytes_v12_epilogue)
        ? smem_bytes_v12_mainloop : smem_bytes_v12_epilogue;
    const size_t smem_bytes_base =
        smem_bytes_opt_v1
      + (size_t)(WARPS * BM * 16) * sizeof(float);
    size_t smem_bytes;
    if      (s.version == 20) smem_bytes = smem_bytes_opt_v10;  // v10s reuses v10 SMEM layout
    else if (s.version == 50) smem_bytes = smem_bytes_opt_v12;  // v12 fp16-acc + sC round-trip
    else if (s.version == 51) smem_bytes = smem_bytes_opt_v12;  // v12_ms3 uses same SMEM layout
    else if (s.version == 60) smem_bytes = smem_bytes_opt_v12;  // v12s same SMEM layout
    else if (s.version >= 80 && s.version <= 91) {
        // v13_rf family: sA[2*BM*BK]half + sB[2*BN*(BK+BK_PAD_)]int8 + sS[2*BN]half.
        // Stride is BK + pad (pad ∈ {16, 32, 48}). v=84 uses 16; v=85 uses 32; v=86 uses 48.
        int pad_ext = 16;
        if (s.version == 85) pad_ext = 32;
        else if (s.version == 86) pad_ext = 48;
        const size_t v13_mainloop =
            2 * (size_t)(BM * BK) * sizeof(__half)
          + 2 * (size_t)(BN * (BK + pad_ext)) * sizeof(int8_t)
          + 2 * (size_t)(BN)            * sizeof(__half);
        const size_t v13_epilogue = (size_t)(BM * (BN + 8)) * sizeof(__half);
        smem_bytes = (v13_mainloop > v13_epilogue) ? v13_mainloop : v13_epilogue;
    }
    else if (s.version == 11) smem_bytes = smem_bytes_opt_v11;
    else if (s.version == 10) smem_bytes = smem_bytes_opt_v10;
    else if (s.version == 9) smem_bytes = smem_bytes_opt_v9;
    else if (s.version == 8) smem_bytes = smem_bytes_opt_v8;
    else if (s.version == 7) smem_bytes = smem_bytes_opt_v7;
    else if (s.version == 6) smem_bytes = smem_bytes_opt_v6;
    else if (s.version == 5) smem_bytes = smem_bytes_opt_v5;
    else if (s.version == 4) smem_bytes = smem_bytes_opt_v4;
    else if (s.version == 3) smem_bytes = smem_bytes_opt_v3;
    else if (s.version == 2) smem_bytes = smem_bytes_opt_v2;
    else if (s.version == 1) smem_bytes = smem_bytes_opt_v1;
    else                     smem_bytes = smem_bytes_base;

    if (smem_bytes > 96 * 1024) { R.note = "smem > 96KiB"; return R; }
    // V100 supports up to 96KB dynamic SMEM per CTA via opt-in (default 48KB).
    // Trades L1 cache space for more SMEM — fair when SMEM is the bottleneck.
    if (s.version == 10 && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v10::mm_int8_lut_v10<128, 128, 64, 4, 8, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v10::mm_int8_lut_v10<128, 64, 64, 4, 8, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v10::mm_int8_lut_v10<64, 128, 64, 4, 4, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
    }
    if (s.version == 11 && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v11::mm_int8_lut_v11<128, 256, 32, 8, 16, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
    }
    if (s.version == 50 && smem_bytes > 48 * 1024) {
        // v12 needs >48 KB for the sC[BM][BN+8] half tile at large BN.
        // BM=128 BN=256 → sC = 128*264*2 = 66 KB > 48 KB.
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v12::mm_int8_lut_v12<128, 256, 16, 8, 16, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v12::mm_int8_lut_v12<128, 256, 32, 8, 16, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
    }
    if (s.version == 51 && smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v12::mm_int8_lut_v12_ms3<128, 256, 16, 8, 16, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v12::mm_int8_lut_v12_ms3<128, 256, 32, 8, 16, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
        // P4 §6.4: larger BM v12_ms3 (sC tile grows with BM).
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v12::mm_int8_lut_v12_ms3<192, 128, 16, 4, 24, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
        cudaFuncSetAttribute(
            (const void *) kernels::int8_v12::mm_int8_lut_v12_ms3<256, 128, 16, 4, 32, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
    }

    // Macro to dispatch on the compile-time template tile.
#define LAUNCH(b_m, b_n, b_k, w)                                                \
    if (BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w)) {                    \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8::mm_int8_lut<(b_m),(b_n),(b_k),(w)>                   \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
        } else {                                                                \
            kernels::int8::mm_int8_bitshift<(b_m),(b_n),(b_k),(w)>              \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
        }                                                                       \
        goto launched;                                                          \
    }

    // v8 (v3 base + FIXED bank-conflict padding: BK_PAD = BK + 6 (odd-element stride))
#define LAUNCH_V8(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==8 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v8::mm_int8_lut_v8<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v8 path-B not implemented"; return R; }              \
    }
    LAUNCH_V8(128, 128, 32, 4, 8, 2)
    LAUNCH_V8(128, 64,  32, 4, 8, 1)
    LAUNCH_V8(64,  128, 32, 4, 4, 2)
    LAUNCH_V8(64,  64,  32, 4, 4, 1)
    LAUNCH_V8(32,  128, 32, 4, 2, 2)
    LAUNCH_V8(32,  64,  32, 4, 2, 1)
#undef LAUNCH_V8

    // v9 (v3 layout + chunked FP16/FP32 mixed-precision accumulator, CHUNK_K templated)
#define LAUNCH_V9(b_m, b_n, b_k, w, fm, fn, ck)                                 \
    if (s.version==9 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && s.split_k==(ck)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v9::mm_int8_lut_v9a<(b_m),(b_n),(b_k),(w),(fm),(fn),(ck)> \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v9 path-B not implemented"; return R; }              \
    }
    // CHUNK_K sweep: 2, 4, 8 — selected via s.split_k field
    LAUNCH_V9(128, 128, 32, 4, 8, 2, 2)
    LAUNCH_V9(128, 128, 32, 4, 8, 2, 4)
    LAUNCH_V9(128, 128, 32, 4, 8, 2, 8)
    LAUNCH_V9(64,  128, 32, 4, 4, 2, 4)
    LAUNCH_V9(64,  64,  32, 4, 4, 1, 4)
#undef LAUNCH_V9

    // v10 (v4 base + row-major B SMEM)
#define LAUNCH_V10(b_m, b_n, b_k, w, fm, fn)                                    \
    if (s.version==10 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v10::mm_int8_lut_v10<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v10 path-B not implemented"; return R; }             \
    }
    LAUNCH_V10(128, 128, 32, 4, 8, 2)
    LAUNCH_V10(128, 64,  32, 4, 8, 1)
    LAUNCH_V10(64,  128, 32, 4, 4, 2)
    LAUNCH_V10(64,  64,  32, 4, 4, 1)
    LAUNCH_V10(64,  256, 32, 8, 4, 2)
    LAUNCH_V10(128, 256, 32, 8, 8, 2)
    LAUNCH_V10(128, 128, 64, 4, 8, 2)
    LAUNCH_V10(128, 64,  64, 4, 8, 1)
    LAUNCH_V10(64,  128, 64, 4, 4, 2)
#undef LAUNCH_V10

    // v11 (v10 base + manual Lds + SM70_MMA_884 atoms). Supports shapes where
    // N_PER_WARP = BN/WARPS >= 32 (= ATOM_N). atoms_m = fm*2 (884 m-step is 8,
    // wmma was 16), atoms_n = fn/2 (884 n-step is 32, wmma was 16).
#define LAUNCH_V11(b_m, b_n, b_k, w, fm, fn)                                    \
    if (s.version==11 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
        constexpr int am = (fm) * 2;                                            \
        constexpr int an = (fn) / 2;                                            \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v11::mm_int8_lut_v11<(b_m),(b_n),(b_k),(w),am,an>     \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v11 path-B not implemented"; return R; }             \
    }
    LAUNCH_V11(128, 128, 32, 4, 8, 2)
    LAUNCH_V11(64,  128, 32, 4, 4, 2)
    LAUNCH_V11(128, 256, 32, 8, 8, 2)
    LAUNCH_V11(64,  256, 32, 8, 4, 2)
    // BK=16: turbomind's shipped CTA_K on sm_70 — half the K-tile size for tighter
    // pipelining. (QK_INT8=32 still divides cleanly because each lane reads a
    // half-block of K=16; the scale lookup uses gk/QK_INT8 which is correct.)
    LAUNCH_V11(128, 128, 16, 4, 8, 2)
    LAUNCH_V11(64,  128, 16, 4, 4, 2)
    LAUNCH_V11(128, 256, 16, 8, 8, 2)
    LAUNCH_V11(64,  256, 16, 8, 4, 2)
    // Experimental: try BN=256 with W=4 (larger N_PER_WARP=64 → ATOMS_N=2)
    LAUNCH_V11(64,  256, 16, 4, 4, 4)
    LAUNCH_V11(128, 256, 16, 4, 8, 4)
    // Grid search: smaller BM (high occupancy)
    LAUNCH_V11(32,  128, 16, 4, 2, 2)
    LAUNCH_V11(32,  256, 16, 8, 2, 2)
    LAUNCH_V11(32,  128, 32, 4, 2, 2)
    // Grid search: larger BM
    LAUNCH_V11(192, 128, 16, 4, 12, 2)
    LAUNCH_V11(256, 128, 16, 4, 16, 2)
    // Grid search: BK=64 (more inner-K iterations, fewer outer-K loop overhead)
    LAUNCH_V11(128, 128, 64, 4, 8, 2)
    LAUNCH_V11(64,  128, 64, 4, 4, 2)
    LAUNCH_V11(64,  256, 64, 8, 4, 2)
    // Grid search: W=2 (fewer warps per CTA, more CTAs)
    LAUNCH_V11(64,  128, 16, 2, 4, 4)
    LAUNCH_V11(64,  128, 32, 2, 4, 4)
    LAUNCH_V11(32,  128, 16, 2, 2, 4)
#undef LAUNCH_V11

    // v12 = v11 + FP16 accumulator + SMEM round-trip epilogue (SPRINT-019 P1).
    // c_frag halved in storage (8 floats → 8 halves per atom-frag per lane).
    // The empirically-derived f16-acc lane mapping (V12-DESIGN.md §2) gives a
    // contiguous 1×8 n-strip per lane, so the epilogue is one uint4 store
    // per (am, an) per lane to sC, then a cooperative half2→fp32 STG.
#define LAUNCH_V12(b_m, b_n, b_k, w, fm, fn)                                    \
    if (s.version==50 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
        constexpr int am = (fm) * 2;                                            \
        constexpr int an = (fn) / 2;                                            \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v12::mm_int8_lut_v12<(b_m),(b_n),(b_k),(w),am,an>     \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v12 path-B not implemented"; return R; }             \
    }
    // Champion-class tiles (mirror the v11 set at BK=16 first; expand in P1.4).
    LAUNCH_V12(128, 128, 16, 4, 8, 2)
    LAUNCH_V12(64,  128, 16, 4, 4, 2)
    LAUNCH_V12(128, 256, 16, 8, 8, 2)
    LAUNCH_V12(64,  256, 16, 8, 4, 2)
    LAUNCH_V12(128, 128, 32, 4, 8, 2)
    LAUNCH_V12(64,  128, 32, 4, 4, 2)
    LAUNCH_V12(192, 128, 16, 4, 12, 2)
    LAUNCH_V12(256, 128, 16, 4, 16, 2)
    LAUNCH_V12(32,  128, 16, 4, 2, 2)
    LAUNCH_V12(32,  256, 16, 8, 2, 2)
    LAUNCH_V12(64,  128, 16, 2, 4, 4)
    LAUNCH_V12(64,  256, 16, 4, 4, 4)
#undef LAUNCH_V12

    // v12_ms3 = v12 + 3-stage pipeline (LDG → rmem → STS → smem → mma).
    // SPRINT-019 P2 / §6.2. Adds ~21 regs/thread persistent rmem; needs
    // v12's 133-reg footprint to fit launch_bounds(2). sprint-017's 3-stage
    // on v11 (194 regs) cratered the BM=128 champ; v12's relief should fix.
#define LAUNCH_V12_MS3(b_m, b_n, b_k, w, fm, fn)                                \
    if (s.version==51 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
        constexpr int am = (fm) * 2;                                            \
        constexpr int an = (fn) / 2;                                            \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v12::mm_int8_lut_v12_ms3<(b_m),(b_n),(b_k),(w),am,an> \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v12_ms3 path-B not implemented"; return R; }         \
    }
    LAUNCH_V12_MS3(128, 128, 16, 4, 8, 2)
    LAUNCH_V12_MS3(64,  128, 16, 4, 4, 2)
    LAUNCH_V12_MS3(128, 256, 16, 8, 8, 2)
    LAUNCH_V12_MS3(64,  256, 16, 8, 4, 2)
    LAUNCH_V12_MS3(128, 128, 32, 4, 8, 2)
    LAUNCH_V12_MS3(64,  128, 32, 4, 4, 2)
    LAUNCH_V12_MS3(64,  256, 16, 4, 4, 4)
    LAUNCH_V12_MS3(64,  128, 16, 2, 4, 4)
    // P4 §6.4: larger BM via v12_ms3 register relief (no explicit SMEM spill yet).
    // c_frag at BM=192 = 24*1*8 = 192 halves = 96 regs (vs 64 at BM=128). +ms3
    // rmem ~19 regs. Total ≈ 130+ regs; still under launch_bounds(2).
    LAUNCH_V12_MS3(192, 128, 16, 4, 12, 2)
    LAUNCH_V12_MS3(256, 128, 16, 4, 16, 2)
#undef LAUNCH_V12_MS3

    // v13_rf = v12_ms3 with register-file dequant (SPRINT-021 P1).
    // sB stores raw INT8 (1B/wt) instead of dequanted FP16 (2B/wt).
    // Mainloop LDS 8 INT8 per lane + PRMT-dequant + mma.
    // Goal: lift HMMA active% from 31.7% toward 50%+.
    // Prototype: BK=16 only (one-scale-per-tile assumption).
#define LAUNCH_V13_RF(b_m, b_n, b_k, w, fm, fn)                                 \
    if (s.version==80 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
        constexpr int am = (fm) * 2;                                            \
        constexpr int an = (fn) / 2;                                            \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v13::mm_int8_lut_v13_rf<(b_m),(b_n),(b_k),(w),am,an>  \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v13_rf path-B not implemented"; return R; }          \
    }
    LAUNCH_V13_RF(128, 128, 16, 4, 8, 2)
    LAUNCH_V13_RF( 64, 128, 16, 4, 4, 2)
#undef LAUNCH_V13_RF

    // v13_rf_v2: K-iter fusion. Single uint4 (16 INT8) LDS per lane per
    // atom; 4× PRMT → 16 halves; 4 back-to-back m8n8k4 mma's. version=81.
#define LAUNCH_V13_RF_V2(b_m, b_n, b_k, w, fm, fn)                              \
    if (s.version==81 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
        constexpr int am = (fm) * 2;                                            \
        constexpr int an = (fn) / 2;                                            \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v13::mm_int8_lut_v13_rf_v2<(b_m),(b_n),(b_k),(w),am,an> \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v13_rf_v2 path-B not implemented"; return R; }       \
    }
    LAUNCH_V13_RF_V2(128, 128, 16, 4, 8, 2)
    LAUNCH_V13_RF_V2( 64, 128, 16, 4, 4, 2)
#undef LAUNCH_V13_RF_V2

    // v13_rf_v3: software pipeline within mainloop (LDS for ki+1 issued
    // before mma of ki). version=82.
#define LAUNCH_V13_RF_V3(b_m, b_n, b_k, w, fm, fn)                              \
    if (s.version==82 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
        constexpr int am = (fm) * 2;                                            \
        constexpr int an = (fn) / 2;                                            \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v13::mm_int8_lut_v13_rf_v3<(b_m),(b_n),(b_k),(w),am,an> \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v13_rf_v3 path-B not implemented"; return R; }       \
    }
    LAUNCH_V13_RF_V3(128, 128, 16, 4, 8, 2)
    LAUNCH_V13_RF_V3( 64, 128, 16, 4, 4, 2)
#undef LAUNCH_V13_RF_V3

    // v13_rf_v4: launch_bounds(*, 1) — higher reg cap (256/thread). v=83.
#define LAUNCH_V13_RF_V4(b_m, b_n, b_k, w, fm, fn)                              \
    if (s.version==83 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
        constexpr int am = (fm) * 2;                                            \
        constexpr int an = (fn) / 2;                                            \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v13::mm_int8_lut_v13_rf_v4<(b_m),(b_n),(b_k),(w),am,an> \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v13_rf_v4 path-B not implemented"; return R; }       \
    }
    LAUNCH_V13_RF_V4(128, 128, 16, 4, 8, 2)
    LAUNCH_V13_RF_V4( 64, 128, 16, 4, 4, 2)
#undef LAUNCH_V13_RF_V4

    // v13_rf_v5: 2x2 warp partition (turbomind Blocked<2,2>). version=84.
    // Fixed BM=BN=128 WARPS=4. ATOMS_M=8 ATOMS_N=2 per warp (vs 16x1 in v4).
    if (s.version == 84 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 128, 16, 16>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }
    // v13_rf_v5 BK_PAD=32 (stride 48). version=85.
    if (s.version == 85 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 128, 16, 32>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }
    // v13_rf_v5 BK_PAD=48 (stride 64). version=86.
    if (s.version == 86 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 128, 16, 48>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }
    // v13_rf_v5 BM=128 BN=256 (each warp covers 64×128 = ATOMS_M=8 × ATOMS_N=4). version=87.
    if (s.version == 87 && BM == 128 && BN == 256 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            cudaFuncSetAttribute(
                (const void *) kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 256, 16, 16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 256, 16, 16>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }
    // v13_rf_v6: v5 + K-iter LDS fusion (uint4 covering all 16 K). v=88.
    if (s.version == 88 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            kernels::int8_v13::mm_int8_lut_v13_rf_v6<128, 128, 16, 16>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }
    // v13_rf_v6 BM=64 (mid-M coverage). v=88, but BM=64 BN=128.
    if (s.version == 88 && BM == 64 && BN == 128 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            kernels::int8_v13::mm_int8_lut_v13_rf_v6<64, 128, 16, 16>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }
    // v13_rf_v7: v5 + mma reorder (independence-maximizing). v=89.
    if (s.version == 89 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            kernels::int8_v13::mm_int8_lut_v13_rf_v7<128, 128, 16, 16>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }
    // v13_rf_v8: v6 + launch_bounds(*, 2) for higher occupancy. v=90.
    if (s.version == 90 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            kernels::int8_v13::mm_int8_lut_v13_rf_v8<128, 128, 16, 16>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }
    // v13_rf_v6 BN=192 (ATOMS_N=3 per warp). v=91.
    if (s.version == 91 && BM == 128 && BN == 192 && BK == 16 && WARPS == 4) {
        if (s.path == UnpackPath::LUT) {
            cudaFuncSetAttribute(
                (const void *) kernels::int8_v13::mm_int8_lut_v13_rf_v6<128, 192, 16, 16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            kernels::int8_v13::mm_int8_lut_v13_rf_v6<128, 192, 16, 16>
                <<<grid, block, smem_bytes>>>(
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                    d_act, d_dst, M, N, K);
            goto launched;
        }
    }

    // v10s (v10 base + SplitK). Grid.z = ks. C must be pre-zeroed when ks > 1.
#define LAUNCH_V10S(b_m, b_n, b_k, w, fm, fn, ks)                                \
    if (s.version==20 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && s.split_k==(ks)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            if ((ks) > 1) cudaMemsetAsync(d_dst, 0, (size_t) M * N * sizeof(float)); \
            kernels::int8_v10s::mm_int8_lut_v10s<(b_m),(b_n),(b_k),(w),(fm),(fn),(ks)> \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v10s path-B not implemented"; return R; }            \
    }
    LAUNCH_V10S(128, 128, 32, 4, 8, 2, 1)
    LAUNCH_V10S(128, 128, 32, 4, 8, 2, 2)
    LAUNCH_V10S(128, 128, 32, 4, 8, 2, 4)
    LAUNCH_V10S(128, 128, 32, 4, 8, 2, 8)
    LAUNCH_V10S(64,  128, 32, 4, 4, 2, 2)
    LAUNCH_V10S(64,  128, 32, 4, 4, 2, 4)
    LAUNCH_V10S(64,  128, 32, 4, 4, 2, 8)
    LAUNCH_V10S(64,  64,  32, 4, 4, 1, 2)
    LAUNCH_V10S(64,  64,  32, 4, 4, 1, 4)
    LAUNCH_V10S(64,  64,  32, 4, 4, 1, 8)
    LAUNCH_V10S(64,  256, 32, 8, 4, 2, 2)
    LAUNCH_V10S(64,  256, 32, 8, 4, 2, 4)
#undef LAUNCH_V10S

    // v12s = v12 base + SplitK. grid.z = s.split_k. C pre-zeroed when ks > 1.
    // Targets small-M (decode-style) production. SPRINT-019 P3 §6.3.
#define LAUNCH_V12S(b_m, b_n, b_k, w, fm, fn, ks)                                \
    if (s.version==60 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && s.split_k==(ks) && ((fn) % 2 == 0)) { \
        constexpr int am = (fm) * 2;                                            \
        constexpr int an = (fn) / 2;                                            \
        if (s.path == UnpackPath::LUT) {                                        \
            if ((ks) > 1) cudaMemsetAsync(d_dst, 0, (size_t) M * N * sizeof(float)); \
            kernels::int8_v12::mm_int8_lut_v12s<(b_m),(b_n),(b_k),(w),am,an,(ks)> \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v12s path-B not implemented"; return R; }            \
    }
    // Small-M decode targets (BM=64 BN=128 W=4 ATOMS_M=8 ATOMS_N=1).
    LAUNCH_V12S(64,  128, 16, 4, 4, 2, 1)
    LAUNCH_V12S(64,  128, 16, 4, 4, 2, 2)
    LAUNCH_V12S(64,  128, 16, 4, 4, 2, 4)
    LAUNCH_V12S(64,  128, 16, 4, 4, 2, 8)
    LAUNCH_V12S(64,  128, 16, 4, 4, 2, 16)
    LAUNCH_V12S(64,  128, 32, 4, 4, 2, 2)
    LAUNCH_V12S(64,  128, 32, 4, 4, 2, 4)
    LAUNCH_V12S(64,  128, 32, 4, 4, 2, 8)
    // Adversarial KSPLIT (P3.1 §1.2 item 2): non-power-of-2 for tile-boundary checks
    LAUNCH_V12S(64,  128, 16, 4, 4, 2, 3)
    LAUNCH_V12S(64,  128, 16, 4, 4, 2, 5)
    // Larger BM at higher M (mid-band)
    LAUNCH_V12S(128, 128, 16, 4, 8, 2, 2)
    LAUNCH_V12S(128, 128, 16, 4, 8, 2, 4)
#undef LAUNCH_V12S

#ifdef TCGRID_HAS_CUTLASS
    // CUTLASS P1: pre-dequant W and A to FP16 once, then call cutlass Gemm.
    // version=40, tile params (BM, BN, BK) map to (CTA_M, CTA_N, CTA_K);
    // (frag_n) packed into a warp-shape choice via convention:
    //   frag_n=2 -> WarpShape 64x64x32 (4 warps per CTA)
    //   frag_n=1 -> WarpShape 32x64x32
    //
    // For P1 we only ship a small set; P3 will expand.
#define LAUNCH_CUTLASS_P1(cta_m, cta_n, cta_k, w_m, w_n, w_k)                  \
    if (s.version == 40 && BM == (cta_m) && BN == (cta_n) && BK == (cta_k)) {  \
        if (cs.W_fp16 == nullptr) {                                            \
            cs.M = M; cs.N = N; cs.K = K;                                      \
            cudaMalloc(&cs.W_fp16, (size_t) N * K * sizeof(__half));           \
            cudaMalloc(&cs.A_fp16, (size_t) M * K * sizeof(__half));           \
            dim3 g_w((K + 255) / 256, N);                                      \
            kernels::int8_cutlass::cast_int8_to_fp16                           \
                <<<g_w, 256>>>(w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),         \
                              cs.W_fp16, N, K);                                \
            dim3 g_a((K + 255) / 256, M);                                      \
            kernels::int8_cutlass::cast_f32_to_fp16                            \
                <<<g_a, 256>>>(d_act, cs.A_fp16, M, K);                        \
        }                                                                       \
        cutlass::Status st = kernels::int8_cutlass::Gemm70<                    \
            (cta_m),(cta_n),(cta_k),(w_m),(w_n),(w_k)                          \
        >::run(cs.A_fp16, cs.W_fp16, d_dst, M, N, K);                          \
        if (st != cutlass::Status::kSuccess) {                                 \
            R.note = "cutlass kSuccess != status"; cudaFree(cs.W_fp16);        \
            cudaFree(cs.A_fp16); return R;                                     \
        }                                                                       \
        goto launched;                                                          \
    }
    LAUNCH_CUTLASS_P1(128, 128, 32, 64, 64, 32)
    LAUNCH_CUTLASS_P1(64,  128, 32, 32, 64, 32)
    LAUNCH_CUTLASS_P1(128, 64,  32, 64, 32, 32)
#undef LAUNCH_CUTLASS_P1
#endif

    // v7 (B3: 3-stage triple-buffer; SMEM-constrained to small tiles)
#define LAUNCH_V7(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==7 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v7::mm_int8_lut_v7<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v7 path-B not implemented"; return R; }              \
    }
    LAUNCH_V7(64,  128, 32, 4, 4, 2)
    LAUNCH_V7(64,  64,  32, 4, 4, 1)
    LAUNCH_V7(32,  128, 32, 4, 2, 2)
    LAUNCH_V7(32,  64,  32, 4, 2, 1)
#undef LAUNCH_V7

    // v6 (BK=64 single-buffer + L2 prefetch; A1-retry without double-buffer)
#define LAUNCH_V6(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==6 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v6::mm_int8_lut_v6<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v6 path-B not implemented"; return R; }              \
    }
    LAUNCH_V6(128, 128, 64, 4, 8, 2)
    LAUNCH_V6(128, 64,  64, 4, 8, 1)
    LAUNCH_V6(64,  128, 64, 4, 4, 2)
    LAUNCH_V6(64,  64,  64, 4, 4, 1)
#undef LAUNCH_V6

    // v5 (v3 + Tier A: persistent CTAs + L2 prefetch + __ldg + __launch_bounds__)
#define LAUNCH_V5(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==5 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v5::mm_int8_lut_v5<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K, num_tile_x, num_tile_y);             \
            goto launched;                                                      \
        } else { R.note = "v5 path-B not implemented"; return R; }              \
    }
    LAUNCH_V5(128, 128, 32, 4, 8, 2)
    LAUNCH_V5(128, 64,  32, 4, 8, 1)
    LAUNCH_V5(64,  128, 32, 4, 4, 2)
    LAUNCH_V5(64,  64,  32, 4, 4, 1)
#undef LAUNCH_V5

    // v4 (v3 + Tier S: L2 prefetch + __launch_bounds__ + __ldg)
#define LAUNCH_V4(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==4 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v4::mm_int8_lut_v4<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v4 path-B not implemented"; return R; }              \
    }
    LAUNCH_V4(128, 128, 32, 4, 8, 2)
    LAUNCH_V4(128, 64,  32, 4, 8, 1)
    LAUNCH_V4(64,  128, 32, 4, 4, 2)
    LAUNCH_V4(64,  64,  32, 4, 4, 1)
#undef LAUNCH_V4

    // v3 (v2 + padded B for bank-conflict-free SMEM)
#define LAUNCH_V3(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==3 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_v3::mm_int8_lut_v3<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v3 path-B not implemented"; return R; }              \
    }
    LAUNCH_V3(128, 128, 32, 4, 8, 2)
    LAUNCH_V3(128, 64,  32, 4, 8, 1)
    LAUNCH_V3(128, 256, 32, 4, 8, 4)
    LAUNCH_V3(128, 256, 32, 8, 8, 2)
    LAUNCH_V3(64,  128, 32, 4, 4, 2)
    LAUNCH_V3(64,  64,  32, 4, 4, 1)
    // B1: FRAG_M=2 (BM=32) high-occupancy tiles
    LAUNCH_V3(32,  128, 32, 4, 2, 2)
    LAUNCH_V3(32,  64,  32, 4, 2, 1)
    LAUNCH_V3(32,  256, 32, 8, 2, 2)
    // B1 alt: 8 warps/CTA (requires BN >= 128 since 8*1*16=128)
    LAUNCH_V3(128, 128, 32, 8, 8, 1)
    LAUNCH_V3(64,  128, 32, 8, 4, 1)
#undef LAUNCH_V3

    // v2 (BM=128 multi-A-frag + double-buffer + uint4 loads)
#define LAUNCH_V2(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==2 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_opt::mm_int8_lut_v2<(b_m),(b_n),(b_k),(w),(fm),(fn)>  \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v2 path-B not implemented"; return R; }              \
    }
    LAUNCH_V2(128, 128, 32, 4, 8, 2)
    LAUNCH_V2(128, 64,  32, 4, 8, 1)
    LAUNCH_V2(128, 256, 32, 4, 8, 4)
    LAUNCH_V2(128, 256, 32, 8, 8, 2)
    LAUNCH_V2(64,  128, 32, 4, 4, 2)
    LAUNCH_V2(64,  64,  32, 4, 4, 1)
#undef LAUNCH_V2

    // v1 (BM=128 multi-A-frag, single buffer)
#define LAUNCH_V1(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==1 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_opt::mm_int8_lut_v1<(b_m),(b_n),(b_k),(w),(fm),(fn)>  \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "v1 path-B not implemented"; return R; }              \
    }
    LAUNCH_V1(128, 128, 32, 4, 8, 2)
    LAUNCH_V1(128, 64,  32, 4, 8, 1)
    LAUNCH_V1(128, 128, 64, 4, 8, 2)
    LAUNCH_V1(128, 256, 32, 4, 8, 4)
    LAUNCH_V1(128, 256, 32, 8, 8, 2)
    LAUNCH_V1(64,  128, 32, 4, 4, 2)
    LAUNCH_V1(64,  64,  32, 4, 4, 1)
#undef LAUNCH_V1

    // Multi-fragment dispatch (path-A LUT only; path-B not implemented for MF).
#define LAUNCH_MF(b_m, b_n, b_k, w, fn)                                         \
    if (BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) {  \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int8_mf::mm_int8_lut_mf<(b_m),(b_n),(b_k),(w),(fn)>        \
                <<<grid, block, smem_bytes>>>(                                   \
                    w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                     \
                    d_act, d_dst, M, N, K);                                     \
            goto launched;                                                      \
        } else { R.note = "MF path-B not implemented"; return R; }              \
    }
    LAUNCH_MF(16, 128, 32, 4, 2)
    LAUNCH_MF(16, 128, 64, 4, 2)
    LAUNCH_MF(16, 256, 32, 4, 4)
    LAUNCH_MF(16, 256, 64, 4, 4)
    LAUNCH_MF(16, 256, 32, 8, 2)
    LAUNCH_MF(16, 512, 32, 8, 4)
#undef LAUNCH_MF

    LAUNCH(16, 64, 32, 4)
    LAUNCH(16, 64, 64, 4)
    LAUNCH(16, 128, 32, 8)
    LAUNCH(16, 128, 64, 8)
    R.note = "no template instantiation for this tile"; return R;
launched:

    // Warmup + timing
    cudaEvent_t evs[2]; cudaEventCreate(&evs[0]); cudaEventCreate(&evs[1]);
    for (int i = 0; i < N_WARMUP_ITERS; ++i) {
        // Re-launch warmup variant
        // (template-specialized launches above; this just re-runs the kernel as a heat-up).
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { R.note = cudaGetErrorString(err); return R; }
    TCG_CHECK(cudaDeviceSynchronize());

    // Timing loop -- re-trigger via the dispatched template
    std::vector<float> times;
    times.reserve(N_TIMING_ITERS);
    for (int i = 0; i < N_WARMUP_ITERS + N_TIMING_ITERS; ++i) {
        cudaEventRecord(evs[0]);
        // Re-dispatch (same shape) so we can time multiple iterations.
#define RERUN(b_m, b_n, b_k, w)                                                  \
        if (BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w)) {                 \
            if (s.path == UnpackPath::LUT) {                                     \
                kernels::int8::mm_int8_lut<(b_m),(b_n),(b_k),(w)>                \
                    <<<grid, block, smem_bytes>>>(                                \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                  \
                        d_act, d_dst, M, N, K);                                  \
            } else {                                                              \
                kernels::int8::mm_int8_bitshift<(b_m),(b_n),(b_k),(w)>           \
                    <<<grid, block, smem_bytes>>>(                                \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                  \
                        d_act, d_dst, M, N, K);                                  \
            }                                                                     \
        }
#define RERUN_V8(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==8 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v8::mm_int8_lut_v8<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V8(128, 128, 32, 4, 8, 2)
        RERUN_V8(128, 64,  32, 4, 8, 1)
        RERUN_V8(64,  128, 32, 4, 4, 2)
        RERUN_V8(64,  64,  32, 4, 4, 1)
        RERUN_V8(32,  128, 32, 4, 2, 2)
        RERUN_V8(32,  64,  32, 4, 2, 1)
#undef RERUN_V8
#define RERUN_V9(b_m, b_n, b_k, w, fm, fn, ck)                                  \
        if (s.version==9 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && s.split_k==(ck)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v9::mm_int8_lut_v9a<(b_m),(b_n),(b_k),(w),(fm),(fn),(ck)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V9(128, 128, 32, 4, 8, 2, 2)
        RERUN_V9(128, 128, 32, 4, 8, 2, 4)
        RERUN_V9(128, 128, 32, 4, 8, 2, 8)
        RERUN_V9(64,  128, 32, 4, 4, 2, 4)
        RERUN_V9(64,  64,  32, 4, 4, 1, 4)
#undef RERUN_V9
#define RERUN_V10(b_m, b_n, b_k, w, fm, fn)                                     \
        if (s.version==10 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v10::mm_int8_lut_v10<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V10(128, 128, 32, 4, 8, 2)
        RERUN_V10(128, 64,  32, 4, 8, 1)
        RERUN_V10(64,  128, 32, 4, 4, 2)
        RERUN_V10(64,  64,  32, 4, 4, 1)
        RERUN_V10(64,  256, 32, 8, 4, 2)
        RERUN_V10(128, 256, 32, 8, 8, 2)
        RERUN_V10(128, 128, 64, 4, 8, 2)
        RERUN_V10(128, 64,  64, 4, 8, 1)
        RERUN_V10(64,  128, 64, 4, 4, 2)
#undef RERUN_V10
#define RERUN_V11(b_m, b_n, b_k, w, fm, fn)                                     \
        if (s.version==11 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
            constexpr int am = (fm) * 2;                                        \
            constexpr int an = (fn) / 2;                                        \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v11::mm_int8_lut_v11<(b_m),(b_n),(b_k),(w),am,an> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V11(128, 128, 32, 4, 8, 2)
        RERUN_V11(64,  128, 32, 4, 4, 2)
        RERUN_V11(128, 256, 32, 8, 8, 2)
        RERUN_V11(64,  256, 32, 8, 4, 2)
        RERUN_V11(128, 128, 16, 4, 8, 2)
        RERUN_V11(64,  128, 16, 4, 4, 2)
        RERUN_V11(128, 256, 16, 8, 8, 2)
        RERUN_V11(64,  256, 16, 8, 4, 2)
        RERUN_V11(64,  256, 16, 4, 4, 4)
        RERUN_V11(128, 256, 16, 4, 8, 4)
        RERUN_V11(32,  128, 16, 4, 2, 2)
        RERUN_V11(32,  256, 16, 8, 2, 2)
        RERUN_V11(32,  128, 32, 4, 2, 2)
        RERUN_V11(192, 128, 16, 4, 12, 2)
        RERUN_V11(256, 128, 16, 4, 16, 2)
        RERUN_V11(128, 128, 64, 4, 8, 2)
        RERUN_V11(64,  128, 64, 4, 4, 2)
        RERUN_V11(64,  256, 64, 8, 4, 2)
        RERUN_V11(64,  128, 16, 2, 4, 4)
        RERUN_V11(64,  128, 32, 2, 4, 4)
        RERUN_V11(32,  128, 16, 2, 2, 4)
#undef RERUN_V11
#define RERUN_V12(b_m, b_n, b_k, w, fm, fn)                                     \
        if (s.version==50 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
            constexpr int am = (fm) * 2;                                        \
            constexpr int an = (fn) / 2;                                        \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v12::mm_int8_lut_v12<(b_m),(b_n),(b_k),(w),am,an> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V12(128, 128, 16, 4, 8, 2)
        RERUN_V12(64,  128, 16, 4, 4, 2)
        RERUN_V12(128, 256, 16, 8, 8, 2)
        RERUN_V12(64,  256, 16, 8, 4, 2)
        RERUN_V12(128, 128, 32, 4, 8, 2)
        RERUN_V12(64,  128, 32, 4, 4, 2)
        RERUN_V12(192, 128, 16, 4, 12, 2)
        RERUN_V12(256, 128, 16, 4, 16, 2)
        RERUN_V12(32,  128, 16, 4, 2, 2)
        RERUN_V12(32,  256, 16, 8, 2, 2)
        RERUN_V12(64,  128, 16, 2, 4, 4)
        RERUN_V12(64,  256, 16, 4, 4, 4)
#undef RERUN_V12
#define RERUN_V12_MS3(b_m, b_n, b_k, w, fm, fn)                                 \
        if (s.version==51 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
            constexpr int am = (fm) * 2;                                        \
            constexpr int an = (fn) / 2;                                        \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v12::mm_int8_lut_v12_ms3<(b_m),(b_n),(b_k),(w),am,an> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V12_MS3(128, 128, 16, 4, 8, 2)
        RERUN_V12_MS3(64,  128, 16, 4, 4, 2)
        RERUN_V12_MS3(128, 256, 16, 8, 8, 2)
        RERUN_V12_MS3(64,  256, 16, 8, 4, 2)
        RERUN_V12_MS3(128, 128, 32, 4, 8, 2)
        RERUN_V12_MS3(64,  128, 32, 4, 4, 2)
        RERUN_V12_MS3(64,  256, 16, 4, 4, 4)
        RERUN_V12_MS3(64,  128, 16, 2, 4, 4)
        RERUN_V12_MS3(192, 128, 16, 4, 12, 2)
        RERUN_V12_MS3(256, 128, 16, 4, 16, 2)
#undef RERUN_V12_MS3
#define RERUN_V13_RF(b_m, b_n, b_k, w, fm, fn)                                  \
        if (s.version==80 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
            constexpr int am = (fm) * 2;                                        \
            constexpr int an = (fn) / 2;                                        \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v13::mm_int8_lut_v13_rf<(b_m),(b_n),(b_k),(w),am,an> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V13_RF(128, 128, 16, 4, 8, 2)
        RERUN_V13_RF( 64, 128, 16, 4, 4, 2)
#undef RERUN_V13_RF
#define RERUN_V13_RF_V2(b_m, b_n, b_k, w, fm, fn)                               \
        if (s.version==81 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
            constexpr int am = (fm) * 2;                                        \
            constexpr int an = (fn) / 2;                                        \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v13::mm_int8_lut_v13_rf_v2<(b_m),(b_n),(b_k),(w),am,an> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V13_RF_V2(128, 128, 16, 4, 8, 2)
        RERUN_V13_RF_V2( 64, 128, 16, 4, 4, 2)
#undef RERUN_V13_RF_V2
#define RERUN_V13_RF_V3(b_m, b_n, b_k, w, fm, fn)                               \
        if (s.version==82 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
            constexpr int am = (fm) * 2;                                        \
            constexpr int an = (fn) / 2;                                        \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v13::mm_int8_lut_v13_rf_v3<(b_m),(b_n),(b_k),(w),am,an> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V13_RF_V3(128, 128, 16, 4, 8, 2)
        RERUN_V13_RF_V3( 64, 128, 16, 4, 4, 2)
#undef RERUN_V13_RF_V3
#define RERUN_V13_RF_V4(b_m, b_n, b_k, w, fm, fn)                               \
        if (s.version==83 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && ((fn) % 2 == 0)) { \
            constexpr int am = (fm) * 2;                                        \
            constexpr int an = (fn) / 2;                                        \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v13::mm_int8_lut_v13_rf_v4<(b_m),(b_n),(b_k),(w),am,an> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V13_RF_V4(128, 128, 16, 4, 8, 2)
        RERUN_V13_RF_V4( 64, 128, 16, 4, 4, 2)
#undef RERUN_V13_RF_V4
        if (s.version == 84 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 128, 16, 16>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
        if (s.version == 85 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 128, 16, 32>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
        if (s.version == 86 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 128, 16, 48>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
        if (s.version == 87 && BM == 128 && BN == 256 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v5<128, 256, 16, 16>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
        if (s.version == 88 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v6<128, 128, 16, 16>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
        if (s.version == 88 && BM == 64 && BN == 128 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v6<64, 128, 16, 16>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
        if (s.version == 89 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v7<128, 128, 16, 16>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
        if (s.version == 90 && BM == 128 && BN == 128 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v8<128, 128, 16, 16>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
        if (s.version == 91 && BM == 128 && BN == 192 && BK == 16 && WARPS == 4) {
            if (s.path == UnpackPath::LUT)
                kernels::int8_v13::mm_int8_lut_v13_rf_v6<128, 192, 16, 16>
                    <<<grid, block, smem_bytes>>>(
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),
                        d_act, d_dst, M, N, K);
        }
#define RERUN_V10S(b_m, b_n, b_k, w, fm, fn, ks)                                \
        if (s.version==20 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && s.split_k==(ks)) { \
            if (s.path == UnpackPath::LUT) {                                    \
                if ((ks) > 1) cudaMemsetAsync(d_dst, 0, (size_t) M * N * sizeof(float)); \
                kernels::int8_v10s::mm_int8_lut_v10s<(b_m),(b_n),(b_k),(w),(fm),(fn),(ks)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
            }                                                                    \
        }
        RERUN_V10S(128, 128, 32, 4, 8, 2, 1)
        RERUN_V10S(128, 128, 32, 4, 8, 2, 2)
        RERUN_V10S(128, 128, 32, 4, 8, 2, 4)
        RERUN_V10S(128, 128, 32, 4, 8, 2, 8)
        RERUN_V10S(64,  128, 32, 4, 4, 2, 2)
        RERUN_V10S(64,  128, 32, 4, 4, 2, 4)
        RERUN_V10S(64,  128, 32, 4, 4, 2, 8)
        RERUN_V10S(64,  64,  32, 4, 4, 1, 2)
        RERUN_V10S(64,  64,  32, 4, 4, 1, 4)
        RERUN_V10S(64,  64,  32, 4, 4, 1, 8)
        RERUN_V10S(64,  256, 32, 8, 4, 2, 2)
        RERUN_V10S(64,  256, 32, 8, 4, 2, 4)
#undef RERUN_V10S
#define RERUN_V12S(b_m, b_n, b_k, w, fm, fn, ks)                                 \
        if (s.version==60 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && s.split_k==(ks) && ((fn) % 2 == 0)) { \
            constexpr int am = (fm) * 2;                                        \
            constexpr int an = (fn) / 2;                                        \
            if (s.path == UnpackPath::LUT) {                                    \
                if ((ks) > 1) cudaMemsetAsync(d_dst, 0, (size_t) M * N * sizeof(float)); \
                kernels::int8_v12::mm_int8_lut_v12s<(b_m),(b_n),(b_k),(w),am,an,(ks)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
            }                                                                    \
        }
        RERUN_V12S(64,  128, 16, 4, 4, 2, 1)
        RERUN_V12S(64,  128, 16, 4, 4, 2, 2)
        RERUN_V12S(64,  128, 16, 4, 4, 2, 4)
        RERUN_V12S(64,  128, 16, 4, 4, 2, 8)
        RERUN_V12S(64,  128, 16, 4, 4, 2, 16)
        RERUN_V12S(64,  128, 32, 4, 4, 2, 2)
        RERUN_V12S(64,  128, 32, 4, 4, 2, 4)
        RERUN_V12S(64,  128, 32, 4, 4, 2, 8)
        RERUN_V12S(64,  128, 16, 4, 4, 2, 3)
        RERUN_V12S(64,  128, 16, 4, 4, 2, 5)
        RERUN_V12S(128, 128, 16, 4, 8, 2, 2)
        RERUN_V12S(128, 128, 16, 4, 8, 2, 4)
#undef RERUN_V12S

#ifdef TCGRID_HAS_CUTLASS
#define RERUN_CUTLASS_P1(cta_m, cta_n, cta_k, w_m, w_n, w_k)                  \
        if (s.version == 40 && BM == (cta_m) && BN == (cta_n) && BK == (cta_k) && cs.W_fp16 != nullptr) { \
            kernels::int8_cutlass::Gemm70<                                    \
                (cta_m),(cta_n),(cta_k),(w_m),(w_n),(w_k)                     \
            >::run(cs.A_fp16, cs.W_fp16, d_dst, M, N, K);                     \
        }
        RERUN_CUTLASS_P1(128, 128, 32, 64, 64, 32)
        RERUN_CUTLASS_P1(64,  128, 32, 32, 64, 32)
        RERUN_CUTLASS_P1(128, 64,  32, 64, 32, 32)
#undef RERUN_CUTLASS_P1
#endif
#define RERUN_V7(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==7 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v7::mm_int8_lut_v7<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V7(64,  128, 32, 4, 4, 2)
        RERUN_V7(64,  64,  32, 4, 4, 1)
        RERUN_V7(32,  128, 32, 4, 2, 2)
        RERUN_V7(32,  64,  32, 4, 2, 1)
#undef RERUN_V7
#define RERUN_V6(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==6 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v6::mm_int8_lut_v6<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V6(128, 128, 64, 4, 8, 2)
        RERUN_V6(128, 64,  64, 4, 8, 1)
        RERUN_V6(64,  128, 64, 4, 4, 2)
        RERUN_V6(64,  64,  64, 4, 4, 1)
#undef RERUN_V6
#define RERUN_V5(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==5 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v5::mm_int8_lut_v5<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K, num_tile_x, num_tile_y);         \
        }
        RERUN_V5(128, 128, 32, 4, 8, 2)
        RERUN_V5(128, 64,  32, 4, 8, 1)
        RERUN_V5(64,  128, 32, 4, 4, 2)
        RERUN_V5(64,  64,  32, 4, 4, 1)
#undef RERUN_V5
#define RERUN_V4(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==4 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v4::mm_int8_lut_v4<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V4(128, 128, 32, 4, 8, 2)
        RERUN_V4(128, 64,  32, 4, 8, 1)
        RERUN_V4(64,  128, 32, 4, 4, 2)
        RERUN_V4(64,  64,  32, 4, 4, 1)
#undef RERUN_V4
#define RERUN_V3(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==3 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_v3::mm_int8_lut_v3<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V3(128, 128, 32, 4, 8, 2)
        RERUN_V3(128, 64,  32, 4, 8, 1)
        RERUN_V3(128, 256, 32, 4, 8, 4)
        RERUN_V3(128, 256, 32, 8, 8, 2)
        RERUN_V3(64,  128, 32, 4, 4, 2)
        RERUN_V3(64,  64,  32, 4, 4, 1)
        RERUN_V3(32,  128, 32, 4, 2, 2)
        RERUN_V3(32,  64,  32, 4, 2, 1)
        RERUN_V3(32,  256, 32, 8, 2, 2)
        RERUN_V3(128, 128, 32, 8, 8, 1)
        RERUN_V3(64,  128, 32, 8, 4, 1)
#undef RERUN_V3
#define RERUN_V2(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==2 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_opt::mm_int8_lut_v2<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V2(128, 128, 32, 4, 8, 2)
        RERUN_V2(128, 64,  32, 4, 8, 1)
        RERUN_V2(128, 256, 32, 4, 8, 4)
        RERUN_V2(128, 256, 32, 8, 8, 2)
        RERUN_V2(64,  128, 32, 4, 4, 2)
        RERUN_V2(64,  64,  32, 4, 4, 1)
#undef RERUN_V2
#define RERUN_V1(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==1 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_opt::mm_int8_lut_v1<(b_m),(b_n),(b_k),(w),(fm),(fn)> \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_V1(128, 128, 32, 4, 8, 2)
        RERUN_V1(128, 64,  32, 4, 8, 1)
        RERUN_V1(128, 128, 64, 4, 8, 2)
        RERUN_V1(128, 256, 32, 4, 8, 4)
        RERUN_V1(128, 256, 32, 8, 8, 2)
        RERUN_V1(64,  128, 32, 4, 4, 2)
        RERUN_V1(64,  64,  32, 4, 4, 1)
#undef RERUN_V1
#define RERUN_MF(b_m, b_n, b_k, w, fn)                                          \
        if (BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) {\
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int8_mf::mm_int8_lut_mf<(b_m),(b_n),(b_k),(w),(fn)>    \
                    <<<grid, block, smem_bytes>>>(                              \
                        w_qs_ptr(d_W), w_scales_ptr(d_W, N, K),                 \
                        d_act, d_dst, M, N, K);                                 \
        }
        RERUN_MF(16, 128, 32, 4, 2)
        RERUN_MF(16, 128, 64, 4, 2)
        RERUN_MF(16, 256, 32, 4, 4)
        RERUN_MF(16, 256, 64, 4, 4)
        RERUN_MF(16, 256, 32, 8, 2)
        RERUN_MF(16, 512, 32, 8, 4)
#undef RERUN_MF
        RERUN(16, 64, 32, 4)
        RERUN(16, 64, 64, 4)
        RERUN(16, 128, 32, 8)
        RERUN(16, 128, 64, 8)
        cudaEventRecord(evs[1]);
        cudaEventSynchronize(evs[1]);
        if (i >= N_WARMUP_ITERS) {
            float ms = 0; cudaEventElapsedTime(&ms, evs[0], evs[1]);
            times.push_back(ms);
        }
    }
    cudaEventDestroy(evs[0]); cudaEventDestroy(evs[1]);

    double sum = 0, mn = times[0], mx = times[0];
    for (float t : times) { sum += t; mn = std::min<double>(mn, t); mx = std::max<double>(mx, t); }
    R.ms_mean = sum / times.size();
    R.ms_min  = mn;
    R.ms_max  = mx;

    // FLOPs (matmul): 2*M*N*K
    R.tflops = (2.0 * M * N * K) / (R.ms_mean * 1e-3) / 1e12;

    // Bytes:
    //   weights: N*K bytes int8 + N*(K/QK_INT8)*2 bytes scales
    //   acts:    M*K * 4 bytes
    //   dst:     M*N * 4 bytes
    size_t bytes_w = (size_t)N * K + (size_t) N * (K / QK_INT8) * 2;
    size_t bytes_a = (size_t)M * K * 4;
    size_t bytes_c = (size_t)M * N * 4;
    R.gbytes_per_s = (double)(bytes_w + bytes_a + bytes_c) / (R.ms_mean * 1e-3) / 1e9;

    // Tolerance vs reference
    ToleranceStats t = evaluate_tolerance(d_dst, d_ref, (size_t) M * N);
    R.max_abs_err = t.max_abs;
    R.p99_abs_err = t.p99_abs;
    R.rel_err     = t.rel_err;

    R.ok = true;

#ifdef TCGRID_HAS_CUTLASS
    if (cs.W_fp16) cudaFree(cs.W_fp16);
    if (cs.A_fp16) cudaFree(cs.A_fp16);
#endif

    return R;
}

}  // namespace tc_grid

// SPRINT-020 P0.3 — Per-(M, shape) production dispatcher for V100 INT8 GEMM.
//
// This header encodes the empirical per-M (and where applicable, per-shape)
// champion kernel selections from SPRINT-019 close (REPORT-13 §1) plus
// any SPRINT-020 P3 updates. It is the SINGLE SOURCE OF TRUTH for runtime
// dispatch — both the tc-grid lab harness and the lmdeploy turbomind runtime
// integration (P4 ceiling-proof branch) consume this header so the two
// dispatch paths cannot diverge.
//
// The selection function `choose_kernel(M, N, K)` returns a `LaunchSpec`
// that the caller passes to its dispatcher. Callers in tc-grid use
// `launch_int8.cu`; runtime callers (LlamaLinear.cu, moe_ffn_layer.cc)
// will see this header through a wrapper in the lmdeploy turbomind tree.
//
// Per-M champion table (SPRINT-021 close, REPORT-16 + asymmetric catalog):
//
//   M ∈ [1,    32]   → v12s_64x128x32_w4_ks8        (v12 SplitK, KS=8; small M)
//   M = 64           → v12s_64x128x32_w4_ks8        (21.55 TF — SPRINT-019 close)
//   M ≥ 128          → v13_rf_v6_128x128x16_w4      (champion across M ∈ [128, 4096])
//
// v13_rf_v6 = register-file dequant + 2x2 warp partition + K-iter LDS fusion.
// Beats v12_ms3 by +27% at M=2048 N=K=7168 (50 TF goal hit), and across 23/24
// asymmetric DSv4 shapes (v13rf_v6-vs-v12ms3-SPRINT-021.csv). Best gain: +177%
// at (256, 7168, 18944) — small M, long K.
//
// BM=64 v6 tested at mid-M but regressed -41% to -52% at M=512/1024 vs BM=128.
// CHAMPION_MID therefore uses BM=128 across the full M range now.
//
// Threshold-adjacent M values verified at SPRINT-020 P0.3:
//   M ∈ {63, 65, 255, 257, 1023, 1025} → expected boundary selections.
//
// Shape-sensitivity (SPRINT-019 P6 finding):
//   Champion `128x128x16_w4` is within 14% across square shapes
//   NK ∈ {4096, 7168, 8192} at M=2048. No per-shape variant required.
//   Asymmetric shapes (7168×18944 etc.) measured at SPRINT-020 P3
//   may extend this table; revise when P3 evidence is in.

#pragma once

#include "tc_grid.h"
#include <cstdint>

namespace tc_grid::dispatch {

// Threshold breakpoints for M-direction routing.
// Inclusive-low, exclusive-high (i.e., a band is [M_LO, M_HI)).
constexpr int M_DECODE_HI   = 128;   // M < 128 → SplitK path
constexpr int M_MID_HI      = 512;   // M ∈ [128, 512) → small-BM ms3
// M ≥ 512 → champion BM=128 W=4 ms3

// Per-M champion descriptors. Fields mirror the kTiles[] schema used by
// tools/tc-grid/src/main.cu (BM, BN, BK, warps, frag_n, version, label,
// chunk_k=split_k for v12s).
struct ChampionSpec {
    int BM, BN, BK;
    int warps;
    int frag_n;
    int version;
    int chunk_k;     // = split_k for SplitK kernels (version=60); 1 otherwise
    const char * label;
};

// SPRINT-019-close champion table. Values reproduced from REPORT-13 §1.
//
// NOTE: these are "preferred" entries; the actual runtime resolution must
// still go through the kernel registry — if a champion isn't registered
// for the active build (e.g., v12s_ks8 requires LAUNCH_V12S(64,128,32,4,4,2,8)
// in launch_int8.cu, which it has), the caller is responsible for falling
// back to v11 or v10.
constexpr ChampionSpec CHAMPION_DECODE = {
    /*BM=*/64, /*BN=*/128, /*BK=*/32,
    /*warps=*/4, /*frag_n=*/2,
    /*version=*/60,  // v12s (SplitK)
    /*chunk_k=*/8,
    /*label=*/"v12s_64x128x32_w4_ks8"
};

constexpr ChampionSpec CHAMPION_MID = {
    /*BM=*/128, /*BN=*/128, /*BK=*/16,
    /*warps=*/4, /*frag_n=*/2,
    /*version=*/88,  // v13_rf_v6 BM=128 wins over BM=64 v6 at mid-M (SPRINT-021)
    /*chunk_k=*/1,
    /*label=*/"v13_rf_v6_128x128x16_w4"
};

constexpr ChampionSpec CHAMPION_LARGE = {
    /*BM=*/128, /*BN=*/128, /*BK=*/16,
    /*warps=*/4, /*frag_n=*/2,
    /*version=*/88,  // v13_rf_v6 (SPRINT-021 close)
    /*chunk_k=*/1,
    /*label=*/"v13_rf_v6_128x128x16_w4"
};

// Select the production champion for a given (M, N, K). N and K are
// reserved parameters for SPRINT-020 P3 asymmetric-shape extensions —
// the current rule depends on M only.
//
// Per SPRINT-019 P6: square shapes NK ∈ {4096, 7168, 8192} share the
// champion at M=2048 within 14%. If SPRINT-020 P3 measurements show
// asymmetric shapes (e.g., 7168×18944) needing a different champion,
// extend this function with (N, K) gating.
__host__ inline ChampionSpec choose_kernel(int M, int /*N*/, int /*K*/) {
    if (M < M_DECODE_HI) {
        // Decode-style small-M: SplitK on v12 base.
        return CHAMPION_DECODE;
    } else if (M < M_MID_HI) {
        // Mid-band: smaller BM ms3 amortizes K-loop overhead better at
        // intermediate batch sizes (M=256 sweet spot).
        return CHAMPION_MID;
    } else {
        // Large-M: champion BM=128 W=4 ms3.
        return CHAMPION_LARGE;
    }
}

// Threshold-adjacent verification helper: returns the boundary M values
// the dispatcher must handle correctly. Used in unit tests.
__host__ inline const int * threshold_M_values(int * count) {
    static constexpr int boundary[] = {
        M_DECODE_HI - 1, M_DECODE_HI, M_DECODE_HI + 1,    // 127, 128, 129
        M_MID_HI - 1,    M_MID_HI,    M_MID_HI + 1,       // 511, 512, 513
        63, 65,                                            // decode-band edges
        255, 257,                                          // sub-band edge
        1023, 1025                                         // sticky-champion edge
    };
    *count = sizeof(boundary) / sizeof(boundary[0]);
    return boundary;
}

}  // namespace tc_grid::dispatch

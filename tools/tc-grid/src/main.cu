// tc-grid main driver:
//   1. Generate FP32 weight + activation matrices with selectable distribution.
//   2. Quantize weights into each target format.
//   3. Dequant -> cuBLAS FP32 GEMM = reference.
//   4. For each (format, path, tile config), launch the kernel and tabulate
//      ms, TFLOPS, GB/s, max_abs, p99_abs, rel_err.
//   5. Emit results as CSV to stdout (machine readable for the report).
//
// CLI:
//   tc-grid [--m-list 1,4,8,16,32,64] [--nk 7168] [--dist uniform_small]
// All optional; defaults give a useful first sweep.

#include "tc_grid.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

using namespace tc_grid;

struct Args {
    std::vector<int> m_list = {1, 4, 8, 16, 32, 64};
    // Square-shape shorthand (preserved): --nk X,Y,Z sets N=K=X, N=K=Y, ...
    // Asymmetric: --n-list a,b --k-list c,d iterates pairwise (a,c) then (b,d).
    // If only one of --n-list / --k-list is provided, the other is auto-filled
    // from --nk (or its default).
    std::vector<int> nk_list = {4096, 7168};
    std::vector<int> n_list;  // empty → fall back to nk_list (square)
    std::vector<int> k_list;  // empty → fall back to nk_list (square)
    DataDist dist = DataDist::UNIFORM_SMALL;
    bool all_dists = false;
};

static DataDist parse_dist(const char * s) {
    if (!strcmp(s, "uniform_small"))  return DataDist::UNIFORM_SMALL;
    if (!strcmp(s, "uniform_wide"))   return DataDist::UNIFORM_WIDE;
    if (!strcmp(s, "lognormal"))      return DataDist::LOGNORMAL;
    if (!strcmp(s, "sparse_spikes"))  return DataDist::SPARSE_SPIKES;
    if (!strcmp(s, "adversarial"))    return DataDist::ADVERSARIAL;
    fprintf(stderr, "unknown dist '%s'\n", s);
    std::exit(2);
}

static std::vector<int> parse_int_list(const char * s) {
    std::vector<int> out;
    const char * p = s;
    while (*p) {
        char * end;
        int v = (int) strtol(p, &end, 10);
        if (end == p) break;
        out.push_back(v);
        p = end;
        if (*p == ',') ++p;
    }
    return out;
}

static Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--m-list") && i+1 < argc) a.m_list = parse_int_list(argv[++i]);
        else if (!strcmp(argv[i], "--nk")   && i+1 < argc) a.nk_list = parse_int_list(argv[++i]);
        else if (!strcmp(argv[i], "--n-list") && i+1 < argc) a.n_list = parse_int_list(argv[++i]);
        else if (!strcmp(argv[i], "--k-list") && i+1 < argc) a.k_list = parse_int_list(argv[++i]);
        else if (!strcmp(argv[i], "--dist") && i+1 < argc) a.dist = parse_dist(argv[++i]);
        else if (!strcmp(argv[i], "--all-dists")) a.all_dists = true;
        else fprintf(stderr, "[warn] ignored arg: %s\n", argv[i]);
    }
    // Resolve to a final NK pair list. SPRINT-020 P0.2: support N≠K.
    if (a.n_list.empty() && a.k_list.empty()) {
        // Default / --nk path: pair each square dim with itself.
        for (int nk : a.nk_list) { a.n_list.push_back(nk); a.k_list.push_back(nk); }
    } else if (a.n_list.empty()) {
        // Only --k-list given: square against the supplied dims.
        for (int k : a.k_list) a.n_list.push_back(k);
    } else if (a.k_list.empty()) {
        for (int n : a.n_list) a.k_list.push_back(n);
    }
    if (a.n_list.size() != a.k_list.size()) {
        fprintf(stderr, "[error] --n-list (%zu) and --k-list (%zu) length mismatch\n",
                a.n_list.size(), a.k_list.size());
        std::exit(2);
    }
    return a;
}

struct Tile { int BM, BN, BK, warps; int frag_n; int version; const char * label; int chunk_k = 1; };
static const Tile kTiles[] = {
    // Baseline single-A-frag (BM=16):
    { 16, 64,  32, 4, 1, 0, "16x64x32_w4"     },
    { 16, 128, 32, 8, 1, 0, "16x128x32_w8"    },
    { 16, 128, 32, 4, 2, 0, "16x128x32_w4f2"  },
    { 16, 256, 32, 4, 4, 0, "16x256x32_w4f4"  },
    // v1: BM>16 multi-A-frag, single-buffered.
    { 128, 64,  32, 4, 1, 1, "128x64x32_w4_v1"   },
    { 128, 128, 32, 4, 2, 1, "128x128x32_w4_v1" },
    { 128, 256, 32, 4, 4, 1, "128x256x32_w4_v1" },
    { 128, 256, 32, 8, 2, 1, "128x256x32_w8_v1" },
    { 64,  64,  32, 4, 1, 1, "64x64x32_w4_v1"    },
    { 64,  128, 32, 4, 2, 1, "64x128x32_w4_v1"   },
    // v2: + double-buffer + uint4 vectorized loads.
    { 128, 64,  32, 4, 1, 2, "128x64x32_w4_v2"   },
    { 128, 128, 32, 4, 2, 2, "128x128x32_w4_v2" },
    { 128, 256, 32, 4, 4, 2, "128x256x32_w4_v2" },
    { 128, 256, 32, 8, 2, 2, "128x256x32_w8_v2" },
    { 64,  64,  32, 4, 1, 2, "64x64x32_w4_v2"    },
    { 64,  128, 32, 4, 2, 2, "64x128x32_w4_v2"   },
    // v3: v2 + bank-conflict-free B (padded BK_PAD=BK+8) + PRMT (FP4/F8).
    // Dropped BN=256 v3 tiles (FRAG_M=8, FRAG_N=4 → 32 c_frags spills regs).
    { 128, 64,  32, 4, 1, 3, "128x64x32_w4_v3"   },
    { 128, 128, 32, 4, 2, 3, "128x128x32_w4_v3" },
    { 128, 256, 32, 8, 2, 3, "128x256x32_w8_v3" },
    { 64,  64,  32, 4, 1, 3, "64x64x32_w4_v3"    },
    { 64,  128, 32, 4, 2, 3, "64x128x32_w4_v3"   },
    // B1: FRAG_M=2 (BM=32) high-occupancy tiles
    { 32,  128, 32, 4, 2, 3, "32x128x32_w4_v3_b1" },
    { 32,  64,  32, 4, 1, 3, "32x64x32_w4_v3_b1"   },
    { 32,  256, 32, 8, 2, 3, "32x256x32_w8_v3_b1" },
    // B1 alt: 8 warps/CTA at v3 winning BM
    { 128, 128, 32, 8, 1, 3, "128x128x32_w8_v3_b1alt" },
    { 64,  128, 32, 8, 1, 3, "64x128x32_w8_v3_b1alt"  },
    // v4: v3 + Tier S (L2 prefetch + __launch_bounds__ + __ldg).
    { 128, 64,  32, 4, 1, 4, "128x64x32_w4_v4"   },
    { 128, 128, 32, 4, 2, 4, "128x128x32_w4_v4" },
    { 64,  64,  32, 4, 1, 4, "64x64x32_w4_v4"    },
    { 64,  128, 32, 4, 2, 4, "64x128x32_w4_v4"   },
    // v5: v3 + Tier A (persistent CTAs + L2 prefetch + __ldg + __launch_bounds__).
    { 128, 64,  32, 4, 1, 5, "128x64x32_w4_v5"   },
    { 128, 128, 32, 4, 2, 5, "128x128x32_w4_v5" },
    { 64,  64,  32, 4, 1, 5, "64x64x32_w4_v5"    },
    { 64,  128, 32, 4, 2, 5, "64x128x32_w4_v5"   },
    // v6: BK=64 single-buffer + L2 prefetch (A1 retry, INT8 LUT only).
    { 128, 128, 64, 4, 2, 6, "128x128x64_w4_v6" },
    { 128, 64,  64, 4, 1, 6, "128x64x64_w4_v6"   },
    { 64,  128, 64, 4, 2, 6, "64x128x64_w4_v6"   },
    { 64,  64,  64, 4, 1, 6, "64x64x64_w4_v6"    },
    // v7: B3 3-stage triple-buffer (INT8 only, SMEM-bound to BM<=64)
    { 64,  128, 32, 4, 2, 7, "64x128x32_w4_v7"   },
    { 64,  64,  32, 4, 1, 7, "64x64x32_w4_v7"    },
    { 32,  128, 32, 4, 2, 7, "32x128x32_w4_v7"   },
    { 32,  64,  32, 4, 1, 7, "32x64x32_w4_v7"    },
    // v8: v3 base + FP16 accumulator (half c_frag → c_frag regs 8→4 per frag).
    //     Directly attacks the 30% Short Scoreboard stall and 6% occupancy ceiling.
    //     Bit-correctness: tolerance contract must be re-validated for FP16 acc.
    { 128, 64,  32, 4, 1, 8, "128x64x32_w4_v8"   },
    { 128, 128, 32, 4, 2, 8, "128x128x32_w4_v8" },
    { 64,  64,  32, 4, 1, 8, "64x64x32_w4_v8"    },
    { 64,  128, 32, 4, 2, 8, "64x128x32_w4_v8"   },
    // B1-style high-occupancy variants too — FP16 acc + smaller BM
    { 32,  128, 32, 4, 2, 8, "32x128x32_w4_v8"   },
    { 32,  64,  32, 4, 1, 8, "32x64x32_w4_v8"    },
    // v9: v3 base + chunked FP16/FP32 mixed-precision accumulator. CHUNK_K via .chunk_k.
    { 128, 128, 32, 4, 2, 9, "128x128x32_w4_v9_ck2", 2 },
    { 128, 128, 32, 4, 2, 9, "128x128x32_w4_v9_ck4", 4 },
    { 128, 128, 32, 4, 2, 9, "128x128x32_w4_v9_ck8", 8 },
    { 64,  128, 32, 4, 2, 9, "64x128x32_w4_v9_ck4",  4 },
    { 64,  64,  32, 4, 1, 9, "64x64x32_w4_v9_ck4",   4 },
    // v10: v4 base + row-major B SMEM layout (attacks MIO/bank-conflict on B loads)
    { 128, 128, 32, 4, 2, 10, "128x128x32_w4_v10" },
    { 128, 64,  32, 4, 1, 10, "128x64x32_w4_v10"  },
    { 64,  128, 32, 4, 2, 10, "64x128x32_w4_v10"  },
    { 64,  64,  32, 4, 1, 10, "64x64x32_w4_v10"   },
    { 64,  256, 32, 8, 2, 10, "64x256x32_w8_v10"  },
    { 128, 256, 32, 8, 2, 10, "128x256x32_w8_v10" },
    // v11: v10 base + manual Lds + SM70_MMA_884 atoms (m8n32k8 per fma).
    // Only N_PER_WARP >= 32 shapes supported initially (atom_n = 32).
    // Bit-correctness baseline; perf parity with v10 expected for this step.
    { 128, 128, 32, 4, 2, 11, "128x128x32_w4_v11" },
    { 64,  128, 32, 4, 2, 11, "64x128x32_w4_v11"  },
    { 128, 256, 32, 8, 2, 11, "128x256x32_w8_v11" },
    { 64,  256, 32, 8, 2, 11, "64x256x32_w8_v11"  },
    // v11 BK=16 variants (turbomind ships CTA_K=16 on sm_70; tighter pipeline).
    { 128, 128, 16, 4, 2, 11, "128x128x16_w4_v11" },
    { 64,  128, 16, 4, 2, 11, "64x128x16_w4_v11"  },
    { 128, 256, 16, 8, 2, 11, "128x256x16_w8_v11" },
    { 64,  256, 16, 8, 2, 11, "64x256x16_w8_v11"  },
    // Experimental: larger N_PER_WARP=64 (ATOMS_N=2)
    { 64,  256, 16, 4, 4, 11, "64x256x16_w4f4_v11"  },
    { 128, 256, 16, 4, 4, 11, "128x256x16_w4f4_v11" },
    // Grid search: smaller BM (high occupancy)
    { 32,  128, 16, 4, 2, 11, "32x128x16_w4_v11"  },
    { 32,  256, 16, 8, 2, 11, "32x256x16_w8_v11"  },
    { 32,  128, 32, 4, 2, 11, "32x128x32_w4_v11"  },
    // Grid search: larger BM
    { 192, 128, 16, 4, 2, 11, "192x128x16_w4_v11" },
    { 256, 128, 16, 4, 2, 11, "256x128x16_w4_v11" },
    // Grid search: BK=64
    { 128, 128, 64, 4, 2, 11, "128x128x64_w4_v11" },
    { 64,  128, 64, 4, 2, 11, "64x128x64_w4_v11"  },
    { 64,  256, 64, 8, 2, 11, "64x256x64_w8_v11"  },
    // Grid search: W=2 (fewer warps, ATOMS_N=2)
    { 64,  128, 16, 2, 4, 11, "64x128x16_w2f4_v11"  },
    { 64,  128, 32, 2, 4, 11, "64x128x32_w2f4_v11"  },
    { 32,  128, 16, 2, 4, 11, "32x128x16_w2f4_v11"  },
    // v12: v11 base + FP16 accumulator + SMEM round-trip epilogue (SPRINT-019 P1).
    // Champion-class set; expand in P1.4 grid sweep.
    { 128, 128, 16, 4, 2, 50, "128x128x16_w4_v12" },
    { 64,  128, 16, 4, 2, 50, "64x128x16_w4_v12"  },
    { 128, 256, 16, 8, 2, 50, "128x256x16_w8_v12" },
    { 64,  256, 16, 8, 2, 50, "64x256x16_w8_v12"  },
    { 128, 128, 32, 4, 2, 50, "128x128x32_w4_v12" },
    { 64,  128, 32, 4, 2, 50, "64x128x32_w4_v12"  },
    { 192, 128, 16, 4, 2, 50, "192x128x16_w4_v12" },
    { 256, 128, 16, 4, 2, 50, "256x128x16_w4_v12" },
    { 32,  128, 16, 4, 2, 50, "32x128x16_w4_v12"  },
    { 32,  256, 16, 8, 2, 50, "32x256x16_w8_v12"  },
    { 64,  128, 16, 2, 4, 50, "64x128x16_w2f4_v12" },
    { 64,  256, 16, 4, 4, 50, "64x256x16_w4f4_v12" },
    // v12_ms3: v12 + 3-stage decoupled pipeline (SPRINT-019 P2). Same tile
    // set as v12 (champion class). version=51.
    { 128, 128, 16, 4, 2, 51, "128x128x16_w4_v12_ms3" },
    { 64,  128, 16, 4, 2, 51, "64x128x16_w4_v12_ms3"  },
    { 128, 256, 16, 8, 2, 51, "128x256x16_w8_v12_ms3" },
    { 64,  256, 16, 8, 2, 51, "64x256x16_w8_v12_ms3"  },
    { 128, 128, 32, 4, 2, 51, "128x128x32_w4_v12_ms3" },
    { 64,  128, 32, 4, 2, 51, "64x128x32_w4_v12_ms3"  },
    { 64,  256, 16, 4, 4, 51, "64x256x16_w4f4_v12_ms3" },
    { 64,  128, 16, 2, 4, 51, "64x128x16_w2f4_v12_ms3" },
    // P4 §6.4 NEGATIVE RESULT (SPRINT-019, commit 0cbcecae1): larger BM
    // v12_ms3 builds clean (no spill needed) but LOSES at every M vs BM=128
    // (-11% at M=2048, -17% at M=256). Kernel is mio_throttle-bound (26%
    // SMEM bw); larger BM amplifies SMEM traffic. Kept as experimental
    // tiles for documentation; NEVER selected by dispatch.h at any (M, shape).
    { 192, 128, 16, 4, 2, 51, "192x128x16_w4_v12_ms3_NEG" },
    { 256, 128, 16, 4, 2, 51, "256x128x16_w4_v12_ms3_NEG" },

    // v13_rf: register-file dequant prototype (SPRINT-021 P1). version=80.
    // Same shape as v12_ms3 champion; sB stores raw INT8 (1B/wt). Mainloop
    // LDS 8 INT8 per lane → PRMT-dequant in registers → mma. Apples-to-apples
    // vs v12_ms3 128x128x16_w4 to isolate the dequant-location effect.
    { 128, 128, 16, 4, 2, 80, "128x128x16_w4_v13_rf" },
    {  64, 128, 16, 4, 2, 80,  "64x128x16_w4_v13_rf" },
    // v13_rf_v2: K-iter fusion (single uint4 LDS + 4× PRMT + 4 mma). version=81.
    { 128, 128, 16, 4, 2, 81, "128x128x16_w4_v13_rf_v2" },
    {  64, 128, 16, 4, 2, 81,  "64x128x16_w4_v13_rf_v2" },
    // v13_rf_v3: software pipeline (LDS ki+1 issued before mma ki). version=82.
    { 128, 128, 16, 4, 2, 82, "128x128x16_w4_v13_rf_v3" },
    {  64, 128, 16, 4, 2, 82,  "64x128x16_w4_v13_rf_v3" },
    // v13_rf_v4: launch_bounds(*, 1) → 256 reg/thread cap (1 CTA/SM). v=83.
    { 128, 128, 16, 4, 2, 83, "128x128x16_w4_v13_rf_v4" },
    {  64, 128, 16, 4, 2, 83,  "64x128x16_w4_v13_rf_v4" },
    // v13_rf_v5: 2×2 warp partition (turbomind Blocked<2,2>). version=84.
    { 128, 128, 16, 4, 2, 84, "128x128x16_w4_v13_rf_v5" },
    // v13_rf_v5 BK_PAD sweep — bank-conflict tuning on INT8 LDS.
    { 128, 128, 16, 4, 2, 85, "128x128x16_w4_v13_rf_v5_pad32" },
    { 128, 128, 16, 4, 2, 86, "128x128x16_w4_v13_rf_v5_pad48" },
    // v13_rf_v5 BN=256 (ATOMS_N=4 per warp — deeper HMMA chain). version=87.
    { 128, 256, 16, 4, 4, 87, "128x256x16_w4_v13_rf_v5_bn256" },
    // v13_rf_v6: v5 + K-iter fusion (uint4 LDS covering 16 K). version=88.
    { 128, 128, 16, 4, 2, 88, "128x128x16_w4_v13_rf_v6" },
    {  64, 128, 16, 4, 2, 88,  "64x128x16_w4_v13_rf_v6" },
    // v13_rf_v7: v5 + mma reorder (max independence between same-c-frag hits). v=89.
    { 128, 128, 16, 4, 2, 89, "128x128x16_w4_v13_rf_v7" },
    // v13_rf_v8: v6 + LB(*,2) — tests if higher occupancy helps. v=90.
    { 128, 128, 16, 4, 2, 90, "128x128x16_w4_v13_rf_v8_occ2" },
    // v6 BN=192 (ATOMS_N=3). Between BN=128 (works) and BN=256 (failed). v=91.
    { 128, 192, 16, 4, 3, 91, "128x192x16_w4_v13_rf_v6_bn192" },

    // v12s: v12 base + SplitK for decode-style small-M (SPRINT-019 P3). version=60.
    { 64,  128, 16, 4, 2, 60, "64x128x16_w4_v12s_ks1",  1 },
    { 64,  128, 16, 4, 2, 60, "64x128x16_w4_v12s_ks2",  2 },
    { 64,  128, 16, 4, 2, 60, "64x128x16_w4_v12s_ks4",  4 },
    { 64,  128, 16, 4, 2, 60, "64x128x16_w4_v12s_ks8",  8 },
    { 64,  128, 16, 4, 2, 60, "64x128x16_w4_v12s_ks16", 16 },
    { 64,  128, 32, 4, 2, 60, "64x128x32_w4_v12s_ks2",  2 },
    { 64,  128, 32, 4, 2, 60, "64x128x32_w4_v12s_ks4",  4 },
    { 64,  128, 32, 4, 2, 60, "64x128x32_w4_v12s_ks8",  8 },
    // Adversarial non-power-of-2 KSPLIT — sprint §1.2 #2 racecheck/initcheck coverage
    { 64,  128, 16, 4, 2, 60, "64x128x16_w4_v12s_ks3",  3 },
    { 64,  128, 16, 4, 2, 60, "64x128x16_w4_v12s_ks5",  5 },
    // Mid-band BM=128
    { 128, 128, 16, 4, 2, 60, "128x128x16_w4_v12s_ks2", 2 },
    { 128, 128, 16, 4, 2, 60, "128x128x16_w4_v12s_ks4", 4 },
    // v10 BK=64 explored — regressed badly due to L1 cache shrinkage from 96KB SMEM opt-in.
    // Documented as negative result in REPORT-11 appendix.
    //
    // v10s: v10 + SplitK (k-direction CTAs with atomicAdd into C). chunk_k = KS.
    // KS=1 is a sanity-check sentinel (bit-identical to v10 since the kernel
    // uses a non-atomic store at KS=1). KS ∈ {2, 4, 8} are the real tests.
    // Per SPRINT-017 Wave 1 A0 + memory feedback "execute every step of the plan".
    { 128, 128, 32, 4, 2, 20, "128x128x32_w4_v10s_ks1", 1 },
    { 128, 128, 32, 4, 2, 20, "128x128x32_w4_v10s_ks2", 2 },
    { 128, 128, 32, 4, 2, 20, "128x128x32_w4_v10s_ks4", 4 },
    { 128, 128, 32, 4, 2, 20, "128x128x32_w4_v10s_ks8", 8 },
    { 64,  128, 32, 4, 2, 20, "64x128x32_w4_v10s_ks2",  2 },
    { 64,  128, 32, 4, 2, 20, "64x128x32_w4_v10s_ks4",  4 },
    { 64,  128, 32, 4, 2, 20, "64x128x32_w4_v10s_ks8",  8 },
    { 64,  64,  32, 4, 1, 20, "64x64x32_w4_v10s_ks2",   2 },
    { 64,  64,  32, 4, 1, 20, "64x64x32_w4_v10s_ks4",   4 },
    { 64,  64,  32, 4, 1, 20, "64x64x32_w4_v10s_ks8",   8 },
    { 64,  256, 32, 8, 2, 20, "64x256x32_w8_v10s_ks2",  2 },
    { 64,  256, 32, 8, 2, 20, "64x256x32_w8_v10s_ks4",  4 },
    // v3s: INT4 SplitK (version=30). LAUNCH_V3S in launch_int4.cu only.
    // Other format launchers fall through to SKIP for version=30 — safe.
    { 128, 128, 32, 4, 2, 30, "128x128x32_w4_v3s_ks1",  1 },
    { 128, 128, 32, 4, 2, 30, "128x128x32_w4_v3s_ks2",  2 },
    { 128, 128, 32, 4, 2, 30, "128x128x32_w4_v3s_ks4",  4 },
    { 128, 128, 32, 4, 2, 30, "128x128x32_w4_v3s_ks8",  8 },
    { 128, 64,  32, 4, 1, 30, "128x64x32_w4_v3s_ks2",   2 },
    { 128, 64,  32, 4, 1, 30, "128x64x32_w4_v3s_ks4",   4 },
    { 64,  128, 32, 4, 2, 30, "64x128x32_w4_v3s_ks2",   2 },
    { 64,  128, 32, 4, 2, 30, "64x128x32_w4_v3s_ks4",   4 },
    { 64,  128, 32, 4, 2, 30, "64x128x32_w4_v3s_ks8",   8 },
    { 64,  64,  32, 4, 1, 30, "64x64x32_w4_v3s_ks2",    2 },
    { 64,  64,  32, 4, 1, 30, "64x64x32_w4_v3s_ks4",    4 },
    { 64,  64,  32, 4, 1, 30, "64x64x32_w4_v3s_ks8",    8 },
    // SPRINT-018 P1: CUTLASS V100 INT8 GEMM (version=40). Pre-dequant to FP16
    // then call cutlass::gemm::device::Gemm. Only INT8 launcher dispatches;
    // other formats fall through. (frag_n is unused for v40 — set to 2 for
    // launcher BN/WARPS==16 check to pass at WARPS=4 BN=128.)
    { 128, 128, 32, 4, 2, 40, "128x128x32_cutlass_p1" },
    { 64,  128, 32, 4, 2, 40, "64x128x32_cutlass_p1"  },
    { 128, 64,  32, 4, 1, 40, "128x64x32_cutlass_p1"  },
};

struct FormatInfo {
    Format f;
    size_t (*W_bytes)(int N, int K);
};

static size_t int8_bytes(int N, int K)  { return (size_t) N * K + (size_t) N * (K / QK_INT8) * 2; }
static size_t int4_bytes(int N, int K)  { return (size_t) N * (K / 2) + (size_t) N * (K / QK_INT4) * 2; }
static size_t mxfp4_bytes(int N, int K) { return (size_t) N * (K / QK_MXFP4) * (1 + QK_MXFP4 / 2); }
static size_t f8_bytes(int N, int K)    { return (size_t) N * (K / QK_F8) * (1 + QK_F8); }

static const FormatInfo kFormats[] = {
    { Format::INT8,         int8_bytes  },
    { Format::INT4,         int4_bytes  },
    { Format::MXFP4,        mxfp4_bytes },
    { Format::F8_E4M3_B128, f8_bytes    },
};

LaunchResult dispatch_launch(Format fmt, const LaunchSpec & s, const void * d_W,
                             const float * d_act, float * d_dst, const float * d_ref) {
    switch (fmt) {
        case Format::INT8:         return launch_int8(s, d_W, d_act, d_dst, d_ref);
        case Format::INT4:         return launch_int4(s, d_W, d_act, d_dst, d_ref);
        case Format::MXFP4:        return launch_fp4 (s, d_W, d_act, d_dst, d_ref);
        case Format::F8_E4M3_B128: return launch_fp8 (s, d_W, d_act, d_dst, d_ref);
    }
    LaunchResult R; R.ok = false; R.note = "unknown fmt"; return R;
}

static void sweep_one(Format fmt, int M, int N, int K, DataDist dist) {
    // Allocate workspace.
    float * d_W_f32 = nullptr;
    float * d_A     = nullptr;
    float * d_dst   = nullptr;
    float * d_ref   = nullptr;
    float * d_W_dq  = nullptr;
    void  * d_W     = nullptr;

    TCG_CHECK(cudaMalloc(&d_W_f32, (size_t) N * K * sizeof(float)));
    TCG_CHECK(cudaMalloc(&d_A,     (size_t) M * K * sizeof(float)));
    TCG_CHECK(cudaMalloc(&d_dst,   (size_t) M * N * sizeof(float)));
    TCG_CHECK(cudaMalloc(&d_ref,   (size_t) M * N * sizeof(float)));
    TCG_CHECK(cudaMalloc(&d_W_dq,  (size_t) N * K * sizeof(float)));
    TCG_CHECK(cudaMalloc(&d_W, kFormats[(int) fmt].W_bytes(N, K)));

    gen_matrix_f32(d_W_f32, N, K, dist, 0xC0FFEE);
    gen_matrix_f32(d_A,     M, K, dist, 0xBADBEEF);
    quantize(fmt, d_W_f32, d_W, N, K);
    dequant_reference(fmt, d_W, d_W_dq, N, K);
    reference_gemm_f32(d_W_dq, d_A, d_ref, M, N, K);

    // Try each tile config x each unpack path.
    for (const Tile & t : kTiles) {
        for (UnpackPath path : { UnpackPath::LUT, UnpackPath::BITSHIFT }) {
            LaunchSpec s;
            s.format = fmt; s.path = path; s.M = M; s.N = N; s.K = K;
            s.BM = t.BM; s.BN = t.BN; s.BK = t.BK; s.warps = t.warps; s.split_k = t.chunk_k; s.frag_n = t.frag_n; s.version = t.version;
            LaunchResult r = dispatch_launch(fmt, s, d_W, d_A, d_dst, d_ref);
            if (!r.ok) {
                printf("%s,%s,%s,%d,%d,%d,%s,SKIP,%s\n",
                       format_name(fmt), unpack_name(path), dist_name(dist),
                       M, N, K, t.label, r.note.c_str());
                continue;
            }
            printf("%s,%s,%s,%d,%d,%d,%s,OK,ms=%.3f,tflops=%.2f,gbps=%.1f,maxabs=%.3e,p99=%.3e,rel=%.3e\n",
                   format_name(fmt), unpack_name(path), dist_name(dist),
                   M, N, K, t.label,
                   r.ms_mean, r.tflops, r.gbytes_per_s,
                   r.max_abs_err, r.p99_abs_err, r.rel_err);
            fflush(stdout);
        }
    }

    cudaFree(d_W_f32); cudaFree(d_A); cudaFree(d_dst);
    cudaFree(d_ref); cudaFree(d_W_dq); cudaFree(d_W);
}

int main(int argc, char ** argv) {
    Args a = parse(argc, argv);

    int dev = 0; cudaDeviceProp p;
    TCG_CHECK(cudaGetDeviceProperties(&p, dev));
    fprintf(stderr, "device: %s, cc=%d.%d, smem=%zuKiB/cta\n",
            p.name, p.major, p.minor, (size_t)(p.sharedMemPerBlock >> 10));

    // CSV header
    printf("format,path,dist,M,N,K,tile,status,detail\n");

    std::vector<DataDist> dists;
    if (a.all_dists) {
        dists = { DataDist::UNIFORM_SMALL, DataDist::UNIFORM_WIDE,
                  DataDist::LOGNORMAL,    DataDist::SPARSE_SPIKES,
                  DataDist::ADVERSARIAL };
    } else {
        dists = { a.dist };
    }

    // cuBLAS FP16 GEMM ceiling per shape (V100 TC peak target).
    fprintf(stderr, "\n=== cuBLAS FP16 GEMM ceiling (V100 TC peak target 125 TFLOPS) ===\n");
    for (size_t i = 0; i < a.n_list.size(); ++i) {
        const int N = a.n_list[i], K = a.k_list[i];
        for (int M : a.m_list) {
            CublasFp16Result r = cublas_fp16_gemm_bench(M, N, K);
            if (r.ok) {
                printf("CUBLAS_FP16,REF,U(-1,1),%d,%d,%d,cublas,OK,ms=%.3f,tflops=%.2f,gbps=%.1f,maxabs=0.0e+0,p99=0.0e+0,rel=0.0e+0\n",
                       M, N, K, r.ms_mean, r.tflops, r.gbytes_per_s);
                fflush(stdout);
            }
        }
    }

    for (DataDist d : dists) {
        for (const FormatInfo & fi : kFormats) {
            for (size_t i = 0; i < a.n_list.size(); ++i) {
                const int N = a.n_list[i], K = a.k_list[i];
                for (int M : a.m_list) {
                    sweep_one(fi.f, M, N, K, d);
                }
            }
        }
    }
    return 0;
}

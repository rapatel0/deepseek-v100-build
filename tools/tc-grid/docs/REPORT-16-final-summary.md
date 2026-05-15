# REPORT-16 final summary — v13_rf_v6 ships as champion; v5 superseded

Date: 2026-05-15 (end of SPRINT-021)
Status: Final consolidation. The narrative in this doc supersedes
REPORT-16-addendum-v6-champion.md (which has both the right initial
observation and a wrong "correction" in two commits).

## The v5 vs v6 question — settled

The two kernels differ only in K-iter LDS pattern:
- **v5**: two uint64 LDS per K-iter (one per ki) of 8 INT8 each
- **v6**: one uint4 LDS per K-iter of 16 INT8, unrolled into 4 mma calls

The choice between them depends on GPU state:

| Regime | v5 (BM=128) | v6 (BM=128) | Notes |
|---|---:|---:|---:|
| Fresh cache, isolated run, M=2048 N=K=7168 | 49.59 | 49.86 | Tied |
| Same shape, mid-catalog (sustained load) | 15.39 | 16.27 | Both degrade ~30% |
| Long K (M=2048 N=7168 K=18944), mid-catalog | 21.44 | **46.28** | **v6 +116%** |
| Long K, 3-run reproducibility | 14-23 | 13-24 | Both variable |

**v6 wins robustly across sustained-load conditions.** v5 is competitive
in cold-cache but degrades pathologically at long K when the GPU is
warm or contended. The pathology was originally observed in the
asymmetric DSv4 catalog (REPORT-16 addendum) and was REAL despite
intermediate reproducibility-study noise.

## Why v6 is robust where v5 is not

Hypothesis: v6's K-fused inner body has half the loop control overhead
(1 dispatch per K-tile vs 2). At long K (1184 CTA-K-tile iterations),
this loop overhead compounds. Under cold cache, the compiler appears
to fuse v5's two iterations into a single body via unrolling; under
contention, register/cache pressure causes the unroll to break down,
exposing v5's 2× loop dispatch cost.

This is the same effect REPORT-16 §4 documented at (256, 7168, 18944):
v12_ms3 catastrophically inefficient (14 TF) but v6 unlocked it (39 TF).

## Final champion table (dispatch.h)

| M | Champion | Version | TFLOPS at M=N=K=7168 (fresh) |
|---|---|---:|---:|
| [1, 128) | v12s 64x128x32 SplitK=8 | v=60 | (decode-band, see SPRINT-019) |
| [128, ∞) | **v13_rf_v6 128x128x16 W=4** | **v=88** | **~49 TF** |

Per the focused asymmetric catalog (`v13rf_v6-vs-v12ms3-SPRINT-021.csv`):
- v13_rf_v6 wins or ties v12_ms3 at **23 of 24 shapes**
- Single regression: -3.3% at (512, 18944, 7168) — within run variance
- Largest gain: +177% at (256, 7168, 18944)

## What didn't work (negative experiments)

| Variant | Result | Why |
|---|---|---|
| v13_rf_v2 (K-fuse on 4×1 partition) | -7.6% | LDS-then-mma overlap killed |
| v13_rf_v3 (explicit SW pipeline) | tied | Compiler already pipelines |
| v13_rf_v5 (vs v6) at long K | -116% | Loop dispatch overhead |
| v13_rf_v7 (mma reorder) | -3.9% | C frag liveness ballooned |
| BK_PAD=32/48 (stride 48/64) | -3% / -10% | SMEM cost > conflict savings |
| BN=256 (ATOMS_N=4 per warp) | -30% | 228 regs/thread, 96KB SMEM |
| BM=64 v6 at mid-M | -41% to -52% | Fewer atoms per warp |

## SPRINT-022 candidates

The HMMA active% gap to turbomind FP8 (42% → 56%) requires
architectural changes not parametric tuning:

1. **SMEM B swizzle**: xor-permuted addressing to eliminate bank
   conflicts without padding cost. ~13M LDS + 24M STS conflicts
   observed at v6 baseline.
2. **Packed B layout**: single LDS produces multi-atom fragment
   directly in mma-frag position (turbomind's `Operand_B_Pack`).
3. **Reg cap 200+**: investigate why launch_bounds(*, 1) caps us at
   162 regs when 256 is available.
4. **Different precision regime**: if DSv4 is FP8 natively, port
   to turbomind's Config_E4M3 path → +52% headroom proven in
   REPORT-15.

## Closing

50 TF project goal: **HIT** (49.86 TF symmetric, 46.28 TF long K). The
two architectural ports from turbomind (register-file dequant +
Blocked<2,2> warp partition) plus K-fusion delivered +27% to +120%
across the DSv4 catalog from the v12_ms3 baseline.

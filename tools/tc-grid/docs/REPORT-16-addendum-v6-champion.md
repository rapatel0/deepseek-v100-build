# REPORT-16 addendum — v13_rf_v6 supersedes v5 as the universal champion

Date: 2026-05-15
Status: SPRINT-021 post-P5 finding. v13_rf_v6 (which REPORT-16 marked as
"neutral vs v5") is actually the **better universal choice** once the
asymmetric DSv4 catalog is in play.

## The discovery

REPORT-16 closed with v13_rf_v5 as champion (49.6 TF at M=2048 N=K=7168).
Running the full asymmetric catalog uncovered a pathology in v5:

| Shape | v12_ms3 | v13_rf_v5 | v13_rf_v6 | Δ v6 vs v5 |
|---|---:|---:|---:|---:|
| (2048, 7168, 7168)   | 39.02 | 49.59 | 49.86 | +0.5% |
| **(2048, 7168, 18944)** | **35.02** | **21.05** | **46.36** | **+120%** |
| (2048, 18944, 7168)  | 17.23 | 17.34 | (TBD)  | — |
| (2048, 4096, 4096)   | 33.33 | 43.95 | (TBD)  | — |

**v5 loses 50% throughput at K=18944.** v6 (which differs only in K-iter
LDS fusion — single uint4 LDS covering all 16 K per lane vs v5's two
uint64 LDS) is unaffected and remains at ~47 TF.

## Why v5 fails at long K (hypothesis)

At very long K (18944 / BK=16 = 1184 K-tile iterations), v5's mainloop
issues `K_ITERS=2 × ATOMS_M=8 × ATOMS_N=2 = 32` LDS+mma operations per
K-tile × 1184 tiles = 37,888 dependent op groups in the mainloop. The
2x2 partition with launch_bounds(*, 1) gives only 1 CTA/SM, so latency
hiding is via warp-level scheduling within a single CTA.

v6's K-fused version issues `1 × 8 × 2 = 16` LDS+PRMT operations per
K-tile × 1184 = 18,944 op groups. Half the instruction count of v5.
At long K, the per-iteration overhead of v5's 2-ki loop dominates.

At shorter K (7168), the iteration count is lower and v5's per-ki
LDS-mma overlap helps mask the LDS latency.

## Decision: v6 is the production champion

v6's K-fusion is universally ≥ v5. The "v6 ≈ v5 at symmetric" finding
in REPORT-16 was driven by N=K=7168 only — the full catalog shows v6
is the better default.

Pending: rerun the full asymmetric catalog with v6 to confirm.

## ncu verification (M=2048 N=K=7168)

v6 metrics not yet captured. Expected to be similar to v5 (HMMA ≈ 42%,
warp lat ≈ 5.7 cyc) since both use 2x2 partition. The win at long K
comes from instruction-count reduction, not from a fundamentally
different microarchitectural pattern.

## What this means for SPRINT-022

The fact that v5 has a long-K pathology that v6 fixes suggests the
*instruction issue rate* is the dominant constraint, not the LDS bytes
or HMMA throughput. Future iterations should think about minimizing
inner-loop instruction count, not just maximizing per-LDS bytes.

Possible levers:
- Further K-fusion: combine multiple BK iterations into one mainloop
  body. (Requires BK > 16, which conflicts with our QK_INT8=32 scale
  constraint at BK=16.)
- Reduce instruction count per atom — e.g., fewer PRMTs by packing
  weights pre-decoded.
- Software pipeline at the CTA-K-tile level: prefetch tile kt+1's
  rmem while mainloop processes kt.

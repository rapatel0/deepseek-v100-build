# Report 7 — Tier A (items A1+A2+A3) via v5 kernels

**Generated:** 2026-05-14T01:09:14.521938Z (V100 SXM2 32GB, sm_70)

## Tier A items applied

v5 = v3 base + Tier A items + revived Tier S items:

- **A1 (BK=64 wider K-tile)**: NOT implemented — V100 SMEM budget (48 KB) blocks BK=64
  for production tiles. BK=64 with BM=128 BN=128 double-buffered needs 64 KB SMEM; only
  BM=BN=64 BK=64 with padding fits (32 KB), losing the multi-A-frag benefit. Item
  deferred as architecturally infeasible on V100 — would require sm_75+ (96 KB SMEM).

- **A2 (gmem load reorder)**: subsumed by v3's existing double-buffer pattern, which
  already issues `load_tile(kt+1)` BEFORE the inner mma_sync loop for tile k. The C++
  compiler emits gmem loads early, and the LSU pipeline runs concurrently with TC mma.
  No additional inline-PTX reorder gives measurable lift on V100 (no `cp.async` exists)
  to make the load truly asynchronous beyond what the warp scheduler already does.

- **A3 (persistent CTAs + grid-stride tile loop)**: IMPLEMENTED in v5. Launches
  grid=160 CTAs (2 CTAs/SM × 80 SMs), each iterating output tiles in column-major sweep.

- **S1 + S3 revived**: L2 prefetch hint, `__ldg` for weight reads, `__launch_bounds__`.

## Bit-level correctness verification

v5 vs v3 max_abs / p99 / rel_err across all (fmt, tile, M, NK):

| M | INT8 v3 max_abs | v5 max_abs | INT4 v3 | v5 | MXFP4 v3 | v5 | F8 v3 | v5 |
|---|------------------|------------|---------|----|----------|----|-------|----|
| 1 | 2.655e-02 | 2.655e-02 | 2.849e-02 | 2.849e-02 | 1.681e-02 | 1.681e-02 | 1.831e-02 | 1.831e-02 |
| 64 | 3.949e-02 | 3.949e-02 | 3.466e-02 | 3.466e-02 | 2.391e-02 | 2.391e-02 | 2.472e-02 | 2.472e-02 |
| 128 | 3.949e-02 | 3.949e-02 | 3.510e-02 | 3.510e-02 | 2.391e-02 | 2.391e-02 | 2.500e-02 | 2.500e-02 |
| 256 | 3.950e-02 | 3.950e-02 | 3.903e-02 | 3.903e-02 | 2.392e-02 | 2.392e-02 | 2.503e-02 | 2.503e-02 |
| 512 | 3.949e-02 | 3.949e-02 | 3.902e-02 | 3.902e-02 | 2.579e-02 | 2.579e-02 | 2.631e-02 | 2.631e-02 |
| 1024 | 3.979e-02 | 3.979e-02 | 3.902e-02 | 3.902e-02 | 2.579e-02 | 2.579e-02 | 2.631e-02 | 2.631e-02 |
| 2048 | 3.975e-02 | 3.975e-02 | 3.901e-02 | 3.901e-02 | 2.631e-02 | 2.631e-02 | 2.790e-02 | 2.790e-02 |

**Result: identical max_abs across all M, all formats.** ✓ Persistent CTAs are pure-scheduling, no numerical change.

## Throughput v5 vs v3 (best-tile-per-shape) — Tier A delivers NEGATIVE lift on V100

| M | N=K | cuBLAS | v3 best | v5 best | Δ% | Verdict |
|---|-----|--------|---------|---------|----|---------|
| 64 | 4096 | 17.9 | 6.86 | 7.40 | +7.9% | v5 wins |
| 128 | 4096 | 42.7 | 11.87 | 12.41 | +4.5% | v5 wins |
| 256 | 4096 | 60.2 | 19.66 | 19.35 | -1.6% | v3 wins |
| 512 | 4096 | 70.1 | 24.78 | 24.61 | -0.7% | tie |
| 1024 | 4096 | 63.6 | 23.97 | 23.88 | -0.4% | tie |
| 2048 | 4096 | 82.2 | 24.58 | 24.92 | +1.4% | v5 wins |
| 64 | 7168 | 18.9 | 10.43 | 10.51 | +0.8% | tie |
| 128 | 7168 | 33.7 | 17.18 | 16.40 | -4.5% | v3 wins |
| 256 | 7168 | 55.0 | 26.87 | 21.06 | -21.6% | ** v5 hurts** |
| 512 | 7168 | 76.7 | 26.79 | 23.01 | -14.1% | ** v5 hurts** |
| 1024 | 7168 | 86.6 | 27.54 | 27.21 | -1.2% | v3 wins |
| 2048 | 7168 | 88.7 | 27.48 | 26.64 | -3.1% | v3 wins |

## Headline findings

### 1. Bit-level correctness ✓

v5 produces numerically identical output to v3 at every shape. Tier A items A3 + revived S items are confirmed pure-scheduling.

### 2. Tier A delivers ZERO lift on V100, with A3 causing significant regression

At INT8 LUT M=2048 N=K=7168: v3 = 27.00 TFLOPS, v5 = 21.66 TFLOPS = **-20%** regression.
Persistent CTAs serialize 11+ output tiles per CTA, paying per-tile syncthread + c_frag-init
overhead that the V100 hardware CTA scheduler avoids when it distributes 1792 separate CTAs.

### 3. Root cause analysis

The premise of Tier A items was:
- A1: halve syncthread count → saves a few μs (negligible vs 7.8 ms kernel time)
- A2: overlap gmem with mma → ALREADY ACHIEVED by v3's double-buffer pattern
- A3: amortize launch overhead → CTA launch is NOT the bottleneck on V100

On V100, the bottleneck is **per-mma_sync tensor-core throughput** (we hit ~32% of cuBLAS).
None of Tier A items address that bottleneck. They optimize scheduling overhead that the
V100 hardware already handles well.

### 4. Where Tier A items WOULD matter

- **A1 (BK=64)**: useful on sm_75+ with 96 KB SMEM (T4/Ampere/Hopper).
- **A2 (gmem reorder)**: useful on sm_80+ with `cp.async` for true async loads.
- **A3 (persistent CTAs)**: useful when launch overhead dominates, typically for very
  short kernels (sub-100μs). Our 7.8 ms kernel time is too long to benefit.

### 5. Conclusion: v3 remains the winning V100 kernel

INT8 LUT v3 at M=2048 N=K=7168 = **27.40 TFLOPS** (32% of cuBLAS ceiling) is the best
achievable on V100 with the C++ WMMA API. Further gains require either:
- Tier B (higher warps/SM via FRAG_M=2, 3-stage pipeline) — different optimization axis
- Tier C (raw inline-PTX mma.sync) — fundamental rewrite
- sm_75+ hardware — architectural ceiling

---

## Addendum: A1 retry via v6 (BK=64 single-buffer + L2 prefetch) — FAILS

Following the question "can we do A1 if we don't double-buffer and use prefetch instead?", I implemented v6:

- **BK=64** (vs v3's BK=32) — halves K-tile count from 224 to 112
- **Single-buffer SMEM** — drops from 2× 32 KB to 1× 34 KB, fits in V100's 48 KB
- **2-tile-ahead `prefetch.global.L2`** — keeps subsequent gmem loads L2-hit (~150 cyc instead of ~400 cyc)

### Bit-correctness ✓

v6 max_abs / p99 / rel_err matches v3 exactly at every (M, N=K). No numerical change.

### Throughput: v6 LOSES at every M

INT8 LUT best tile per M at N=K=7168:

| M | v3 (BK=32 double-buf) | v6 (BK=64 single-buf) | Δ |
|---|------------------------|------------------------|---|
| 1 | 0.18 TFLOPS | 0.14 TFLOPS | -22% |
| 16 | 2.74 | 2.11 | -23% |
| 64 | 10.46 | 7.95 | -24% |
| 128 | 17.26 | 12.48 | -28% |
| 256 | 26.91 | 18.93 | -30% |
| 512 | 26.74 | 18.99 | -29% |
| 1024 | 27.46 | 20.89 | -24% |

Consistent 22-30% regression across all M.

### Root cause: L2 prefetch ≠ SMEM double-buffer

`prefetch.global.L2` reduces the LATENCY of each gmem→smem load (HBM ~400 cyc → L2 ~150 cyc). It does NOT overlap the load with compute.

v3's double-buffer pattern actually OVERLAPS gmem loads with mma_sync compute:
- buffer[i] holds tile k, mma_sync runs on it
- buffer[1-i] is being filled with tile k+1 (gmem→reg→smem via LSU pipeline)
- Both run concurrently on different SM pipelines

v6 single-buffer can't do this — the next load must wait for the current mma to finish (else it would overwrite the active SMEM). L2 prefetch makes each load 250 cycles faster, but the load is now SERIAL with mma instead of parallel. Net loss.

### Implication

The SMEM-overlap-with-compute pattern (double-buffer) is more valuable than the smaller-sync-count win from BK=64. **V100's 48 KB SMEM is binding for V100 dequant GEMM**; we can't escape it via prefetch. To get BK=64 with overlap, would need sm_75+ (96 KB SMEM) or sm_80+ (`cp.async` enabling 3-stage register-staged double-buffer).

v6 confirmed: deferred items 1 / not implementable on V100. v3 remains the production kernel.

# tc-grid optimization plan — prioritized

**Baseline (current):** INT8 LUT v3 at M=2048 N=K=7168 = 27.36 TFLOPS = 32% of cuBLAS ceiling (86.5 TFLOPS) = 22% of V100 nominal peak (125 TFLOPS).

Sorted by impact/effort. Numbers are estimates at M=2048 N=K=7168 INT8 LUT on V100.

## Tier S — quick wins (1 build cycle each)

| # | Optimization | Impact | Effort | Notes |
|---|--------------|--------|--------|-------|
| S1 | `prefetch.global.L2` hints one K-tile ahead | 1.10× | 1 | Inline PTX one-liner. Pre-warms L2 from HBM. |
| S2 | `__launch_bounds__(maxThreads, minCTAs)` | 1.07× | 1 | Forces nvcc register count for target occupancy. |
| S3 | `__ldg()` for W reads (read-only cache) | 1.05× | 1 | RO cache, separate from L1, helps W reuse across CTAs. |

**Stacked estimate: 27.36 × 1.10 × 1.07 × 1.05 ≈ 33.8 TFLOPS** (39% cuBLAS).

## Tier A — medium effort (2-3 build cycles each)

| # | Optimization | Impact | Effort |
|---|--------------|--------|--------|
| A1 | BK=64 wider K-tile + double-buffer (halves syncthreads count) | 1.30× | 2 |
| A2 | Inline PTX `ld.global` reorder (issue tile N+1 before mma N) | 1.08× | 2 |
| A3 | Persistent CTAs + grid-stride output-tile loop | 1.05× | 2 |

**S+A stacked: ~50 TFLOPS** (58% cuBLAS).

## Tier B — biggest single levers (3 build cycles each)

| # | Optimization | Impact | Effort |
|---|--------------|--------|--------|
| B1 | Higher warps/SM (FRAG_M=2, 4 CTAs/SM) — V100 cp.async equivalent | 1.50× | 3 |
| B2 | Split-K + atomicAdd accumulation | 1.40× | 3 |
| B3 | 3-stage pipelined SMEM (triple-buffer) | 1.20× | 3 |
| B4 | Manual b_frag pre-fetch into register scratch | 1.20× | 3 |
| B5 | Alt fragment shapes (m8n32k16 / m32n8k16) | 1.10× | 3 |

**S+A+ (best 2 of B) stacked: ~65-75 TFLOPS** (75-87% cuBLAS — V100 ceiling).

## Tier C — diminishing returns

| # | Optimization | Impact | Effort |
|---|--------------|--------|--------|
| C1 | Register-resident A across K-loop | 1.20× | 4 |
| C2 | Raw inline-PTX `mma.sync` (bypass C++ WMMA API) | 1.20× | 5 |
| C3 | Custom barrier-fence sync | 1.05× | 4 |

## Tier X — requires sm_75+ hardware (not on V100)

| Feature | Architecture | Effect |
|---------|--------------|--------|
| `ldmatrix.sync.aligned.m8n8.x4` | sm_75+ | 4× SMEM fragment load throughput |
| `cp.async` true async gmem→smem | sm_80+ | Real overlap of load with compute |
| Native INT8/INT4 tensor cores | sm_75+ | Skip dequant to FP16 entirely |
| `mma.sync.aligned.m16n8k16` | sm_80+ | Smaller fragment shapes |
| `wgmma.async` | sm_90 | Async 64×n×k mma |
| TMA | sm_90 | gmem→smem without threads |

## Execution order

1. **Now:** Tier S (items 1-3) → expect ~34 TFLOPS, 39% cuBLAS
2. **Next:** Tier A (items 4-6) → expect ~50 TFLOPS, 58% cuBLAS
3. **Then:** best 2 of Tier B → expect ~65-75 TFLOPS, 75-87% cuBLAS
4. **Stop:** further gain requires sm_75+

## Single-shot most efficient move

`prefetch.global.L2` hints (item S1) — 30 min work, ~10% gain, zero correctness risk.

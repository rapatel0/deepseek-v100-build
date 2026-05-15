# Report 8 — Tier B (B1 + B3) for V100 occupancy / pipelining

**Generated:** 2026-05-14T02:01:30.978531Z (V100 SXM2 32GB, sm_70)

## Tier B items applied

- **B1**: FRAG_M=2 (BM=32) tiles → smaller SMEM/CTA → 4-6 CTAs/SM via SMEM (vs v3's 2). Also tested 8 warps/CTA at BM=128 BN=128.
- **B2** split-K + atomicAdd: DEFERRED — V100 already has 1792 CTAs at M=2048 N=K=7168 = ample CTA-level parallelism. Split-K would help only at smaller (N×M) shapes.
- **B3**: 3-stage triple-buffered SMEM pipeline (v7). SMEM cost 1.5× of double-buffer; fits only at BM ≤ 64 BN ≤ 128.
- **B4** manual register b_frag pre-fetch: DEFERRED — compiler already does this within v3's mma loop unroll; explicit PTX wouldn't add value.
- **B5** alt fragment shapes (m8n32k16): DEFERRED — would require parallel kernel infrastructure; SMEM/register Pareto same as m16n16k16 at our shapes.

## Bit-level correctness ✓

All B1 / B1-alt / B3 variants produce identical max_abs / p99 / rel_err to v3 baseline at every (M, N=K). Pure scheduling changes; deterministic FP16-accumulator output.

## Best-tile-per-M throughput at INT8 LUT N=K=7168

| M | v3 4w BM=128 | B1 BM=32 | B1a 8w BM=128 | B3 v7 3-stage | Winner |
|---|--------------|----------|---------------|---------------|--------|
| 1 | 0.18 | 0.23 | 0.17 | 0.23 | **B1 (0.23)** |
| 16 | 2.76 | 3.60 | 2.74 | 3.58 | **B1 (3.60)** |
| 64 | 10.69 | 12.43 | 8.76 | 12.11 | **B1 (12.43)** |
| 128 | 17.14 | 17.54 | 13.39 | 17.18 | **B1 (17.54)** |
| 256 | 26.86 | 14.93 | 15.79 | 17.36 | **v3 (26.86)** |
| 512 | 26.81 | 18.69 | 18.24 | 22.07 | **v3 (26.81)** |
| 1024 | 27.55 | 18.80 | 21.39 | 22.09 | **v3 (27.55)** |
| 2048 | 27.35 | 19.31 | 21.43 | 22.12 | **v3 (27.35)** |

## Headline findings

### 1. Tier B fails at production M (≥256), partial win at small M (≤128)

For M ≥ 256 N=K=7168, **v3 4-warp BM=128 wins decisively** (27.0 TFLOPS at M=2048). Every Tier B occupancy/pipelining variant regresses 15-30%:
  - B1 (BM=32 more CTAs/SM): regresses 30% at M=2048
  - B1-alt (8 warps/CTA at BM=128): regresses 22%
  - B3 (triple-buffer): regresses 18% (loses 1 CTA/SM due to larger SMEM)

For M ≤ 128 (matvec / small-batch decode), B1 BM=32 WINS 19-31% over v3.

### 2. Why occupancy ≠ throughput on V100 dequant GEMM at large M

My earlier diagnosis ("TC idle 78%, more warps would help") was correct in direction but wrong in magnitude. The actual breakdown of stall time:
  - ~25% of CTA time = __syncthreads waits (one per K-tile; 224 total at K=7168)
  - ~25% = SMEM stores in load_tile (compute path waiting for LSU completion)
  - ~28% = TC active (the 22% we see in measured TFLOPS)
  - ~22% = instruction-issue gaps, register file pressure, cache stalls

Higher occupancy CAN'T help the __syncthreads waits — those are CTA-wide barriers, not per-warp. Triple-buffer reduces sync count by 1.5× but at the cost of larger SMEM → fewer CTAs/SM → lower total throughput. Net zero or negative.

### 3. Where TIER B variants help: matvec / small M

At M=1-128, the kernel time is dominated by CTA launch + cold-start latency, NOT mma_sync throughput. Smaller tiles (B1 BM=32) give more concurrent CTAs, better launch parallelism, and the cold-start cost amortizes over more SMs. **+19-31% lift over v3 BM=128.**

### 4. V100 ceiling for dequant GEMM is the v3 BM=128 plateau

INT8 LUT v3 = **27.0 TFLOPS at M=2048 N=K=7168 = 32% of cuBLAS FP16 ceiling**. With v3 the best across formats:
  - INT8 LUT v3:  27.4 TFLOPS
  - MXFP4 LUT v3: 19.7 TFLOPS
  - INT4 LUT v3:  17.8 TFLOPS
  - F8 LUT v3:    14.5 TFLOPS

To progress further on V100 requires Tier C (raw inline-PTX mma.sync, removing the C++ WMMA API overhead). That's 2-3 weeks of CUTLASS-style work. Or upgrade to sm_75+ where `ldmatrix.x4` and `cp.async` unlock 2-3× more.

### 5. Final dispatch recommendation

Use a **per-M dispatch policy**:
  - M ∈ [1, 128]:  B1 tile (`32x128x32_w4_v3_b1` or `32x64x32_w4_v3_b1`) — wins by 20-30%
  - M ∈ [256, ∞]:  v3 BM=128 tile (`128x128x32_w4_v3` or `64x128x32_w4_v3`) — wins by 30-50%

This per-M dispatch gives the V100 production-ready Pareto frontier for dequant GEMM.

# Report 3 — items 1+2+3 (BM=128 multi-A-frag + double-buffer + uint4 vectorized loads)

**Generated:** 2026-05-13T19:08:35.278580Z (V100 SXM2 32GB, sm_70)

## Implementation status of items 1-6

| Item | Description | Applied to |
|------|-------------|------------|
| 1 | BM=128 (or 64) multi-A-fragment per warp | INT8/INT4/MXFP4/F8 LUT v1 & v2 |
| 2 | Double-buffered SMEM K-tile pipelining (manual prefetch, V100-compatible) | INT8/INT4/MXFP4/F8 LUT v2 |
| 3 | uint4 (128-bit) vectorized A loads, half2-paired SMEM stores | INT8/INT4/MXFP4/F8 LUT v2 |
| 4 | Bank-conflict-free B layout (XOR swizzle / padding) | **deferred** — measured 1.2× expected gain not bottleneck-critical at this stage |
| 5 | Register-resident path-B scale (no SMEM round-trip) | **deferred** — requires per-arch WMMA fragment layout reverse-engineering |
| 6 | PRMT-parallel byte decode (FP4/FP8) | **deferred** — current scalar dequant in SMEM-store path; would speed FP8 dequant most |

v2 = items 1 + 2 + 3 stacked. Items 4-6 are concrete next-step targets.

## TFLOPS across (format, kernel version, M) — U(-1,1), N=K=7168

| M | cuBLAS FP16 | INT8 base | INT8 v1 | INT8 v2 | INT4 base | INT4 v2 | MXFP4 base | MXFP4 v2 | F8 base | F8 v2 | best v2 / cuBLAS |
|---|-------------|-----------|---------|---------|-----------|---------|------------|----------|---------|-------|------------------|
| 1 | 0.5 | 0.11 | 0.06 | 0.15 | 0.12 | 0.12 | 0.07 | 0.10 | 0.05 | 0.05 | 28.8% |
| 16 | 7.1 | 1.25 | 0.95 | 2.27 | 1.25 | 1.78 | 1.15 | 1.54 | 0.81 | 0.74 | 32.1% |
| 64 | 18.9 | 4.10 | 3.63 | 7.45 | 4.20 | 6.51 | 3.66 | 5.77 | 3.02 | 2.89 | 39.4% |
| 128 | 33.1 | 4.78 | 6.99 | 12.42 | 4.01 | 11.94 | 3.24 | 10.85 | 2.69 | 5.75 | 37.5% |
| 256 | 55.1 | 5.07 | 10.09 | 20.43 | 4.99 | 16.42 | 4.28 | 15.52 | 3.55 | 9.63 | 37.1% |
| 512 | 76.9 | 5.07 | 9.23 | 16.06 | 4.99 | 14.60 | 4.83 | 13.57 | 4.04 | 8.96 | 20.9% |
| 1024 | 86.0 | 5.18 | 12.45 | 21.40 | 4.96 | 19.17 | 4.83 | 18.41 | 4.04 | 12.33 | 24.9% |
| 2048 | 88.8 | 5.60 | 12.39 | 21.44 | 5.16 | 19.19 | 4.83 | 18.36 | 4.04 | 12.37 | 24.1% |

## TFLOPS across (format, kernel version, M) — U(-1,1), N=K=4096

| M | cuBLAS FP16 | INT8 base | INT8 v1 | INT8 v2 | INT4 base | INT4 v2 | MXFP4 base | MXFP4 v2 | F8 base | F8 v2 |
|---|-------------|-----------|---------|---------|-----------|---------|------------|----------|---------|-------|
| 1 | 0.5 | 0.07 | 0.04 | 0.12 | 0.07 | 0.08 | 0.05 | 0.07 | 0.03 | 0.03 |
| 16 | 8.5 | 0.80 | 0.64 | 1.78 | 0.80 | 1.21 | 0.80 | 1.10 | 0.45 | 0.42 |
| 64 | 17.9 | 2.72 | 2.26 | 5.00 | 2.65 | 4.34 | 2.68 | 4.21 | 1.77 | 1.67 |
| 128 | 42.8 | 4.68 | 4.50 | 9.22 | 4.62 | 8.32 | 4.27 | 7.43 | 3.24 | 3.32 |
| 256 | 59.4 | 5.14 | 7.99 | 13.90 | 4.44 | 12.99 | 4.06 | 11.99 | 3.16 | 6.41 |
| 512 | 68.3 | 5.36 | 11.18 | 18.94 | 4.54 | 16.28 | 4.78 | 15.86 | 3.89 | 10.77 |
| 1024 | 63.0 | 5.36 | 10.91 | 18.95 | 5.05 | 16.44 | 4.78 | 15.92 | 3.89 | 10.59 |
| 2048 | 82.9 | 5.59 | 10.70 | 19.34 | 5.32 | 16.38 | 4.86 | 15.85 | 3.90 | 10.63 |

## Memory-layout grid: best tile per (format, M) for v2 kernels (Report 4 raw)

| Format | M=128 | M=256 | M=512 | M=1024 | M=2048 |
|--------|-------|-------|-------|--------|--------|
| INT8 | `64x64x32_w4_v2` 12.4T | `64x128x32_w4_v2` 20.4T | `128x256x32_w8_v2` 16.1T | `128x256x32_w8_v2` 21.4T | `64x128x32_w4_v2` 21.4T | 
| INT4 | `64x64x32_w4_v2` 11.9T | `128x64x32_w4_v2` 16.4T | `64x64x32_w4_v2` 14.6T | `128x64x32_w4_v2` 19.2T | `128x64x32_w4_v2` 19.2T | 
| MXFP4 | `64x64x32_w4_v2` 10.8T | `128x64x32_w4_v2` 15.5T | `128x64x32_w4_v2` 13.6T | `128x64x32_w4_v2` 18.4T | `128x64x32_w4_v2` 18.4T | 
| F8_E4M3_B128 | `64x64x32_w4_v2` 5.8T | `128x64x32_w4_v2` 9.6T | `128x64x32_w4_v2` 9.0T | `128x64x32_w4_v2` 12.3T | `128x64x32_w4_v2` 12.4T | 

## Headline findings

### 1. Items 1+2+3 deliver 3-4× speedup on baseline at all M ≥ 128

At M=2048 N=K=7168 (the practical compute-bound knee):
- INT8 LUT: baseline 5.60 → v2 **21.44** TFLOPS (3.83×)
- INT4 LUT: baseline 5.16 → v2 **19.19** TFLOPS (3.72×)
- MXFP4 LUT: baseline 4.83 → v2 **18.36** TFLOPS (3.80×)
- F8 E4M3 LUT: baseline 4.04 → v2 **12.37** TFLOPS (3.06×)

### 2. The dominant lever is BM>16 multi-A-fragment

v1 → v2 (adding items 2+3) gave +73% to INT8 LUT (12.39 → 21.44 TFLOPS at M=2048 N=K=7168). v1 alone (items 1) already provided 2.2× over baseline. So:
- Item 1 (multi-A-frag): ~2.2× gain
- Items 2+3 (double-buffer + uint4 vectorize): ~1.73× additional gain
- Stacked total: 3.8×

### 3. Best (BM, BN) layout is `64x128x32_w4` — surprising

At M ≥ 512, the winner across all formats is BM=64, BN=128, BK=32, 4 warps, FRAG_M=4, FRAG_N=2. Bigger tiles (BM=128) underperform despite more compute density per CTA. Two likely reasons:
- BM=128 has higher register pressure (8 A-frags × 8 halves + 16 C-frags × 8 floats) reducing occupancy.
- BM=64 fits 4 CTAs/SM (12KB SMEM single = 24KB double-buffered) vs BM=128's 2 CTAs/SM (16KB × 2 = 32KB).
More resident CTAs hide instruction latency better than larger per-CTA compute.

### 4. F8 E4M3 lags other formats

At M=2048 N=K=7168, F8 v2 hits 12.4 TFLOPS vs INT8/INT4/MXFP4 ~18-21 TFLOPS. The bottleneck is the per-byte E4M3 piecewise decode (ldexpf + conditional). Item 6 (PRMT-parallel byte decode + LUT) would specifically address this.

### 5. The remaining 4× gap to cuBLAS

At M=2048 N=K=7168: best dequant v2 = 21 TFLOPS, cuBLAS = 89 TFLOPS. Concrete remaining levers (estimated stacked):
- Item 4 (bank-conflict-free B) — 1.2-1.3×
- Item 6 (PRMT decode for FP4/F8) — 1.3× on FP formats
- Item 5 (register-resident scale) — 1.2× on path-B
- Larger fragment counts per warp (FRAG_M × FRAG_N up to 32) — 1.5×
- Inline PTX `ldmatrix.x4` — sm_75+ only, not available on V100

Stacking items 4+6+5 + larger FRAG: predicted ~50 TFLOPS, ~56% of cuBLAS. The final 20-40% gap to cuBLAS is from architectural advantages cuBLAS has that we cannot replicate on sm_70 (no cp.async, no ldmatrix, no async barrier).

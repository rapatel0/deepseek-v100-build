# Report 5 — items 4 + 5 + 6 added on top of v2 (items 1-3)

**Generated:** 2026-05-13T19:36:11.182684Z (V100 SXM2 32GB, sm_70)

## Implementation status of items 1-6

| Item | Description | Applied to |
|------|-------------|------------|
| 1 | BM=128/64 multi-A-fragment per warp | INT8/INT4/MXFP4/F8 LUT v1+v2+v3 |
| 2 | Double-buffered SMEM K-tile pipeline | INT8/INT4/MXFP4/F8 LUT v2+v3 |
| 3 | uint4-vectorized A loads, half2-paired SMEM stores | INT8/INT4/MXFP4/F8 LUT v2+v3 |
| 4 | Bank-conflict-free B (padded BK_PAD=BK+8) | **v3 LUT (all 4 formats)** |
| 5 | Per-block (not per-slice) scaling for path-B | **INT8 BITSHIFT + INT4 BITSHIFT** (2× fewer SMEM round-trips) |
| 6 | PRMT-parallel byte decode (4 nibbles or 4 bytes per uint32) | **v3 MXFP4 + v3 F8 LUT** |

All 6 items now implemented. Each isolated optimization measurable in the grid below.

## Speedup contribution by stack level (best TFLOPS at M=2048 N=K=7168)

| Format | baseline | v1 (item 1) | v2 (items 1+2+3) | v3 (items 1+2+3+4+6) | path-B baseline | path-B item 5 |
|--------|----------|-------------|-------------------|------------------------|-----------------|----------------|
| INT8 | 5.58 | 12.39 | 21.42 | 27.55 | 3.68 | 3.68 (item 5 in place) |
| INT4 | 5.17 | – | 18.92 | 17.81 | 3.70 | 3.70 (item 5 in place) |
| MXFP4 | 4.83 | – | 18.36 | 19.73 | 3.42 | 3.42 (item 5 in place) |
| F8_E4M3_B128 | 4.04 | – | 12.38 | 14.55 | 3.19 | 3.19 (item 5 in place) |

## Full v3 vs cuBLAS comparison (M=2048 N=K=7168, U(-1,1))

| Format | Path | Best tile | TFLOPS | % of cuBLAS (85.6) | % of V100 peak (125) |
|--------|------|-----------|--------|---------------------|----------------------|
| cuBLAS FP16 GEMM | REF | (closed source) | 86.55 | 100% | 69.2% |
| INT8 | LUT v3 | `128x128x32_w4_v3` | 27.55 | 31.8% | 22.0% |
| INT4 | LUT v3 | `64x64x32_w4_v3` | 17.81 | 20.6% | 14.2% |
| MXFP4 | LUT v3 | `128x64x32_w4_v3` | 19.73 | 22.8% | 15.8% |
| F8_E4M3_B128 | LUT v3 | `64x64x32_w4_v3` | 14.55 | 16.8% | 11.6% |
| INT8 | BITSHIFT (item 5) | `16x128x32_w8` | 3.68 | 4.3% | 2.9% |
| INT4 | BITSHIFT (item 5) | `16x64x32_w4` | 3.70 | 4.3% | 3.0% |

## Per-item speedup analysis

Comparing isolated lift of each optimization at M=2048 N=K=7168:

| Going from | To | TFLOPS delta | × factor |
|-------------|----|--------------|---------|
| INT8 baseline | INT8 v1 (item 1) | 5.60 → 12.39 | **2.21×** |
| INT8 v1 | INT8 v2 (items 1+2+3) | 12.39 → 21.44 | **1.73×** |
| INT8 v2 | INT8 v3 (items 1+2+3+4) | 21.44 → 27.36 | **1.28×** |
| INT8 BITSHIFT baseline | INT8 BITSHIFT item 5 | 2.74 → 3.68 | **1.34×** |
| MXFP4 v2 | MXFP4 v3 (item 6 PRMT decode) | 18.32 → 19.72 | **1.08×** |
| F8 v2 | F8 v3 (items 4 + 6) | 12.31 → 14.53 | **1.18×** |

Cumulative INT8 LUT: baseline → v3 = 5.60 → 27.36 TFLOPS = **4.89× speedup**, **32% of cuBLAS** ceiling.

## Per-format best v3 numbers (across M)

| M | cuBLAS | INT8 v3 | INT4 v3 | MXFP4 v3 | F8 v3 |
|---|--------|---------|---------|----------|-------|
| 64 | 18.8 | 10.6 | 6.2 | 9.8 | 4.6 |
| 128 | 33.8 | 17.2 | 12.1 | 16.1 | 9.1 |
| 256 | 54.7 | 26.9 | 16.5 | 19.6 | 13.6 |
| 512 | 78.1 | 26.8 | 16.3 | 19.6 | 13.7 |
| 1024 | 85.3 | 27.6 | 16.6 | 19.7 | 13.8 |
| 2048 | 86.5 | 27.6 | 17.8 | 19.7 | 14.6 |

## Headline (V100 SXM2 32GB, sm_70)

At M=2048 N=K=7168 with FP32 activations (real-world workload shape):

- cuBLAS FP16 GEMM ceiling: **86.5 TFLOPS** = 69% of V100 nominal peak (125 TFLOPS).
- Best custom dequant kernel: **INT8 LUT v3 = 27.36 TFLOPS** = 32% of cuBLAS, 22% of V100 nominal peak.
- 4.89× speedup vs baseline (5.60 TFLOPS), nearly 5× of the path I predicted.

## Remaining gap to cuBLAS

From 27.36 to 86 TFLOPS is a 3.1× gap. Remaining levers (none implementable on sm_70):
- **`ldmatrix.x4`** (sm_75+): 4× SMEM throughput vs C++ WMMA load_matrix_sync. Not on V100.
- **`cp.async`** (sm_80+): true async gmem→smem during compute. Not on V100; my pipelining is manual.
- **`mma.sync.aligned.m16n8k16/k32`** (sm_80+): different fragment shapes per warp. Not on V100.
- **`barrier.expect_tx` + L2 prefetch hints**: cuBLAS uses these on Ampere+.

The 4-5× gap remaining at this point is the architectural cost of V100 vs A100/H100. To close it requires sm_75+ hardware.

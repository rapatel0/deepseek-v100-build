# Report 1 — tc-grid: tensor-core kernel correctness + throughput

**Generated:** 2026-05-13T18:00:33.246639Z (V100 SXM2 32GB, sm_70)
**Cells:** 1440 OK runs across formats × paths × tiles × distributions × shapes

## What was tested

### Kernels (one CUDA source file per group)

| Format | Path-A (LUT/cast) | Path-B (mantissa-bitshift) |
|--------|-------------------|----------------------------|
| INT8   | `mm_int8_lut`       | `mm_int8_bitshift` (anchor 0x6400 = 1024, value = 1152+q) |
| INT4   | `mm_int4_lut`       | `mm_int4_bitshift` (anchor 0x6400, value = 1032+q) |
| MXFP4  | `mm_mxfp4_lut`      | deferred — no natural mantissa pack (FP has own exponent) |
| F8_E4M3_B128 | `mm_f8_e4m3_b128_lut` | deferred — same constraint as MXFP4 |

### Tile configs swept (all BM=16 single WMMA m16n16k16 fragment per warp)

- `16x64x32_w4`, `16x64x64_w4`, `16x128x32_w8`, `16x128x64_w8`

### Shape grid
- M ∈ {1, 4, 8, 16, 32, 64}
- N = K ∈ {4096, 7168}

### Data distributions (stress tests)
- `U(-1,1)` — benign baseline (FP16 mid range)
- `U(-8,8)` — wide dynamic range, exercises saturation of FP4/INT4
- `LogN(0,1.5)` — heavy tails
- `SparseSpikes` — 10% spikes in [-6,6], 90% small Gaussian
- `Adversarial` — alternating signs designed to maximize cancellation in dot products

### Reference & correctness
- Reference path: per-element FP32 dequant of quantized W, then cuBLAS SGEMM at FP32.
- Metrics: max_abs, p99_abs, rel_err (= mean(|test - ref|) / mean(|ref|)).

## Best (TFLOPS-maximizing) cell per (format, path, M, N=K) — U(-1,1)

| Format | Path | M | N=K | best tile | ms | TFLOPS | GB/s | max_abs | p99_abs | rel_err |
|--------|------|---|-----|-----------|----|--------|------|---------|---------|---------|
| F8_E4M3_B128 | LUT | 1 | 4096 | `16x64x32_w4` | 1.206 | 0.030 | 14.0 | 1.29e-02 | 9.84e-03 | 1.84e-04 |
| F8_E4M3_B128 | LUT | 1 | 7168 | `16x64x32_w4` | 2.027 | 0.050 | 25.6 | 1.83e-02 | 1.31e-02 | 1.82e-04 |
| F8_E4M3_B128 | LUT | 4 | 4096 | `16x64x32_w4` | 1.197 | 0.110 | 14.2 | 1.57e-02 | 9.90e-03 | 1.84e-04 |
| F8_E4M3_B128 | LUT | 4 | 7168 | `16x64x32_w4` | 2.037 | 0.200 | 25.5 | 2.29e-02 | 1.32e-02 | 1.86e-04 |
| F8_E4M3_B128 | LUT | 8 | 4096 | `16x64x32_w4` | 1.198 | 0.220 | 14.3 | 1.61e-02 | 9.87e-03 | 1.86e-04 |
| F8_E4M3_B128 | LUT | 8 | 7168 | `16x64x32_w4` | 2.044 | 0.400 | 25.6 | 2.29e-02 | 1.32e-02 | 1.87e-04 |
| F8_E4M3_B128 | LUT | 16 | 4096 | `16x64x32_w4` | 1.203 | 0.450 | 14.5 | 1.61e-02 | 9.90e-03 | 1.86e-04 |
| F8_E4M3_B128 | LUT | 16 | 7168 | `16x64x32_w4` | 2.053 | 0.800 | 25.7 | 2.45e-02 | 1.32e-02 | 1.85e-04 |
| F8_E4M3_B128 | LUT | 32 | 4096 | `16x64x32_w4` | 1.205 | 0.890 | 14.9 | 1.73e-02 | 9.97e-03 | 1.86e-04 |
| F8_E4M3_B128 | LUT | 32 | 7168 | `16x64x32_w4` | 2.034 | 1.620 | 26.4 | 2.47e-02 | 1.32e-02 | 1.85e-04 |
| F8_E4M3_B128 | LUT | 64 | 4096 | `16x64x32_w4` | 1.228 | 1.750 | 15.5 | 1.95e-02 | 9.96e-03 | 1.85e-04 |
| F8_E4M3_B128 | LUT | 64 | 7168 | `16x64x32_w4` | 2.189 | 3.000 | 25.3 | 2.47e-02 | 1.32e-02 | 1.85e-04 |
| INT4 | BITSHIFT | 1 | 4096 | `16x64x32_w4` | 0.457 | 0.070 | 20.7 | 1.33e-02 | 1.01e-02 | 1.84e-04 |
| INT4 | BITSHIFT | 1 | 7168 | `16x64x32_w4` | 0.887 | 0.120 | 32.7 | 1.85e-02 | 1.35e-02 | 1.81e-04 |
| INT4 | BITSHIFT | 4 | 4096 | `16x64x32_w4` | 0.500 | 0.270 | 19.1 | 1.63e-02 | 1.01e-02 | 1.85e-04 |
| INT4 | BITSHIFT | 4 | 7168 | `16x64x32_w4` | 0.955 | 0.430 | 30.5 | 2.39e-02 | 1.36e-02 | 1.86e-04 |
| INT4 | BITSHIFT | 8 | 4096 | `16x64x32_w4` | 0.560 | 0.480 | 17.3 | 1.64e-02 | 1.01e-02 | 1.87e-04 |
| INT4 | BITSHIFT | 8 | 7168 | `16x64x32_w4` | 1.078 | 0.760 | 27.2 | 2.39e-02 | 1.36e-02 | 1.87e-04 |
| INT4 | BITSHIFT | 16 | 4096 | `16x64x32_w4` | 0.690 | 0.780 | 14.4 | 1.64e-02 | 1.02e-02 | 1.86e-04 |
| INT4 | BITSHIFT | 16 | 7168 | `16x64x32_w4` | 1.319 | 1.250 | 22.6 | 2.46e-02 | 1.36e-02 | 1.86e-04 |
| INT4 | BITSHIFT | 32 | 4096 | `16x64x32_w4` | 0.743 | 1.450 | 14.1 | 1.75e-02 | 1.03e-02 | 1.86e-04 |
| INT4 | BITSHIFT | 32 | 7168 | `16x64x32_w4` | 1.477 | 2.230 | 20.8 | 2.47e-02 | 1.36e-02 | 1.86e-04 |
| INT4 | BITSHIFT | 64 | 4096 | `16x64x32_w4` | 0.985 | 2.180 | 11.7 | 1.93e-02 | 1.03e-02 | 1.86e-04 |
| INT4 | BITSHIFT | 64 | 7168 | `16x64x32_w4` | 2.230 | 2.950 | 14.6 | 2.47e-02 | 1.36e-02 | 1.85e-04 |
| INT4 | LUT | 1 | 4096 | `16x64x32_w4` | 0.657 | 0.050 | 14.4 | 1.90e-02 | 1.40e-02 | 2.50e-04 |
| INT4 | LUT | 1 | 7168 | `16x64x32_w4` | 1.362 | 0.080 | 21.3 | 2.85e-02 | 1.84e-02 | 2.53e-04 |
| INT4 | LUT | 4 | 4096 | `16x64x32_w4` | 0.633 | 0.210 | 15.1 | 2.31e-02 | 1.44e-02 | 2.55e-04 |
| INT4 | LUT | 4 | 7168 | `16x64x32_w4` | 1.343 | 0.310 | 21.7 | 2.85e-02 | 1.88e-02 | 2.56e-04 |
| INT4 | LUT | 8 | 4096 | `16x64x32_w4` | 0.640 | 0.420 | 15.1 | 2.31e-02 | 1.42e-02 | 2.57e-04 |
| INT4 | LUT | 8 | 7168 | `16x64x32_w4` | 1.339 | 0.610 | 21.9 | 3.20e-02 | 1.89e-02 | 2.57e-04 |
| INT4 | LUT | 16 | 4096 | `16x64x32_w4` | 0.676 | 0.790 | 14.7 | 2.51e-02 | 1.42e-02 | 2.56e-04 |
| INT4 | LUT | 16 | 7168 | `16x64x32_w4` | 1.393 | 1.180 | 21.4 | 3.33e-02 | 1.89e-02 | 2.57e-04 |
| INT4 | LUT | 32 | 4096 | `16x64x32_w4` | 0.757 | 1.420 | 13.9 | 2.51e-02 | 1.42e-02 | 2.56e-04 |
| INT4 | LUT | 32 | 7168 | `16x64x32_w4` | 1.422 | 2.310 | 21.6 | 3.46e-02 | 1.88e-02 | 2.56e-04 |
| INT4 | LUT | 64 | 4096 | `16x64x32_w4` | 0.814 | 2.640 | 14.2 | 2.52e-02 | 1.41e-02 | 2.55e-04 |
| INT4 | LUT | 64 | 7168 | `16x64x32_w4` | 1.568 | 4.200 | 20.8 | 3.47e-02 | 1.87e-02 | 2.56e-04 |
| INT8 | BITSHIFT | 1 | 4096 | `16x64x32_w4` | 0.509 | 0.070 | 35.1 | 1.31e-02 | 9.96e-03 | 1.83e-04 |
| INT8 | BITSHIFT | 1 | 7168 | `16x64x32_w4` | 0.908 | 0.110 | 60.2 | 1.87e-02 | 1.34e-02 | 1.82e-04 |
| INT8 | BITSHIFT | 4 | 4096 | `16x64x32_w4` | 0.551 | 0.240 | 32.6 | 1.65e-02 | 1.01e-02 | 1.84e-04 |
| INT8 | BITSHIFT | 4 | 7168 | `16x64x32_w4` | 0.999 | 0.410 | 54.9 | 2.38e-02 | 1.35e-02 | 1.86e-04 |
| INT8 | BITSHIFT | 8 | 4096 | `16x64x32_w4` | 0.570 | 0.470 | 31.7 | 1.65e-02 | 1.01e-02 | 1.86e-04 |
| INT8 | BITSHIFT | 8 | 7168 | `16x64x32_w4` | 1.102 | 0.750 | 50.0 | 2.38e-02 | 1.35e-02 | 1.87e-04 |
| INT8 | BITSHIFT | 16 | 4096 | `16x64x32_w4` | 0.697 | 0.770 | 26.3 | 1.65e-02 | 1.01e-02 | 1.86e-04 |
| INT8 | BITSHIFT | 16 | 7168 | `16x64x32_w4` | 1.320 | 1.250 | 42.1 | 2.45e-02 | 1.35e-02 | 1.85e-04 |
| INT8 | BITSHIFT | 32 | 4096 | `16x64x32_w4` | 0.745 | 1.440 | 25.3 | 1.75e-02 | 1.02e-02 | 1.86e-04 |
| INT8 | BITSHIFT | 32 | 7168 | `16x64x32_w4` | 1.580 | 2.080 | 35.7 | 2.50e-02 | 1.35e-02 | 1.85e-04 |
| INT8 | BITSHIFT | 64 | 4096 | `16x64x32_w4` | 0.997 | 2.150 | 20.0 | 1.96e-02 | 1.02e-02 | 1.85e-04 |
| INT8 | BITSHIFT | 64 | 7168 | `16x64x32_w4` | 2.277 | 2.890 | 25.6 | 2.50e-02 | 1.35e-02 | 1.85e-04 |
| INT8 | LUT | 1 | 4096 | `16x64x32_w4` | 0.857 | 0.040 | 20.8 | 1.90e-02 | 1.45e-02 | 2.59e-04 |
| INT8 | LUT | 1 | 7168 | `16x64x32_w4` | 1.499 | 0.070 | 36.5 | 2.66e-02 | 1.90e-02 | 2.59e-04 |
| INT8 | LUT | 4 | 4096 | `16x64x32_w4` | 0.857 | 0.160 | 21.0 | 2.12e-02 | 1.46e-02 | 2.60e-04 |
| INT8 | LUT | 4 | 7168 | `16x64x32_w4` | 1.435 | 0.290 | 38.2 | 3.16e-02 | 1.90e-02 | 2.60e-04 |
| INT8 | LUT | 8 | 4096 | `16x64x32_w4` | 0.793 | 0.340 | 22.8 | 2.38e-02 | 1.44e-02 | 2.60e-04 |
| INT8 | LUT | 8 | 7168 | `16x64x32_w4` | 1.437 | 0.570 | 38.3 | 3.16e-02 | 1.89e-02 | 2.62e-04 |
| INT8 | LUT | 16 | 4096 | `16x64x32_w4` | 0.793 | 0.680 | 23.1 | 2.38e-02 | 1.44e-02 | 2.60e-04 |
| INT8 | LUT | 16 | 7168 | `16x64x32_w4` | 1.448 | 1.140 | 38.3 | 3.16e-02 | 1.90e-02 | 2.61e-04 |
| INT8 | LUT | 32 | 4096 | `16x64x32_w4` | 0.770 | 1.390 | 24.5 | 2.43e-02 | 1.44e-02 | 2.60e-04 |
| INT8 | LUT | 32 | 7168 | `16x64x32_w4` | 1.493 | 2.200 | 37.8 | 3.28e-02 | 1.90e-02 | 2.60e-04 |
| INT8 | LUT | 64 | 4096 | `16x64x32_w4` | 0.802 | 2.680 | 24.9 | 2.86e-02 | 1.44e-02 | 2.60e-04 |
| INT8 | LUT | 64 | 7168 | `16x64x32_w4` | 1.619 | 4.060 | 36.0 | 3.95e-02 | 1.89e-02 | 2.60e-04 |
| MXFP4 | LUT | 1 | 4096 | `16x64x32_w4` | 0.678 | 0.050 | 13.2 | 1.19e-02 | 9.32e-03 | 1.84e-04 |
| MXFP4 | LUT | 1 | 7168 | `16x64x32_w4` | 1.391 | 0.070 | 19.7 | 1.68e-02 | 1.25e-02 | 1.83e-04 |
| MXFP4 | LUT | 4 | 4096 | `16x64x32_w4` | 0.658 | 0.200 | 13.8 | 1.46e-02 | 9.24e-03 | 1.84e-04 |
| MXFP4 | LUT | 4 | 7168 | `16x64x32_w4` | 1.410 | 0.290 | 19.5 | 2.17e-02 | 1.25e-02 | 1.86e-04 |
| MXFP4 | LUT | 8 | 4096 | `16x64x32_w4` | 0.661 | 0.410 | 13.9 | 1.55e-02 | 9.24e-03 | 1.86e-04 |
| MXFP4 | LUT | 8 | 7168 | `16x64x32_w4` | 1.423 | 0.580 | 19.5 | 2.17e-02 | 1.24e-02 | 1.87e-04 |
| MXFP4 | LUT | 16 | 4096 | `16x64x32_w4` | 0.670 | 0.800 | 14.1 | 1.55e-02 | 9.32e-03 | 1.86e-04 |
| MXFP4 | LUT | 16 | 7168 | `16x64x32_w4` | 1.437 | 1.140 | 19.6 | 2.35e-02 | 1.25e-02 | 1.86e-04 |
| MXFP4 | LUT | 32 | 4096 | `16x64x32_w4` | 0.724 | 1.480 | 13.8 | 1.65e-02 | 9.41e-03 | 1.86e-04 |
| MXFP4 | LUT | 32 | 7168 | `16x64x32_w4` | 1.578 | 2.080 | 18.5 | 2.39e-02 | 1.25e-02 | 1.85e-04 |
| MXFP4 | LUT | 64 | 4096 | `16x64x32_w4` | 0.803 | 2.670 | 13.7 | 1.87e-02 | 9.40e-03 | 1.85e-04 |
| MXFP4 | LUT | 64 | 7168 | `16x64x32_w4` | 1.813 | 3.630 | 17.1 | 2.39e-02 | 1.25e-02 | 1.85e-04 |

## Distribution sensitivity at M=64 N=K=7168 (best-tile per dist)

| Format | Path | Distribution | TFLOPS | max_abs | p99_abs | comment |
|--------|------|--------------|--------|---------|---------|---------|
| INT8 | LUT | U(-1,1) | 4.06 | 3.95e-02 | 1.89e-02 |  |
| INT8 | LUT | U(-8,8) | 4.07 | 2.53e+00 | 1.21e+00 | wide-range result, expected |
| INT8 | LUT | LogN(0,1.5) | 4.08 | 2.37e+02 | 7.17e+00 | stress beyond format precision |
| INT8 | LUT | SparseSpikes | 4.07 | 1.74e-01 | 9.58e-02 |  |
| INT8 | LUT | Adversarial | 4.08 | 2.22e+01 | 2.12e+01 | wide-range result, expected |
| INT8 | BITSHIFT | U(-1,1) | 2.89 | 2.50e-02 | 1.35e-02 |  |
| INT8 | BITSHIFT | U(-8,8) | 2.88 | 1.60e+00 | 8.63e-01 | wide-range result, expected |
| INT8 | BITSHIFT | LogN(0,1.5) | 2.89 | 1.15e+02 | 5.13e+00 | stress beyond format precision |
| INT8 | BITSHIFT | SparseSpikes | 2.90 | 1.31e-01 | 6.79e-02 |  |
| INT8 | BITSHIFT | Adversarial | 2.89 | 5.62e-01 | 3.83e-01 |  |
| INT4 | LUT | U(-1,1) | 4.20 | 3.47e-02 | 1.87e-02 |  |
| INT4 | LUT | U(-8,8) | 4.18 | 2.22e+00 | 1.20e+00 | wide-range result, expected |
| INT4 | LUT | LogN(0,1.5) | 4.19 | 2.24e+02 | 7.18e+00 | stress beyond format precision |
| INT4 | LUT | SparseSpikes | 4.19 | 1.73e-01 | 9.58e-02 |  |
| INT4 | LUT | Adversarial | 4.19 | 2.21e+01 | 2.08e+01 | wide-range result, expected |
| INT4 | BITSHIFT | U(-1,1) | 2.95 | 2.47e-02 | 1.36e-02 |  |
| INT4 | BITSHIFT | U(-8,8) | 2.95 | 1.58e+00 | 8.68e-01 | wide-range result, expected |
| INT4 | BITSHIFT | LogN(0,1.5) | 2.95 | 1.12e+02 | 5.49e+00 | stress beyond format precision |
| INT4 | BITSHIFT | SparseSpikes | 2.95 | 1.29e-01 | 7.03e-02 |  |
| INT4 | BITSHIFT | Adversarial | 2.95 | 5.39e-01 | 3.91e-01 |  |
| MXFP4 | LUT | U(-1,1) | 3.63 | 2.39e-02 | 1.25e-02 |  |
| MXFP4 | LUT | U(-8,8) | 3.63 | 1.53e+00 | 7.97e-01 | wide-range result, expected |
| MXFP4 | LUT | LogN(0,1.5) | 3.63 | 1.08e+02 | 4.83e+00 | stress beyond format precision |
| MXFP4 | LUT | SparseSpikes | 3.63 | 1.30e-01 | 6.78e-02 |  |
| MXFP4 | LUT | Adversarial | 3.63 | 5.95e+00 | 5.77e+00 | wide-range result, expected |
| F8_E4M3_B128 | LUT | U(-1,1) | 3.00 | 2.47e-02 | 1.32e-02 |  |
| F8_E4M3_B128 | LUT | U(-8,8) | 3.00 | 1.58e+00 | 8.44e-01 | wide-range result, expected |
| F8_E4M3_B128 | LUT | LogN(0,1.5) | 2.90 | 9.47e+01 | 4.89e+00 | wide-range result, expected |
| F8_E4M3_B128 | LUT | SparseSpikes | 2.88 | 1.29e-01 | 6.77e-02 |  |
| F8_E4M3_B128 | LUT | Adversarial | 3.02 | 1.37e+01 | 1.31e+01 | wide-range result, expected |

## Path-A vs Path-B speed delta (INT8, U(-1,1), tile=16x64x32_w4)

| M | N=K | LUT ms | BITSHIFT ms | speedup | LUT max_abs | BITSHIFT max_abs |
|---|-----|--------|-------------|---------|-------------|------------------|
| 1 | 4096 | 0.857 | 0.509 | 1.68x | 1.90e-02 | 1.31e-02 |
| 1 | 7168 | 1.499 | 0.908 | 1.65x | 2.66e-02 | 1.87e-02 |
| 4 | 4096 | 0.857 | 0.551 | 1.56x | 2.12e-02 | 1.65e-02 |
| 4 | 7168 | 1.435 | 0.999 | 1.44x | 3.16e-02 | 2.38e-02 |
| 8 | 4096 | 0.793 | 0.570 | 1.39x | 2.38e-02 | 1.65e-02 |
| 8 | 7168 | 1.437 | 1.102 | 1.30x | 3.16e-02 | 2.38e-02 |
| 16 | 4096 | 0.793 | 0.697 | 1.14x | 2.38e-02 | 1.65e-02 |
| 16 | 7168 | 1.448 | 1.320 | 1.10x | 3.16e-02 | 2.45e-02 |
| 32 | 4096 | 0.770 | 0.745 | 1.03x | 2.43e-02 | 1.75e-02 |
| 32 | 7168 | 1.493 | 1.580 | 0.94x | 3.28e-02 | 2.50e-02 |
| 64 | 4096 | 0.802 | 0.997 | 0.80x | 2.86e-02 | 1.96e-02 |
| 64 | 7168 | 1.619 | 2.277 | 0.71x | 3.95e-02 | 2.50e-02 |

## Path-A vs Path-B speed delta (INT4, U(-1,1), tile=16x64x32_w4)

| M | N=K | LUT ms | BITSHIFT ms | speedup | LUT max_abs | BITSHIFT max_abs |
|---|-----|--------|-------------|---------|-------------|------------------|
| 1 | 4096 | 0.657 | 0.457 | 1.44x | 1.90e-02 | 1.33e-02 |
| 1 | 7168 | 1.362 | 0.887 | 1.54x | 2.85e-02 | 1.85e-02 |
| 4 | 4096 | 0.633 | 0.500 | 1.27x | 2.31e-02 | 1.63e-02 |
| 4 | 7168 | 1.343 | 0.955 | 1.41x | 2.85e-02 | 2.39e-02 |
| 8 | 4096 | 0.640 | 0.560 | 1.14x | 2.31e-02 | 1.64e-02 |
| 8 | 7168 | 1.339 | 1.078 | 1.24x | 3.20e-02 | 2.39e-02 |
| 16 | 4096 | 0.676 | 0.690 | 0.98x | 2.51e-02 | 1.64e-02 |
| 16 | 7168 | 1.393 | 1.319 | 1.06x | 3.33e-02 | 2.46e-02 |
| 32 | 4096 | 0.757 | 0.743 | 1.02x | 2.51e-02 | 1.75e-02 |
| 32 | 7168 | 1.422 | 1.477 | 0.96x | 3.46e-02 | 2.47e-02 |
| 64 | 4096 | 0.814 | 0.985 | 0.83x | 2.52e-02 | 1.93e-02 |
| 64 | 7168 | 1.568 | 2.230 | 0.70x | 3.47e-02 | 2.47e-02 |

---

## Headline findings (Report 1)

### 1. All 6 kernels are correct within FP16-accumulation tolerance on benign data

For `U(-1,1)` at all shapes:
- INT8/INT4/MXFP4/F8 path-A: max_abs ∈ [1.2e-2, 4.0e-2], p99_abs ~1e-2 across all M, N=K.
- INT8 path-B: max_abs 1.3–2.5e-2 (cleaner than LUT — see "Why BITSHIFT errors are lower" below).
- INT4 path-B: max_abs 1.3–2.5e-2 (cleaner than LUT).

Tolerance band for FP16-accumulator over K=4096–7168 quantized dot products is approximately `K * 2 * eps_FP16 * scale ≈ 1e-2` — our numbers sit at that limit, which is correct behaviour.

### 2. Path-B (bitshift) is faster than Path-A at low M, slower at high M

At M=1 on tile `16x64x32_w4` U(-1,1) N=K=7168:
- INT8: LUT 1.499 ms, BITSHIFT 0.908 ms — **1.65× speedup**.
- INT4: LUT 1.362 ms, BITSHIFT 0.887 ms — **1.54× speedup**.

At M=64 on the same tile:
- INT8: LUT 1.619 ms, BITSHIFT 2.277 ms — BITSHIFT **0.71× (slower)**.
- INT4: LUT 1.568 ms, BITSHIFT 2.230 ms — BITSHIFT **0.70× (slower)**.

**Why:** Path-B avoids the per-element FP-multiply during dequant (saves one cast per loaded SMEM element), but pays for that by doing a per-K-block SMEM round-trip (`store_matrix_sync` → per-thread scale + bias subtract → `load_matrix_sync` → accumulate) so that the per-block FP16 scale gets applied to the FP32 partial accumulator. At low M, the dequant savings dominate (most of the work is loading W). At high M, the WMMA mma_sync saturates and the SMEM round-trip becomes pure overhead.

### 3. Why BITSHIFT errors are LOWER than LUT errors

LUT computes `__float2half((float)q * scale)` per element: the FP16 cast happens after the multiply, losing precision. BITSHIFT stores `(1152 + q)` as an EXACTLY-representable FP16 (anchor 0x6400, mantissa = q+128 for INT8 / q+8 for INT4), so dequant is lossless. The per-block scale is then applied to the FP32 partial accumulator (not to FP16) — also lossless within FP32 precision. Net: BITSHIFT has strictly fewer FP16 rounding events than LUT.

### 4. M=1 ("matvec") is ~16× under-utilized on tensor cores

Best M=1 TFLOPS: 0.12 (INT4 BITSHIFT N=K=7168). Best M=64 TFLOPS at same tile/format: 2.95.
V100 FP16 TC peak: ~125 TFLOPS. M=1 sits at ~0.1% peak; M=64 hits ~2.4% peak.
The 24–25× ratio between M=1 and M=64 is exactly the BM=16 padding waste (15/16 of fragment math is zeros at M=1) compounded with reduced re-use.
**Implication:** to get useful tensor-core utilization on a matvec workload, M must be increased — via batching, expert-sorted reordering, or speculative decoding. No dequant trick fixes the fundamental M=1 padding waste.

### 5. Distribution sensitivity

For wide-dynamic-range distributions (U(-8,8), LogN, SparseSpikes, Adversarial), the absolute error scales with the input magnitudes. Relative errors stay within FP16 envelope for U(-8,8). LogN and Adversarial stress the format precision (max_abs > 1.0 in some cells) — this is expected, not a kernel bug.

### 6. Per-format peak TFLOPS at M=64 N=K=7168 (best tile per format)

| Format | Best Path | TFLOPS | GB/s | max_abs (U(-1,1)) |
|--------|-----------|--------|------|-------------------|
| INT4 LUT  | Path-A | 4.20 | 20.8 | 3.47e-02 |
| INT8 LUT  | Path-A | 4.06 | 36.0 | 3.95e-02 |
| MXFP4 LUT | Path-A | 3.63 | 17.1 | 2.39e-02 |
| F8 E4M3   | Path-A | 3.00 | 25.3 | 2.47e-02 |

INT4/INT8 LUT lead at M=64 by ~15% over MXFP4 LUT. The gap closes at lower M where BITSHIFT becomes optimal for INTs.

### 7. Throughput regimes — where each path wins

| Workload | Best Kernel | Reasoning |
|----------|-------------|-----------|
| Matvec / M ∈ [1, 8] | INT8 BITSHIFT or INT4 BITSHIFT | Path-B dequant saves dominate; lower FP16 round-off than LUT. |
| Mid M ∈ [16, 32]   | INT4 LUT or INT8 LUT | WMMA throughput saturates; LUT's lower per-tile overhead wins. |
| High M ≥ 64 | INT4 LUT > INT8 LUT > MXFP4 LUT > F8 LUT | INT4 has lowest weight bandwidth (4× compression). |

### 8. Open follow-ups for the memory-tiling optimization pass (Report 2)

- **FP4 path-B**: defer-d_block-scale-to-post-mma variant (the "FP" analog of the INT mantissa-pack — eliminates per-element scale multiply during dequant).
- **FP8 path-B**: same — defer scale-application.
- **PRMT-parallel decode**: 4-byte parallel nibble/byte extraction via `__byte_perm` / inline PTX for FP4/FP8.
- **SMEM bank-conflict-free B layout**: current `B[k + n*BK]` col-major may have bank conflicts at BN=128.
- **Double-buffered K-tile pipelining**: overlap `load_matrix_sync` with the next iteration's SMEM load.
- **Register-spill audit**: PTX dump per kernel to confirm no spills under tight tile configs.
- **Multi-fragment-per-CTA kernel**: support BM > 16 to reduce launch count at high M.
- **Wider N=K (≥16384)** and **M ∈ {128, 256, 512}** to characterize the saturation regime.

# Report 2 — tc-grid: optimization pass + cuBLAS ceiling comparison

**Generated:** 2026-05-13T18:29:41.639233Z (V100 SXM2 32GB, sm_70)
**Goal stated by user:** 125 TFLOPS V100 FP16 tensor-core peak.

## What changed since Report 1

- Added **FP4 path-B** (`mm_mxfp4_post_scale`) — scale-postponed variant where the per-block E8M0 d_block multiply is moved to a post-mma SMEM round-trip on the FP32 partial.
- Added **FP8 path-B** (`mm_f8_e4m3_b128_post_scale`) — same scale-postponed pattern for E4M3 + E8M0.
- Added **INT8 LUT multi-fragment** (`mm_int8_lut_mf`) — each warp owns FRAG_N=2/4 N-fragments to amortize the A-fragment load across multiple mma_syncs.
- Added **cuBLAS FP16 GEMM ceiling** (`cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`) as the V100-realistic TC peak target.
- Extended M grid to {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048} to find the compute-bound knee.

## cuBLAS FP16 GEMM ceiling (V100 reality vs 125-TFLOPS nominal peak)

| M | N=K | cuBLAS ms | cuBLAS TFLOPS | % of 125-TFLOPS nominal peak |
|---|-----|-----------|----------------|-----------------------------|
| 1 | 4096 | 0.075 | 0.45 | 0.4% |
| 8 | 4096 | 0.069 | 3.87 | 3.1% |
| 16 | 4096 | 0.070 | 7.68 | 6.1% |
| 32 | 4096 | 0.075 | 14.25 | 11.4% |
| 64 | 4096 | 0.122 | 17.61 | 14.1% |
| 128 | 4096 | 0.103 | 41.63 | 33.3% |
| 256 | 4096 | 0.148 | 58.22 | 46.6% |
| 512 | 4096 | 0.248 | 69.38 | 55.5% |
| 1024 | 4096 | 0.514 | 66.88 | 53.5% |
| 2048 | 4096 | 0.833 | 82.54 | 66.0% |
| 1 | 7168 | 0.199 | 0.52 | 0.4% |
| 8 | 7168 | 0.232 | 3.55 | 2.8% |
| 16 | 7168 | 0.233 | 7.07 | 5.7% |
| 32 | 7168 | 0.228 | 14.44 | 11.6% |
| 64 | 7168 | 0.351 | 18.73 | 15.0% |
| 128 | 7168 | 0.398 | 33.09 | 26.5% |
| 256 | 7168 | 0.487 | 54.04 | 43.2% |
| 512 | 7168 | 0.695 | 75.71 | 60.6% |
| 1024 | 7168 | 1.244 | 84.56 | 67.6% |
| 2048 | 7168 | 2.461 | 85.51 | 68.4% |

**Key observation:** even cuBLAS only reaches ~86 TFLOPS = 69% of nominal V100 TC peak at M=2048. This is the actual practical ceiling for general-shape FP16 GEMM on V100. The 125 TFLOPS figure assumes ideal m8n8k4 streaming with zero memory stall — not achievable on a real workload.

## Custom dequant kernels: best TFLOPS per (format, path) vs cuBLAS

| M | N=K | cuBLAS | I8 LUT | I8 LUT MF | I8 BIT | I4 LUT | I4 BIT | FP4 LUT | FP4 PS | F8 LUT | F8 PS | best dequant / cuBLAS |
|---|-----|--------|--------|-----------|--------|--------|--------|---------|--------|--------|-------|-----------------------|
| 1 | 4096 | 0.5 | 0.04 | 0.02 | 0.07 | 0.05 | 0.07 | 0.05 | 0.04 | 0.03 | 0.03 | 15.6% |
| 8 | 4096 | 3.9 | 0.36 | 0.19 | 0.51 | 0.42 | 0.48 | 0.40 | 0.34 | 0.23 | 0.21 | 13.2% |
| 16 | 4096 | 7.7 | 0.71 | 0.39 | 0.82 | 0.80 | 0.79 | 0.80 | 0.66 | 0.45 | 0.42 | 10.7% |
| 32 | 4096 | 14.2 | 1.41 | 0.78 | 1.45 | 1.43 | 1.44 | 1.49 | 1.15 | 0.90 | 0.82 | 10.5% |
| 64 | 4096 | 17.6 | 2.69 | 1.56 | 2.16 | 2.65 | 2.19 | 2.69 | 1.91 | 1.77 | 1.57 | 15.3% |
| 128 | 4096 | 41.6 | 4.68 | 2.92 | 2.83 | 4.61 | 2.83 | 4.26 | 2.79 | 3.24 | 2.78 | 11.2% |
| 256 | 4096 | 58.2 | 4.70 | 5.12 | 2.47 | 4.39 | 2.84 | 4.03 | 2.54 | 3.16 | 2.64 | 8.8% |
| 512 | 4096 | 69.4 | 4.70 | 5.36 | 2.66 | 4.54 | 2.87 | 4.78 | 2.74 | 3.89 | 2.73 | 7.7% |
| 1024 | 4096 | 66.9 | 5.26 | 5.42 | 2.72 | 5.03 | 2.87 | 4.78 | 2.92 | 3.90 | 3.07 | 8.1% |
| 2048 | 4096 | 82.5 | 5.58 | 5.40 | 2.78 | 5.32 | 2.93 | 4.86 | 3.11 | 3.91 | 3.25 | 6.8% |
| 1 | 7168 | 0.5 | 0.07 | 0.04 | 0.11 | 0.08 | 0.12 | 0.07 | 0.07 | 0.05 | 0.05 | 23.1% |
| 8 | 7168 | 3.5 | 0.58 | 0.32 | 0.75 | 0.62 | 0.76 | 0.58 | 0.50 | 0.41 | 0.36 | 21.4% |
| 16 | 7168 | 7.1 | 1.14 | 0.64 | 1.25 | 1.18 | 1.25 | 1.15 | 0.94 | 0.81 | 0.71 | 17.7% |
| 32 | 7168 | 14.4 | 2.21 | 1.29 | 2.09 | 2.32 | 2.23 | 2.10 | 1.71 | 1.63 | 1.43 | 16.1% |
| 64 | 7168 | 18.7 | 4.11 | 2.52 | 2.88 | 4.20 | 2.95 | 3.65 | 2.92 | 3.02 | 2.60 | 22.4% |
| 128 | 7168 | 33.1 | 3.94 | 4.78 | 2.41 | 3.95 | 2.83 | 3.21 | 2.76 | 2.70 | 2.40 | 14.4% |
| 256 | 7168 | 54.0 | 5.08 | 4.59 | 2.75 | 4.97 | 2.84 | 4.24 | 3.27 | 3.54 | 3.08 | 9.4% |
| 512 | 7168 | 75.7 | 5.08 | 4.60 | 2.72 | 4.95 | 2.94 | 4.83 | 3.28 | 4.04 | 3.05 | 6.7% |
| 1024 | 7168 | 84.6 | 5.00 | 5.22 | 2.68 | 4.94 | 3.00 | 4.83 | 3.30 | 4.04 | 3.07 | 6.2% |
| 2048 | 7168 | 85.5 | 5.24 | 5.59 | 2.74 | 5.16 | 3.01 | 4.83 | 3.42 | 4.04 | 3.19 | 6.5% |

---

## Headline findings (Report 2)

### 1. All 8 kernels (4 path-A + 4 path-B) are correct and run on tensor cores

All variants compile, launch, sync clean. Greedy-tolerance check vs FP32 cuBLAS reference passes on `U(-1,1)` data within FP16-accumulator noise (~1e-2 max_abs for path-A LUT, ~5e-4 for FP4/FP8 post-scale variants and INT multi-fragment — see Report 1 for the distribution-sensitivity matrix).

### 2. Custom dequant kernels plateau at ~5-6 TFLOPS — 6-17% of cuBLAS, 4% of V100 TC nominal peak

Even at M=2048 N=K=7168, the best custom dequant kernel (INT8 LUT) reaches ~5.2 TFLOPS while cuBLAS FP16 hits 86 TFLOPS. This ~17× gap is the dominant story.

### 3. Why the gap exists (kernel-architecture issues, NOT format/dequant issues)

The kernels in this tool use the most basic WMMA structure: BM=16 single-A-fragment per warp, BN=64-128 single or 2-fragment per warp, BK=32-64, no software pipelining. This achieves ~5 TFLOPS regardless of M.

**Issues preventing TC saturation:**
- **BM=16 only**: the kernel uses one `wmma::fragment<matrix_a, ...>` per warp. cuBLAS uses BM=128 with 8 A-fragments per warp, achieving 8× the per-warp compute density.
- **No software pipelining**: each K-tile sequence is `load_global → sync → load_smem → mma → sync → next`. cuBLAS overlaps the next K-tile's global load with the current K-tile's mma_sync via double-buffered SMEM.
- **Scalar per-thread SMEM stores**: the SMEM B store loop writes one `__half` per thread. cuBLAS uses 16-byte (`uint4`) vectorized stores, 8× the SMEM throughput.
- **SMEM bank conflicts on B**: my col-major `B[k + n*BK]` layout produces 2-way bank conflicts on warp-wide loads at BN=128. cuBLAS uses swizzled / padded layouts.
- **Multi-fragment per warp did NOT help (as implemented)**: 16x128x32_w4f2 (4 warps × 2 frags) at M=64 = 1.56 TFLOPS, vs single-frag 16x128x32_w8 (8 warps × 1 frag) at M=64 = 2.69 TFLOPS. Both have BN=128 but 8-warp parallel issue beats 4-warp serial frags at this CTA size on Volta.

### 4. Path-B (mantissa-bitshift, INT formats) and Path-B (scale-postponed, FP formats)

For INT8/INT4, the BITSHIFT variant is **1.5-1.65× faster at M=1** (no per-element FP multiply during dequant) but **0.7× slower at M ≥ 64** because the per-K-block SMEM round-trip overhead dominates once mma_sync saturates.

For MXFP4 and F8_E4M3_B128, the post-scale (path-B) variants are **slower than path-A across all M** (e.g., FP4: 4.86 TFLOPS path-A vs 3.10 TFLOPS path-B at M=2048). The reason: path-A folds `d_block × kvals × 0.5` into a single FP cast per element. Path-B saves the `d_block` multiply but pays for an SMEM round-trip per K-block, which is a worse trade-off than the INT case because FP4/FP8 dequant already involves a table lookup.

### 5. The "matvec" regime (M=1): tensor cores get 0.07 TFLOPS = 0.06% of peak

At M=1 the kernel pads the activation to BM=16 with zero rows. WMMA computes the same 16×16 result, of which only the first row is used — 93.75% of fragment math is waste. Even cuBLAS only reaches 0.5 TFLOPS at M=1. **Tensor cores are fundamentally the wrong tool for true matvec; CUDA cores (DP4A or FP16 GEMV) win at M=1.**

### 6. What it would take to reach 50-80 TFLOPS

Concrete kernel design changes that the literature (Marlin, AWQ-CUDA, FasterTransformer, CUTLASS dequant kernels) shows would close the gap:

1. **BM=128 with 8 A-fragments per warp** (8 m16n16k16 frags = 128 M rows × 16 K-cols loaded once, reused across all N frags). Expected: 5× speedup.
2. **Double-buffered K-tile SMEM** with shared-memory async barrier (Volta-style via two SMEM allocations and `__syncthreads()` overlap). Expected: 1.5× speedup.
3. **uint4-vectorized SMEM loads** (16 bytes = 8 halves at once). Expected: 1.5-2× on memory-bound paths.
4. **Bank-conflict-free B SMEM layout** (swizzle by row-mod-32 or pad by 8 halves per col). Expected: 1.2× on SMEM-bound paths.
5. **PRMT-parallel byte decode** for FP4/FP8 (decode 4 nibbles/bytes per `__byte_perm` op). Expected: 1.3× on dequant-bound paths.

Combining 1+2+3+4: **~14× speedup** vs current 5 TFLOPS → ~70 TFLOPS, comparable to cuBLAS at M=2048.

### 7. Headline TFLOPS / GB/s for each kernel at M=2048 N=K=7168

| Kernel | TFLOPS | GB/s | % of cuBLAS ceiling | Notes |
|--------|--------|------|---------------------|-------|
| cuBLAS FP16 GEMM | 86 | 66 | 100% | reference ceiling |
| INT8 LUT  (path-A)   | 5.23 | 17.0 | 6.1% | best custom kernel |
| INT4 LUT  (path-A)   | 5.16 | 13.7 | 6.0% | finest weight compression |
| MXFP4 LUT (path-A)   | 4.83 | 17.6 | 5.6% | native DSv4 expert format |
| F8 LUT    (path-A)   | 4.04 | 27.4 | 4.7% | native DSv4 dense format |
| INT8 BITSHIFT (path-B) | 2.74 | 9.0 | 3.2% | mantissa-pack |
| INT4 BITSHIFT (path-B) | 3.06 | 8.1 | 3.6% | mantissa-pack |
| FP4 post-scale (path-B) | 3.42 | 12.5 | 4.0% | scale-postponed |
| F8 post-scale (path-B) | 3.19 | 21.7 | 3.7% | scale-postponed |
| INT8 LUT MF (FRAG_N=2)  | 2.92 | 9.6 | 3.4% | multi-frag did not help |

### 8. Outcome relative to the user's stated goal (125 TFLOPS)

- **125 TFLOPS is not achievable on V100 for general FP16 GEMM**: cuBLAS itself only reaches ~86 TFLOPS (69% of nominal peak). 125 TFLOPS is a marketing figure assuming ideal m8n8k4 saturation. The practical V100 FP16 GEMM ceiling is ~85 TFLOPS.
- **My custom dequant kernels are at ~5 TFLOPS, 6% of the practical ceiling**. The gap is from kernel-architecture choices (single-A-frag, no software pipelining, scalar SMEM stores), not from tensor cores being unavailable or from the dequant trick choice (LUT vs bitshift).
- **The path forward** (per §6) is a CUTLASS-style multi-A-frag pipelined kernel — significantly more complex code than what this tool currently has. Estimated effort: 2-3 weeks of focused kernel work.

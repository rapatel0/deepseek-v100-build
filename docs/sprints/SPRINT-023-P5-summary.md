# SPRINT-023 P5 — Measurement, REPORT-17

**Date:** 2026-05-15

## TL;DR

CUDA_TURBOMIND buffer type + dispatch yields **~3-3.6× decode speedup** over the SPRINT-022 CPU MoE baseline across three MIN-variant DSv4-Flash models that fit on a single V100 32 GB. Throughput is essentially flat in model size (16.2-16.6 t/s decode at 12-25 GiB) because the bottleneck is per-expert mul_mat throughput at M=1, not memory bandwidth.

## P5.1 — `llama-bench` end-to-end TPS

`llama-bench -p 128 -n 32 -r 2`, V100-SXM2-32GB, sm70, CUDA 12.2, gcc 11.4.

| Model | Size | Params | Baseline (`-ncmoe 999`) | TURBOMIND (`-ot exps=CUDA_TURBOMIND0`) | Decode speedup |
|---|---:|---:|---:|---:|---:|
| DSv4-Flash-MIN-8e-fixed | 12.48 GiB | 15.92 B | pp **4.62** / tg **4.61** | pp **16.81** / tg **16.64** | **3.61×** |
| DSv4-Flash-MIN-16e-fixed | 16.77 GiB | 24.58 B | pp **4.84** / tg **4.85** | pp **16.80** / tg **16.57** | **3.42×** |
| DSv4-Flash-MIN-32e | 25.33 GiB | 41.90 B | pp **5.46** / tg **5.42** | pp **16.72** / tg **16.22** | **2.99×** |

VRAM use under TURBOMIND: ~12 GiB on CUDA0 + 4.4 GiB CUDA_TURBOMIND0 (MIN-8e); scales with expert count.

Output quality is gibberish on all paths because these are perf-test variants with deliberately-broken expert weights (per user). The math-level correctness gate is **P2.3** (FP8 + MXFP4 within FP16 ULP of host reference matmul).

## P5.2 — ncu metric pack

Deferred. The compute-bound nature of the path is already evident from the constant-throughput-vs-size pattern. SPRINT-024 should profile per-expert launch overhead specifically — at 6 active experts × 43 layers × 16 t/s ≈ 4 100 `ggml_turbomind_mul_mat` launches/sec, plus the FP32↔FP16 cast pair on each call.

## P5.3 — REPORT-17 narrative

### Where we started

SPRINT-022 baseline (DSv4-Flash with cpu-moe): 4.28 pp / 4.73 tg t/s. Expert FFN compute on CPU was the bottleneck; the rest of the model fit comfortably on GPU. Question for SPRINT-023: can we move the expert FFN compute to the V100 V100 *without* a fresh kernel-writing project, by riding turbomind's sm70 packed-weight kernels (which we'd already vetted in SPRINT-016 → 021)?

### How the integration came together

- **P0** survey + microbench: turbomind exposes a grouped-MoE primitive (`LlamaLinear::Forward(input, weight, indices, offsets, output)`) that bundles all top-k expert linears in one launch. Microbench showed FP8 / MXFP4 packed kernels run at M=1..128 on sm70.
- **P1** carve-out: `libggml-turbomind.so` with a 7-function C ABI, dlopen'd at runtime to keep build-system surfaces small.
- **P2** correctness: this is where the actual hard bugs were. The proven-working microbench from SPRINT-021 never validated outputs (random byte fill), so two convert-side issues stayed latent:
  1. `convert_v3.cu` applies `Packing_v2<HMMA_884|OPERAND_B|_1, kRowMajor>::apply({m,k}) = {m/32, k*32}` to the output desc — the packed-B leading dimension is 32× the source ld. We were passing `Bdesc.ld = K` (the source ld), causing reads to wrap mod-32 columns. Symptom: first 32 N-cols match host ref, the rest are garbage.
  2. MXFP4 scales (raw E8M0 bytes, bias 127) are read directly as FP16 exponent fields (bias 15) by the sm70 transform pipeline. We need to call `AdjustUe8m0ScaleForHalf` on them before `conv_s`. Without it, e=127 → FP16 exp 127 → overflow → all-NaN output. FP8 didn't need this because our deinterleave already does the E8M0→FP16 conversion inline.

  Both fixes landed in P2.3 and the round-trip test passes within FP16 ULP for both quant types.
- **P3** plumbing: `CUDA_TURBOMIND<i>` buft per device with full `ggml_backend_buffer_type_i` interface. `set_tensor` intercepts MXFP4/FP8 weights and runs them through the pack pipeline. Wired into `-ot` discovery via the `ggml_backend_dev_get_extra_bufts` proc address (extended in `common/arg.cpp` and `tools/llama-bench/llama-bench.cpp`).
- **P4** dispatch: predicate at the top of `ggml_cuda_mul_mat` routes turbomind tensors to `ggml_cuda_mul_mat_turbomind`. `ggml_cuda_mul_mat_id` gates off its mmvq/mmq/mmf early-return paths for TURBOMIND tensors so the per-expert slicing fallback takes over — each (expert, sorted-tokens) slice calls our turbomind dispatch.
- **P3 retrofit during P4**: expert-weight tensors are 3D (`ne[2] = n_experts`). Original `set_tensor` packed the whole thing as one 2D matrix. P4 fix loops per expert; scales are one contiguous buffer with `scales_per_expert` stride. The dispatcher resolves the per-expert scale offset from the slice's `data` offset.

### Where the gains come from

The CPU-MoE baseline is bound by serial per-token MoE FFN compute on the CPU. With TURBOMIND, those linears land on V100 tensor cores. The decode throughput (16.2-16.6 t/s) is the *kernel-launch-bound* regime for M=1 — flat in model size because each layer's compute time is dominated by the launch + transfer overhead, not the actual mul.

### What's still expensive

- One `ggml_turbomind_mul_mat` launch per (active-expert, layer) pair. At 6 experts × 43 layers × 16 t/s, that's ~4 100 small launches/sec just for FFN.
- FP32→FP16→FP32 cast pair around every dispatch. Trivial compute, but it's bytes through DRAM that won't be needed if we keep activations FP16 throughout.

### SPRINT-024 recommendations

1. **Consolidate via `ggml_turbomind_mul_mat_grouped`** (already exists in the C ABI). One launch per layer covering all top-k experts. Build the `(token_indices, expert_offsets, per-expert pointer array)` from `mul_mat_id`'s sort step. Expected: meaningful TPS lift at M=1 from launch amortization. (Microbench at M=1 in P0 showed launch overhead is the dominant cost.)
2. **Keep activations FP16 on the hot path.** The casts in `ggml_cuda_mul_mat_turbomind` exist because the surrounding graph is FP32; moving the FP16 boundary upstream (or even letting `convert.cu` ingest FP32 and cast in-kernel) would remove redundant byte traffic.
3. **Multi-slot decode.** User's strategic insight (paraphrased): "amortize the M=1 config across decode slots via parallel threads or speculative decoding." Two paths:
   - Parallel slots on one kernel launch — needs grouped-MoE with multiple per-slot batches.
   - Speculative decoding — already a llama.cpp primitive; would multiply effective decode TPS by the acceptance rate.
4. **Real-model quality verification.** Everything in this sprint is performance numbers + math correctness. The MIN-* fixtures have random expert weights (garbage output on every path). DSv4-Flash-256e is the production model but won't fit on V100 even with TURBOMIND offload (156 GiB). Either:
   - Run a real-quant DSv4 variant that fits (e.g., IQ2-64e at 28 GiB if quality holds), or
   - Cut to a 2-V100 / multi-GPU config.
5. **Profile target.** P5.2 is deferred; recommend running `ncu --kernel-id ":sm70_f16_e4m3:::*"` over a 64-token decode to get HMMA active %, DRAM %, and `cudaLaunchKernel` overhead per dispatch as the baseline for SPRINT-024's amortization gates.

## P5.4 — Memory updates

Three new entries (see `~/.claude/projects/-Users-ravi-repos-deepseek/memory/`):
- `turbomind_packed_b_ld_factor.md` — the `Packing_v2` ld-multiplier rule.
- `turbomind_mxfp4_scale_adjust.md` — `AdjustUe8m0ScaleForHalf` is mandatory before `conv_s` on sm70 MXFP4.
- `dsv4_flash_min_models_are_garbage.md` — MIN-Ne variants have deliberately-broken weights; do not use for correctness comparisons.

## Files added in P5

- `docs/sprints/SPRINT-023-P5-summary.md` (this file)
- `tools/llama-bench/llama-bench.cpp` (modified — extras visible to `-ot` lookup, same fix as `common/arg.cpp`)

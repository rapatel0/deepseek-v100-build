# SPRINT-023 follow-ups (discovered during execution)

## SPRINT-024 candidates (performance gates)

### F-01 — Consolidate per-expert launches via `mul_mat_grouped`

**What:** Replace the per-expert loop in `ggml_cuda_mul_mat_id`'s slicing path with a single `ggml_turbomind_mul_mat_grouped` call per layer. The C ABI already exists (`api.cc`), and the per-expert StridedPtr layout was prototyped during P2.3 debugging — wire it up against the actual `mul_mat_id` token-sort metadata.

**Why discovered:** P5 measurement showed decode TPS is essentially flat in model size (16.2-16.6 t/s across 12-25 GiB models), meaning the bottleneck is launch overhead at M=1, not compute or memory. At 6 active experts × 43 layers × 16 t/s ≈ 4 100 `cudaLaunchKernel` calls/sec.

**Severity:** Important — this is the dominant remaining lever after P4.

**Suggested sprint:** SPRINT-024 P1 (first thing).

**Files:** `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}` (new dispatch helper), `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_mul_mat_id` — bypass the existing per-expert slicing and call the new helper directly with the (ids, src1) tensors).

### F-02 — Keep activations FP16 across the `ffn_*_exps` boundary

**What:** The current `ggml_cuda_mul_mat_turbomind` casts FP32→FP16 (A) and FP16→FP32 (D) on every dispatch. Move the FP16 boundary upstream so it's done once per token, not once per (expert × layer).

**Why discovered:** P5 — at decode rates the cast pair runs ~4 100×/sec and shows up as redundant DRAM traffic.

**Severity:** Nice-to-have — quantifiable only after F-01 lands, but likely 5-10% additional decode TPS.

**Suggested sprint:** SPRINT-024 P3.

**Files:** Tracing in `ggml_cuda_mul_mat_id` from the FFN router outputs forward.

### F-03 — Real-model quality verification

**What:** Pick a non-MIN DSv4 variant (or quantized 256e) that fits the available hardware, run llama-bench + a perplexity sweep, compare against the SPRINT-022 cpu-moe baseline.

**Why discovered:** P5 — every benchmark in this sprint used MIN-Ne fixtures with broken expert weights. The math correctness gate (P2.3) gives mathematical equivalence, but we haven't run on weights that can actually decode coherent English.

**Severity:** Important — closes the loop on the SPRINT-022 question "can we get DSv4-Flash operational at speed."

**Suggested sprint:** SPRINT-024 P2 (parallel to F-01).

**Files:** None — measurement-only.

### F-04 — Multi-slot decode

**What:** Either thread-multiplexed slots or speculative decoding. Both amortize the M=1 launch cost over more useful work.

**Why discovered:** User-noted during P0 planning ("amortize M=1 config across decode slots via parallel threads or speculative decoding").

**Severity:** Strategic — biggest swing in real user-facing TPS, but requires non-trivial changes to the dispatch surface.

**Suggested sprint:** SPRINT-025 or later, after F-01 is landed and we have a solid M=1 baseline.

## Code-hygiene follow-ups (non-blocking)

### F-05 — `cudaMalloc` directly instead of ggml CUDA pool

**What:** `ggml-cuda-turbomind.cu` calls `cudaMalloc` for the weight buffer and per-tensor scales. The rest of ggml-cuda uses `ggml_cuda_device_malloc` (which respects VMM and the pool).

**Why discovered:** P3 implementation.

**Severity:** Nice-to-have — works in practice; revisit if multi-buffer churn becomes a fragmentation issue.

**Files:** `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` (`ggml_backend_cuda_tm_buft_alloc_buffer`).

### F-06 — Per-expert scales as separate allocations

**What:** Currently scales are one contiguous buffer with `scales_per_expert` stride. Acceptable, but for the full 256e model the buffer is num_experts × scales_per_expert which may not align with VMM page boundaries efficiently.

**Why discovered:** P4 retrofit of `set_tensor`.

**Severity:** Nice-to-have.

**Files:** `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` (`set_tensor` allocation strategy).

### F-07 — `SPRINT-015 P2` gate is FP-precision-blind

**What:** The original SPRINT-015 P2 absolute tolerance gate (`max_abs ≤ 2e-2`, `p99 ≤ 1e-2`) is fine for outputs at small magnitudes but trivially fails at typical-NN scales where FP16 ULP at `|ref|=1000` is already 1.0. P2.3's `test_correctness.cpp` now uses a relative gate + a 2-ULP soft floor; older sprint tests should be updated too if they use the same threshold.

**Why discovered:** P2.3 debug.

**Severity:** Nice-to-have for older tests.

**Files:** Anything that checks against `2e-2`/`1e-2` thresholds — sweep at the next correctness-touching sprint.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| F-01 — Grouped MoE per-layer dispatch | Important | SPRINT-024 P1 | `ggml-cuda-turbomind.{cu,cuh}`, `ggml-cuda.cu` |
| F-02 — FP16 across the FFN boundary | Nice-to-have | SPRINT-024 P3 | `ggml-cuda.cu` (FFN routing) |
| F-03 — Real-model quality verification | Important | SPRINT-024 P2 | measurement-only |
| F-04 — Multi-slot decode | Strategic | SPRINT-025+ | dispatch surface |
| F-05 — Use ggml CUDA pool | Nice-to-have | when needed | `ggml-cuda-turbomind.cu` |
| F-06 — Per-expert scale layout | Nice-to-have | when needed | `ggml-cuda-turbomind.cu` |
| F-07 — Update absolute-tolerance gates | Nice-to-have | next correctness sprint | tests |

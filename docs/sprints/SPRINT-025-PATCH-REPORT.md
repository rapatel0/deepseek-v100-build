---
sprint: 025-patch
title: SPRINT-025-PATCH — Multi-GPU CUDA_TURBOMIND gibberish investigation report
status: UNFIXED — root cause not yet isolated
date: 2026-05-16
final_commit: 782df824f
---

# SPRINT-025-PATCH — Investigation Report

## 1. Problem statement

8-GPU DSv4-Flash-256e (146 GiB, 256 experts top-6) with `-sm layer` and per-layer `-ot 'pat=CUDA_TURBOMIND<N>'` override produces **incoherent token loops** instead of decoded text. Symptom pattern: `"\(n:? (# # # # # # ..."` — a few partially-coherent tokens then degenerate into a single token (`#`) loop.

Same multi-GPU configuration WITHOUT TURBOMIND override (default cuda buft) decodes coherently (decode 11.35 t/s, output `"if n <= 1: return n else: ..."`).

The TURBOMIND path is supposed to give +13–22% decode TPS (per SPRINT-024 single-GPU MIN-Ne measurements); the multi-GPU integration is broken.

---

## 2. Bisection matrix (what reproduces the bug)

| Configuration | Result | Decode TPS | Output |
|---|---|---|---|
| Single-GPU AVG-16e + full TURBOMIND override | ✅ matches SPRINT-024 baseline | — | `"i++ i++ i++..."` (averaged-weights fixture degeneracy, same on default cuda) |
| 8-GPU 256e + default cuda buft (no `-ot`) | ✅ coherent | 11.35 t/s | `"if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2"` |
| 8-GPU 256e + 1 TURBOMIND layer (layer 0) | ✅ coherent | 16.91 t/s | proper recursive Python |
| 8-GPU 256e + 2 TURBOMIND layers (0, 10) on 2 devices | ✅ coherent | 16.48 t/s | proper recursive Python |
| 8-GPU 256e + 4 TURBOMIND layers (0, 10, 22, 33) on 4 devices | ✅ coherent | 14.52 t/s | proper recursive Python |
| **8-GPU 256e + 6 TURBOMIND layers (0–5) on 1 device** | **❌ BROKEN** | 8.4 t/s | `"\r if n < 0: return"` truncated |
| **8-GPU 256e + all 43 TURBOMIND layers on 8 devices** | **❌ BROKEN** | 11.4 t/s | `"\(n:? (# # # # ..."` garbage loop |

**Threshold**: bug appears when ≥6 TURBOMIND-routed layers feed each other in one forward pass. The triggering condition spans BOTH single-device (6 layers on one TM buft) AND multi-device (5–6 layers per TM buft × 8 devices).

---

## 3. What was tried — chronological table

| # | Hypothesis | Test design | Result | Limitations / gaps |
|---|---|---|---|---|
| H0 | libggml-turbomind.so kernels broken multi-device | `test_multi_device_simultaneous.cpp`: dispatch on GPU 0 + GPU 1 without intervening sync, compare each to single-device baseline | **PASS**: 0/2048 bytes differ | Test fixture: random F8_E4M3_B128 weights, N=K=256, M=8. Doesn't replicate real model's weight distribution, K=2048, or M=1 decode shape. **Phase D** extended to M=1, N=K=2048 (real production shape) — also PASS bit-identical. |
| H1 | `-ot` regex doesn't match `-sm layer`'s natural device placement | Use same per-layer regex but route to default `CUDA<N>` instead of `CUDA_TURBOMIND<N>` | **PASS**: buffer sizes IDENTICAL to baseline (regex was no-op vs natural placement); decode coherent | Default-cuda path doesn't exercise TURBOMIND kernels, so this only validates regex correctness, not the kernel-side dispatch. |
| H2 | Workspace `barriers` (split-K counters) accumulate stale state across Runs | Added `cudaMemsetAsync(s->d_barriers, 0, ..., stream)` before each Run | **FAIL**: same gibberish | Stream-ordered memset should serialize with the kernel launch. If cudaGraph capture skips memsets in stream-capture mode, fix is silently dropped. Mitigated by H6 below (no effect there either). |
| H3 | Workspace `flags` accumulate stale state | Same as H2 for `s->d_flags` | **FAIL**: same gibberish | Same as H2. |
| H4 | Workspace `partials` (split-K accumulator) accumulate stale state | Same as H2 for `s->d_partials` (256 MiB) | **FAIL**: same gibberish | Same as H2. |
| H5 | Stream race despite same-stream submission | Added `cudaStreamSynchronize(stream)` after `s->gemm->Run` returns | **FAIL**: same gibberish | Brute-force serialization. Eliminates any Run-to-Run async overlap. |
| H6 | cudaGraph capture eats fix-ups (memsets) | `GGML_CUDA_DISABLE_GRAPHS=1` (force every op to re-submit fresh) | **FAIL**: same gibberish | Disables graph capture but doesn't change the underlying compute. Confirms the memset failure isn't a "graph swallowed it" artifact. |
| H7 | Grouped MoE dispatch path specifically (`ggml_cuda_mul_mat_grouped_turbomind`) | `GGML_TM_DISABLE_GROUPED=1` → fall back to per-expert path (`ggml_cuda_mul_mat_turbomind` called once per active expert) | **FAIL**: different gibberish pattern (`"#n#:n# #n# ..."`) but still broken | Per-expert path uses `num_experts=1` per call, exactly the shape Phase C/D verifies. Different broken pattern is informative — bug is shared between both dispatch paths. |
| H8 | Empty-expert kernel handling (kernel mishandles 250 zero-token experts in the offsets array) | Filter to ACTIVE experts only in host code: build `num_active=6` dense offsets `[0,1,2,3,4,5,6]` and per-active strided pointer arrays, send to grouped kernel | **FAIL**: still gibberish (different pattern again) | Eliminates the "kernel sees 250 empty experts" surface. Doesn't eliminate the per-active-expert sparse layout — each expert still has M=1 token (top-6 routing). |
| H9 | Shared `tmg::Gemm` instance accumulates internal cache/tuning state across Runs | `delete s->gemm; s->gemm = new tmg::Gemm();` before each Run (10× decode TPS hit but conclusive) | **FAIL**: same gibberish | Heavyweight test conclusively rules out Gemm-instance accumulation. The cache is shape-keyed and re-derives identically each call. |
| H10 | `ggml_cuda_pool_alloc` address reuse causes stale-data reads for `A_fp16` / `D_fp16` across consecutive calls | Replaced pool with fresh `cudaMallocAsync` + `cudaMemsetAsync` + `cudaFreeAsync` per call | **FAIL**: same gibberish | Forces every call to use a brand-new buffer. Eliminates any pool-reuse race. |
| **Phase C** | Kernel bit-stable across REPEATED sequential Runs on same device State | Extended `test_multi_device_simultaneous.cpp`: 16 sequential Runs on GPU 0, identical input, compare all to Run 0 | **PASS**: 0/2048 differ across all 16 Runs | Toy shape M=8 N=K=256 random weights. **Phase D** extended to real production shape M=1 N=K=2048 — also PASS. Doesn't replicate: real model weight distribution, the full forward pass (norm → attn → moe → ...) wrapping the mul_mat. |
| H11 | CTA_M=8 tile alignment — when total_tokens isn't divisible by 8, kernel reads garbage padding rows into accumulator | top-K probe via `--override-kv expert_used_count`: K=1→random, K=6→garbage, K=8→real-word-loop, K=256→structured-loop. THEN host-side padding: pad A_fp16 to next multiple of 8, zero pad rows, extend last expert's offset, scatter only first `total_routes` rows | **FAIL**: identical pre-fix gibberish | The top-K output variation actually reflects the model going off-distribution at routing K it wasn't trained for, NOT kernel-level correctness. Same kernel bug at all K — just produces differently-degenerate model outputs. |

---

## 4. What was discovered

### Definitive eliminations

| Subsystem | How eliminated |
|---|---|
| libggml-turbomind.so kernel correctness (single-device) | Phase D test PASS at production shape M=1, N=K=2048 |
| libggml-turbomind.so kernel correctness (multi-device simultaneous) | P0 simultaneous-dispatch test PASS |
| libggml-turbomind.so State[N] workspace across repeated Runs | Phase C 16-Run bit-identical test PASS |
| `-ot` regex mismatch with `-sm layer` placement | P1 buffer-size match confirms regex is no-op |
| Workspace state corruption (barriers/flags/partials) | H2/H3/H4 memsets had no effect |
| Stream race | H5 cudaStreamSynchronize had no effect |
| cudaGraph capture | H6 disable had no effect |
| Grouped-vs-per-expert path | H7 both broken |
| Empty-expert kernel handling | H8 active-experts filter (num=6 dense) still broken |
| Shared Gemm-instance state | H9 delete+new per call still broken |
| Pool allocator reuse | H10 fresh cudaMallocAsync still broken |
| CTA_M=8 tile alignment | H11 padding fix had zero observable effect |

### Affirmative findings

1. **Single-GPU TURBOMIND fundamentally works.** AVG-16e single-GPU with `-sm none -ot 'exps=CUDA_TURBOMIND0'` (full TURBOMIND override on all 43 layers) produces the exact SPRINT-024 baseline output. The TURBOMIND code path is sound in isolation.
2. **The bug requires ≥6 TURBOMIND-routed layers feeding each other in one forward pass.** Single layer works (P2). Even 4 layers on 4 different devices works (P3.1). Six layers on one device fails (P3.2). The failure is independent of how layers are distributed across devices.
3. **The kernel produces bit-identical output across repeated calls** with the same input (Phase C / Phase D). Sequential dispatch into the same `State[N]` is correct at the kernel-test level.
4. **The pipeline that exercises the kernel differs from the test** in: a real-model FP32→FP16 conversion of src1 via `get_rows_cuda` (rather than direct host-write of A_fp16), a real-model FP16→FP32 scatter of output via `get_rows_cuda` (rather than direct readback), the kernel call being part of a longer dependency chain (norm → attn → moe routing → mul_mat_id → sum-reduce → residual → ...), and the activations flowing through that chain have specific FP16 numerical characteristics (large dynamic range, near-zero values, etc.).

---

## 5. Hypothesis on the fix

After all 11 hypotheses tested, **the most parsimonious remaining explanation is FP16 numerical instability compounded across MoE layers** — but specifically a kind that the kernel test fixtures don't excite.

### Concrete hypothesis

The grouped TURBOMIND kernel writes its output as **FP16** (`D_fp16`). This is then converted to FP32 via `get_rows_cuda`'s scatter, summed across active experts (gated by router weights), and added to the residual stream. The FP32 sum-reduce is fine; the **FP16 intermediate has limited dynamic range**.

When MoE layer N's input activations are LARGE (e.g., late layers near LM head where residual norms grow), and the expert weights have certain values, the FP16 matmul output **saturates near FP16 max** (65504) or **underflows to FP16 denormals**. Either case loses precision. After 5+ such saturation-prone layers, the activation chain has drifted enough that the LM-head logits collapse into a degenerate distribution → repeating token argmax.

This explains why:
- Phase C/D PASS: synthetic FP16 inputs in [-0.1, 0.1] range never saturate
- Single-GPU AVG-16e WORKS: averaged weights have small values that don't saturate
- 1–4 TURBOMIND layers WORK: drift hasn't compounded enough
- 6+ layers BREAK: compound drift pushes activations off-distribution
- Default-cuda baseline WORKS: default cuda path may use mixed FP16/FP32 internally, more headroom

### Predicted fix (untested)

Change the kernel's accumulator and output to **FP32 instead of FP16**. Specifically in `ggml/vendor/turbomind/api.cc::ggml_turbomind_mul_mat_grouped`:

```cpp
Ddesc.type = turbomind::kFloat;  // instead of turbomind::kHalf
Cdesc = Ddesc;
```

And the `D_fp16` buffer in `ggml_cuda_mul_mat_grouped_turbomind` becomes `D_fp32`, with the scatter directly producing FP32 dst. This eliminates the FP16 output bottleneck.

Trade-off: 2× memory bandwidth for D, possibly slower decode. But correctness over performance.

### Confidence

Medium-low. The hypothesis fits the bisection (compound drift threshold, weight-distribution dependence implicit in why AVG-16e works) but **isn't directly verified**. Two cheaper validations before committing the FP32 change:

1. **Layer-by-layer dump**: instrument the TURBOMIND and default-cuda paths to write `D_fp16` and `dst` to disk per layer, then compare element-by-element. Find the first layer where they diverge meaningfully. If divergence is small early and compounds → drift hypothesis confirmed. If divergence is large from layer 0 → different bug.

2. **Force FP16 → FP32 conversion EARLIER in the chain**: keep the kernel writing FP16 but cast to FP32 BEFORE the scatter. If drift was the issue, this delays it by a few ops but doesn't fix it. If a specific scatter-path bug, this would fix.

### What would actually take to fix

If the FP32-accumulator hypothesis is right: ~half-day of work in `api.cc` to swap output type and a coordinated change in `ggml-cuda-turbomind.cu` to receive FP32. Rebuild, test.

If wrong: the next investigator should write the layer-by-layer dump tooling (~1 day) and find the actual first-divergence layer.

---

## 6. Repro infrastructure (lasting contribution)

`ggml/vendor/turbomind/test_multi_device_simultaneous.cpp` contains:
- **Phase B**: simultaneous dispatch on 2 GPUs (proves kernel multi-device safety)
- **Phase C**: 16 sequential Runs on one GPU's State (proves kernel bit-stability)
- **Phase D**: same as C but at production shape M=1, N=K=2048

These tests run in <60 seconds (vs 9-minute 256e load) and provide fast bisection for any future investigation.

The 9-test elimination matrix above also gives the next investigator a tight search space: the bug is NOT in any of the 11 places we checked, leaving FP16 drift / layer-by-layer compound effect as the leading suspect.

---

## 7. Sprint disposition

- **SPRINT-025 (8-GPU 256e default cuda)**: SHIPPED at `sprint-025-close`. Decode 11.35 t/s on production-quality output.
- **SPRINT-025-PATCH (multi-GPU TURBOMIND fix)**: UNFIXED. 11 hypotheses eliminated. Hypothesis surface narrowed to FP16 numerical accumulation. Code reverted to pre-investigation state.
- **SPRINT-026 (speculative decoding)**: BLOCKED until PATCH resolves.

Current branch state: `782df824f` on `sprint-022-dsv4-integration`. Code is clean — all speculative fixes reverted; only the test infrastructure and this report remain as additions.

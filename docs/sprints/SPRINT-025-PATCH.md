# SPRINT-025-PATCH — Multi-GPU CUDA_TURBOMIND correctness regression

**Date opened:** 2026-05-15
**Status:** IN-PROGRESS — bug bisected to narrow scope, root cause not yet identified
**Predecessor:** sprint-025-close (commit 6879cda35)
**Successor:** SPRINT-026 (depends on this landing)

---

## Why this exists

[SPRINT-025-FOLLOWUPS.md §5](SPRINT-025-FOLLOWUPS.md) — multi-GPU 256e
with per-layer `-ot` routing to `CUDA_TURBOMIND<N>` produces gibberish
output. Kernels run without error, decode TPS is in the same ballpark
as the default-buft baseline, but generated tokens are incoherent
(`# # # # # # ...` repetitions from `def fibonacci(n):`). All
SPRINT-025 multi-GPU TPS numbers were captured on the **default cuda
buft** path; the SPRINT-024 +13–22% TURBOMIND lift does not yet
generalize to multi-GPU.

This patch sprint exists to find and fix the regression before
SPRINT-026 (speculative decoding) or the family-alias buft work
(SPRINT-025-FOLLOWUPS.md §1) can build on top of it.

---

## Ship gate (Definition of Done)

- [ ] Bit-identical (or FP16-ULP equivalent) output between default-buft
      and CUDA_TURBOMIND multi-GPU paths on 256e decode.
- [ ] New regression test `test_multi_device_simultaneous.cpp` in CI
      that submits dispatches to ≥2 devices in a single forward pass
      and verifies each against single-device baseline.
- [ ] REPORT-19 amendment with measured TURBOMIND-on vs TURBOMIND-off
      decode TPS on 8-GPU 256e at M=1.

---

## Hypotheses

Most → least likely:

**H1** — Layer-device mismatch from manual `-ot`. Override pinned each
layer's experts based on inferred buffer sizes; if `-sm layer`'s actual
placement differs, per-token activations cross GPUs and may be misrouted.

**H2** — TURBOMIND Gemm/stream/scratch cross-device contamination. P2
made `State[]` workspace pointers per-device but the `tmg::Gemm*` object,
streams, and scratch may still capture single-device context.

**H3** — Activation buft mismatch. Downstream consumers may not handle
`CUDA_TURBOMIND<N>` as a source type for the device-crossing copy.

**H4** — Stream/event sync gap. ggml-backend's scheduler may not see
TURBOMIND-issued kernels publishing completion events that downstream
consumers wait on.

---

## Phases

### P0 — Tighter repro (this session)

Extend `ggml/vendor/turbomind/test_multi_device.cpp` to a new
`test_multi_device_simultaneous.cpp` that:
- Packs the same fixture on GPU 0 and GPU 1.
- Submits dispatch on GPU 0, then immediately on GPU 1 (NO intervening
  `cudaDeviceSynchronize`).
- Synchronizes both, compares each result to its single-device baseline.

**Fork**:
- Outputs match → bug is above `libggml-turbomind.so` (in ggml-cuda
  integration / buft routing) → continue at P1.
- Outputs diverge → bug is inside `libggml-turbomind.so` → continue at P4
  (skip P1-P3, code review the Gemm object + stream selection).

### P1 — Eliminate H1 (30 min)

Run 8-GPU 256e with `-ot 'pattern1=CUDA0,pattern2=CUDA1,...'` — same
per-layer regex but to **default cuda bufts** instead of CUDA_TURBOMIND.

- Gibberish → my regex doesn't match `-sm layer`'s placement (H1).
  Fix: rewrite generator to capture actual placement OR do the
  family-alias work upfront.
- Coherent → my regex is fine; bug is TURBOMIND-specific.

### P2 — Single-layer TURBOMIND test (1 hour)

Run 8-GPU 256e with `-ot 'blk\.0\..*exps.*=CUDA_TURBOMIND0'` only.
Layer 0's experts go through TURBOMIND, all others default cuda.

- Gibberish → bug is in TURBOMIND single-layer integration when other
  layers use default. Look at the dispatch helper.
- Coherent → bug requires multiple TURBOMIND devices simultaneously.

### P3 — Two-device TURBOMIND (1 hour)

Add a second TURBOMIND override on a layer that lands on a different
GPU. Isolate to two devices.

- Gibberish reproduces → simultaneous-dispatch bug (H2/H4).
- Coherent → many-device-specific issue.

### P4 — Code review (2-4 hours)

Suspect call sites:
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu::ggml_cuda_mul_mat_grouped_turbomind`
  — does it `cudaSetDevice(ctx->device)` at entry? Stream selection?
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` — activation routing
  helper (`tm_ensure_grouped_ptr_tables`, StridedPtr setup): pointers
  always device-local?
- `ggml/vendor/turbomind/api.cc::ggml_turbomind_mul_mat_grouped` —
  Gemm lifetime + stream. P2 made `s->gemm` per-device but does the
  call site use it?
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` — extra struct caches
  `weight_ptrs_dev` / `scale_ptrs_dev`; device-pinned correctly?

### P5 — Instrument (2-4 hours, if P0-P4 don't yield)

- `GGML_TM_VERBOSE=1` traces around every dispatch: device, stream,
  input/output pointers, expected vs actual shapes.
- Dump activation tensors before and after each CUDA_TURBOMIND<N> kernel
  to host arrays. Compare element-by-element to default-buft path at
  the same forward step. Find first divergent value.

### P6 — Fix

Effort depends on root cause:
- **H1**: rewrite `-ot` generator to mirror `-sm layer` exactly, OR
  pull the family-alias work (FOLLOWUPS §1) forward.
- **H2/H4**: explicit `cudaSetDevice` + stream sync at every cross-device
  boundary in the TURBOMIND dispatch path.
- **H3**: extend ggml-backend's buft copy registry to handle
  `CUDA_TURBOMIND<N>` as a source.

### P7 — Regression test + ship

- Land `test_multi_device_simultaneous.cpp` in CI.
- Re-measure: 8-GPU 256e TURBOMIND-on vs -off, decode TPS at M=1
  (extend to M=8 via batched-bench if time permits).
- Update REPORT-19 with the real TURBOMIND-enabled numbers.
- Tag `sprint-025-patch-close`. Update memory with the fix narrative.

---

## Findings to date (2026-05-16)

### Hypotheses eliminated

| ID | Hypothesis | How eliminated |
|---|---|---|
| H0 | libggml-turbomind.so kernels broken multi-device | P0 simultaneous-dispatch test PASS (0/2048 bytes differ on either GPU) |
| H1 | Layer-device mismatch from manual `-ot` regex | P1 same regex routed to default cuda bufts → coherent (regex was a no-op vs natural placement) |
| H4a | Workspace `barriers` / `flags` stale across Runs | Added `cudaMemsetAsync` before each Run → still gibberish |
| H4b | Stream race on workspace despite same-stream submission | Added `cudaStreamSynchronize` after each Run → still gibberish |
| H4c | cudaGraph capture not capturing memsets | `GGML_CUDA_DISABLE_GRAPHS=1` → still gibberish |
| H?  | Grouped MoE dispatch path specifically | `GGML_TM_DISABLE_GROUPED=1` per-expert path → also gibberish |

### Bug bisection matrix

| Test | TURBOMIND devices | TURBOMIND layers | Result |
|---|---|---|---|
| P2 | 1 (CUDA_TURBOMIND0) | 1 (layer 0) | ✅ Coherent, 16.91 t/s |
| P3 | 2 (0, 1) | 2 (layers 0, 10) | ✅ Coherent, 16.48 t/s |
| P3.1 | 4 (0, 1, 3, 6) | 4 (layers 0, 10, 22, 33) | ✅ Coherent, 14.52 t/s |
| P3.2 | 1 (CUDA_TURBOMIND0) | 6 (layers 0–5) | ❌ Gibberish |
| Original | 8 (0–7) | 43 (all) | ❌ Gibberish |

**Threshold**: bug appears when ≥6 layers route their experts through CUDA_TURBOMIND in the same forward pass, regardless of how those layers are distributed across TURBOMIND devices. SPRINT-024 MIN-Ne baseline (43 layers single-GPU TURBOMIND) does NOT exhibit this — distinguishing variable is unclear (256-expert routing? Multi-GPU layer-split context? Specific kernel selection?).

### Symptom

Output is degenerate token loops:
- Grouped path on 256e: `\(n:? (# # # # # # # ...`
- Per-expert path on 256e: `#n#:n# #n# #n# ...`

First 3–5 generated tokens have some structure (suggesting first decode step partially succeeds), then activations collapse into a fixed value → argmax of fixed logits → token loop.

### Update: bug is sparse-activation-specific (user hypothesis confirmed)

Forced `--override-kv 'deepseek4.expert_used_count=int:256'` to make EVERY token activate ALL 256 experts (dense routing). Re-ran the original 8-GPU full-TURBOMIND failing case:

| Mode | n_expert_used | Output character | Decode TPS |
|---|---|---|---|
| Sparse (production default) | 6 / 256 | `\(n:? (# # # # # # ...` — degenerate garbage | 13 t/s |
| **Dense (forced override)** | **256 / 256** | **`f n fibonacci n: f n fibonacci n: ...` — actual prompt words looping** | **3.4 t/s** |

The dense case shows the kernel produces **numerically valid** output that the model interprets (badly, since untrained for this config — but the TOKENS are real words from the prompt). The sparse case produces garbage.

**Conclusion**: the bug is in how the dispatch path handles sparse routing — specifically the case where `expert_offsets[i+1] == expert_offsets[i]` for the majority of experts. With 6/256 active, 250 experts have zero-token offsets. With 16-expert MIN-Ne and top-6, only 10 experts are zero-token — far below whatever threshold trips at 250.

### Likely fix surfaces (eliminated)

1. ~~**StridedPtr table for zero-token experts**~~: tested via active-experts host-side filter (built tighter weight pointer array, offsets, with `num_active=6` instead of 256). Kernel sees no empty experts. **Still gibberish.** Bug ISN'T sparse-expert-handling at the kernel level.
2. ~~**Workspace partials/barriers/flags stale across Runs**~~: tested by `cudaMemsetAsync` of all three buffers before each Run. **Still gibberish.**
3. ~~**Stream race**~~: `cudaStreamSynchronize` after each Run. **Still gibberish.**
4. ~~**cudaGraph capture eating memsets**~~: `GGML_CUDA_DISABLE_GRAPHS=1`. **Still gibberish.**

### Updated narrowing

The active-experts filter test is **especially significant** — it sends only 6 active experts (no empty ones) to the kernel via the grouped path, yet still produces gibberish. This means:

- The bug is NOT in the kernel's handling of empty experts (since we eliminated them).
- The bug is NOT in the host-side dispatch helper's offset computation (since the filter rebuilt them cleanly).
- The bug IS triggered by **repeated TURBOMIND dispatch into the same per-device `State` across multiple layers in one forward pass**.

Threshold: 5–6+ TURBOMIND-routed dispatches per device per fwd is enough to corrupt. P2 (1 layer) works; P3.1 (4 layers, 1 layer per device) works; P3.2 (6 layers on 1 device) and original (43 layers / 8 devices ≈ 5–6/device) fail. The full-activation test (forced `n_expert_used=256`) producing structured output suggests the bug is sparse-activation-related at SOME layer in the stack — but ruling out kernel-side empty-expert handling and host-side workspace state leaves the cause unidentified.

### Reverted experimental changes

All four speculative fixes were REVERTED to keep the code clean:
- Workspace memset (barriers, flags, partials) — didn't help
- `cudaStreamSynchronize` after Run — didn't help
- Active-experts host-side filter — didn't help

`ggml/vendor/turbomind/api.cc` and `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` are at their pre-investigation state. The bisection test `test_multi_device_simultaneous.cpp` stays — it's validated infrastructure that proved the kernel side works correctly in isolation.

### What's left to investigate (revised priority)

1. **Read `Gemm::Run` and inner kernel-impl for sparse-offsets handling** — focus on how `Adesc.offsets` is consumed to determine grid shape and partials allocation.
2. **Instrument: dump `D_fp16` output after a sparse vs dense call**, compare element-by-element. Find where sparse output diverges from "what it should be".
3. **Test single-layer at 256 experts with FORCED ZERO offsets for some experts** — minimal repro to compare working-sparse-1-layer behavior vs broken-sparse-many-layer behavior.

### Reverted experimental changes

The two fix attempts (workspace memset, cudaStreamSynchronize) were REVERTED to keep
`ggml/vendor/turbomind/api.cc` in its pre-investigation state. The bisection test
(`test_multi_device_simultaneous.cpp`) stays — it's already validated infrastructure.

---

## Effort estimate

Per [[feedback-effort-estimation-undocumented-hardware]] × 3:
- Gut: 1.5–3 days
- Adjusted: **4.5–9 days**

The 3x factor accounts for:
- ggml-backend buft copy logic being dense and undocumented
- TURBOMIND/CUTLASS templates carrying stream/device assumptions deep in
  instantiations
- Multi-GPU race conditions being non-deterministic to repro

---

## Out of scope

- The slot KV-position bug (FOLLOWUPS §4). Fixed independently.
- Family-alias buft (FOLLOWUPS §1). Conditional on this sprint's
  resolution — if root cause is H1, family-alias is the right fix; if
  root cause is H2/H3/H4, family-alias is orthogonal and follows later.
- Speculative decoding (SPRINT-026). Blocked behind this sprint.
- 2/4-GPU scaling sweep (FOLLOWUPS §6). Independent.

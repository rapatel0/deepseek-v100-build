# SPRINT-025-PATCH — Multi-GPU CUDA_TURBOMIND correctness regression

**Date opened:** 2026-05-15
**Status:** EXECUTING
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

---
sprint: 025
title: SPRINT-025 follow-up items discovered during execution
---

# SPRINT-025 — Execution follow-ups

Items discovered while executing SPRINT-025 that were not planned in
SPRINT-025.md. These are different from `SPRINT-025-DEFERRED.md` (which
captures items explicitly scoped out during planning).

---

## 1. CUDA_TURBOMIND family-alias buft

- **What**: A sentinel buft named `CUDA_TURBOMIND` (no device suffix) whose
  `alloc_buffer` resolves to the concrete `CUDA_TURBOMIND<layer_device>` at
  tensor allocation time, **without** populating `model.tensor_buft_overrides[]`.
  Lets `-ot 'exps=CUDA_TURBOMIND'` distribute experts across all GPUs
  participating in `-sm layer` while preserving pipeline parallelism.
- **Why discovered**: P3 implementation surfaced that the substitution point
  is inside `src/llama-model.cpp`'s tensor placement / override-resolution
  closure, after layer-to-device assignment. The standard
  `tensor_buft_overrides[]` machinery short-circuits before that point, so
  any user-facing override (including `CUDA_TURBOMIND`) trips
  `model.has_tensor_overrides()` and disables pipeline parallel
  (`llama-context.cpp:316-321`).
- **Severity**: Important (degrades quality — pipeline parallelism is
  unavailable for the layer-split + family-alias combination), **not** Critical
  for the ship gate. At M=1 decode there is no in-flight pipelining, so the
  PP loss is negligible. The TURBOMIND speed advantage (+13-22% TPS from
  SPRINT-024) only applies on the subset of experts that match `-ot`.
- **Suggested sprint**: Sprint 27 (after multi-GPU baseline is measured on
  256e in this sprint; if measurements show TURBOMIND speedup is worth
  preserving on multi-GPU 256e, prioritize family-alias).
- **Files**:
  - `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` (new
    `ggml_backend_cuda_turbomind_buffer_type_family()` factory)
  - `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` (predicate
    `ggml_backend_buft_is_cuda_turbomind_family()` and prototype)
  - `src/llama-model.cpp` (override-resolution closure: detect family-alias
    via predicate, substitute `CUDA_TURBOMIND<layer_device>` directly on
    `tensor->buft` without touching `tensor_buft_overrides[]`)

## 2. Pod `/models/dsv4-flash/` subdir convention

- **What**: The PVC `llm-models-local` is mounted as `/models` and GGUF files
  live at the root (`/models/DSv4-Flash-256e-fixed.gguf`). The
  `llamacpp-build-4gpu` pod had a `/models/dsv4-flash/` symlink that the
  fresh `llamacpp-build-6gpu` pod lacks. SPRINT-025.md's example
  invocations reference `/models/dsv4-flash/...`.
- **Severity**: Nice-to-have (path convention drift).
- **Suggested sprint**: Whenever the pod manifests are next touched —
  either bake the symlink into the container's `command` field, or update
  all sprint docs to use `/models/` directly.
- **Files**:
  - `manifests/llamacpp-build-*gpu.yaml` (add init `ln -sf /models /models/dsv4-flash` or similar)
  - `docs/sprints/SPRINT-025.md` (update example commands)

## 3. tar-pipe over kubectl exec is fragile for large directories

- **What**: Streaming `tar cf - … | kubectl exec - tar xf -` through two
  kubectl exec stdin/stdout pipes was repeatedly interrupted by either
  buffering issues or transport hiccups on the 274 MB `build_so/` tree
  (270 MB of CUTLASS/fmt artifacts we don't need). `kubectl cp` to a local
  intermediate then `kubectl cp` to the target was reliable.
- **Severity**: Nice-to-have (operational ergonomics).
- **Suggested sprint**: Any future sprint that needs cross-pod artifact
  copies — use `kubectl cp` rather than tar-pipe.
- **Files**: Operational, no code change.

## 4. llama-server "Invalid input batch." — slot KV-position desync

- **What**: HTTP 500 `Invalid input batch.` on `/completion` requests when
  `cache_prompt:false` is passed AND the chosen slot has prior state.
  Actual error from the server log (8-GPU run):
  ```
  init: the tokens of sequence 1 in the input batch have inconsistent
  sequence positions:
    - the last position stored in the memory module of the context
      (i.e. the KV cache) for sequence 1 is X = 36
    - the tokens for sequence 1 in the input batch have a starting
      position of Y = 0
  it is required that the sequence positions remain consecutive: Y = X + 1
  decode: failed to initialize batch
  llama_decode: failed to decode, ret = -1
  srv  update_slots: Invalid input batch. i = 0, n_batch = 2048, ret = -1
  ```
- **Why discovered**: P4 ship-gate coherence checks on both 6-GPU and
  8-GPU runs. First request on a fresh slot works; second request to
  the same slot with `cache_prompt:false` fails because the slot's KV
  cache wasn't fully cleared between requests.
- **Severity**: Important. Not a ship blocker (each slot's first request
  works, and `cache_prompt:true` works on subsequent requests by finding
  the longest matching prefix), but a real bug when callers explicitly
  want a clean cache reset.
- **Workaround**: Pass `cache_prompt:true` (the default), OR cycle through
  different `id_slot` values, OR send a tiny `n_predict:1` reset request
  between real requests.
- **Suggested sprint**: SPRINT-026 P0. The fix is in
  `tools/server/server.cpp` slot reset path — when `cache_prompt:false`
  the slot should call `memory_seq_rm(seq_id, 0, end)` before queueing
  the new batch, not after.
- **Files**: `tools/server/server.cpp` (`launch_slot_` / `update_slots`).

## 5. Multi-GPU CUDA_TURBOMIND correctness regression — **CRITICAL**

- **What**: Routing expert tensors through `CUDA_TURBOMIND<N>` (N=0..7) on
  an 8-GPU layer-split 256e load produces **gibberish output**. The
  kernels run without error, decode TPS is in the same ballpark as the
  default-buft baseline (~11.7 t/s), but the generated tokens are
  incoherent — repeating single tokens like `# # # # # # ...` from a
  Python `def fibonacci(n):` prompt.
- **What was tested**: 8-GPU 256e launched with per-layer `-ot` regex
  routing each layer's expert tensors to the matching device's
  `CUDA_TURBOMIND<N>` buft. Load succeeded; 8 CUDA_TURBOMIND buffer
  groups total 140 GiB of expert weights. Decode test produced broken
  output on the one prompt that didn't hit the §4 batch bug.
- **Why discovered**: Followup to a pointed question — none of the
  multi-GPU numbers in REPORT-19 use TURBOMIND. SPRINT-024 verified
  TURBOMIND correctness at single-GPU only (`test_correctness.cpp`,
  `test_grouped.cpp`); SPRINT-025 P2 verified the per-device State
  refactor (`test_multi_device.cpp`) doesn't corrupt the workspace
  pointers, but did NOT verify that simultaneous CUDA_TURBOMIND<i> and
  CUDA_TURBOMIND<j> dispatch in the same forward pass produce the
  same output as the default-buft path.
- **Severity**: **CRITICAL**. The architectural promise of CUDA_TURBOMIND
  on multi-GPU 256e is broken. The +13–22% SPRINT-024 lift only applies
  if/when this is fixed. Without it, multi-GPU 256e ships on the
  default cuda buft baseline (which IS coherent — verified in REPORT-19).
- **Likely root causes** (need investigation):
  1. **Layer-device mismatch from manual `-ot`**: my override pinned
     layer L's experts to `CUDA_TURBOMIND<k>` based on inferred buffer
     sizes; if `-sm layer` actually placed layer L's attention on a
     different GPU, cross-GPU activation transfers happen per token
     and may be misrouted. The family-alias work ([[1]]) would
     intrinsically avoid this by binding to the layer's actual device.
  2. **TURBOMIND kernel cross-device contamination**: the per-device
     `State[]` refactor was workspace-pointer-only. The Gemm object,
     CUDA streams, or scratch buffers may have implicit assumptions
     about single-device context that break when dispatched alternately
     across devices in the same forward pass.
  3. **Activation buft mismatch**: the input/output activations for a
     CUDA_TURBOMIND kernel call may live in the regular cuda buft for
     a different device. The dispatch helper in
     `ggml-cuda-turbomind.cu` may not handle the device-crossing copy
     correctly.
- **Suggested sprint**: Before SPRINT-027 family-alias work. The
  family-alias addresses cause (1) but not (2) or (3) — those need
  independent fixes. **Minimum repro test**: extend `test_multi_device.cpp`
  to verify that *simultaneous* dispatch on GPU 0 and GPU 1 produce
  outputs matching independent single-device runs on the same input.
- **Files**:
  - `ggml/vendor/turbomind/test_multi_device.cpp` (extend to simultaneous-
    dispatch correctness check)
  - `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` (review
    `ggml_cuda_mul_mat_grouped_turbomind` + the helper that handles
    activation device-crossing)
  - `ggml/vendor/turbomind/api.cc` (review whether
    `ggml_turbomind_mul_mat_grouped`'s Gemm object handles streams /
    scratch correctly when called from a different device in the same
    forward pass)

## 6. 2-GPU / 4-GPU scaling sweep points

- **What**: SPRINT-025 P5 spec called for 2/4/6/8 GPU sweep. 6-GPU and
  8-GPU data points are captured in REPORT-19. 2/4-GPU sub-runs via
  `CUDA_VISIBLE_DEVICES=0,1` / `0,1,2,3` on the existing 8-GPU pod would
  complete the curve.
- **Why discovered**: P5 execution captured the two largest-fit
  configurations but not the lower end. 256e at 146 GiB needs ≥ 5 GPUs
  to fit at all (≤ 4×32 = 128 GiB < 146 GiB), so 2/4-GPU sub-runs would
  need a different smaller model (MIN-Ne) to draw a complete curve, or
  would have to use the 256e with CPU layer offload (degraded).
- **Severity**: Nice-to-have for capacity planning; not blocking. The
  6-vs-8 comparison already in REPORT-19 captures the meaningful
  trade-off (8 GPUs = headroom, 6 GPUs = slightly faster decode).
- **Suggested sprint**: Whichever sprint asks "how does scaling shape
  decode TPS at M=1?". Add 4-GPU + MIN-Ne data points.
- **Files**: No code change; REPORT-19 amendment.

---

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| CUDA_TURBOMIND family-alias buft | Important | 027 | ggml-cuda-turbomind.{cu,cuh}, llama-model.cpp |
| /models/dsv4-flash/ subdir convention | Nice-to-have | next manifest touch | manifests/*.yaml, SPRINT-025.md |
| tar-pipe fragility for cross-pod copies | Nice-to-have | operational | none |
| llama-server "Invalid input batch." 500 | Important | 026 P0 | tools/server/server.cpp, src/llama-batch.cpp |
| **Multi-GPU CUDA_TURBOMIND correctness regression** | **CRITICAL** | before SPRINT-027 | ggml-cuda-turbomind.cu, api.cc, test_multi_device.cpp |
| 2/4-GPU scaling sweep points (MIN-Ne) | Nice-to-have | when needed | REPORT-19 amendment |

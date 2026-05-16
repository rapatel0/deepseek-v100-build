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

## 4. llama-server "Invalid input batch." 500 on certain prompts

- **What**: 256e on 6 GPUs returns HTTP 500 `Invalid input batch.` on
  certain `/completion` requests with multi-word ASCII content prompts
  (`"Hello world! My name is"`, `"Once upon a time, there was a"`).
  Reproduces deterministically. Code/identifier prompts
  (`"def fibonacci(n):"`) and prompts with leading punctuation work.
  No backtrace in server log — error returned at request validation.
- **Why discovered**: P4 ship-gate coherence checks. 3/5 prompts decoded
  fine; 2/5 returned the 500 error. The two passing real-content prompts
  (`"The capital of France is"`, `"def fibonacci(n):"`) confirm the model
  weights are healthy.
- **Severity**: Important. Not a ship blocker for the SPRINT-025 ship gate
  (which only requires decode coherence on _some_ prompts), but a real
  bug that affects ~half of natural prompts.
- **Suggested sprint**: SPRINT-026 P0 sanity-check phase or earlier. Reproduce
  with `--verbose`, look at the request validation path in
  `tools/server/server.cpp` for the "Invalid input batch" string.
- **Files**: `tools/server/server.cpp` (request validation), possibly
  `src/llama-batch.cpp` (input batch construction).

## 5. 8-GPU 256e scaling sweep

- **What**: SPRINT-025 P5 spec called for a 2/4/6/8 GPU scaling sweep.
  Only 6 GPUs reservable on gpu-01 without disturbing `tcg-dev` and
  `llamacpp-build` (each holding 1 GPU). 2/4-GPU sub-runs via
  `CUDA_VISIBLE_DEVICES=0,1` / `0,1,2,3` on the existing pod are easy
  (~1 hour total); 8-GPU requires either deleting one of the other pods
  or waiting until they release.
- **Why discovered**: P5 execution. The 6-GPU data point is already in
  REPORT-19; 2/4 sub-runs would let us draw the scaling curve, and 8-GPU
  would extend it to the original sprint target.
- **Severity**: Important if scaling characterization is needed for
  capacity planning, otherwise Nice-to-have. Decode TPS at M=1 should
  not improve dramatically beyond what the per-GPU expert-traffic
  bottleneck allows.
- **Suggested sprint**: Whichever sprint asks "how much faster on N
  GPUs?". Can be folded into REPORT-19 retroactively.
- **Files**: No code change; a script + REPORT-19 amendment.

---

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| CUDA_TURBOMIND family-alias buft | Important | 027 | ggml-cuda-turbomind.{cu,cuh}, llama-model.cpp |
| /models/dsv4-flash/ subdir convention | Nice-to-have | next manifest touch | manifests/*.yaml, SPRINT-025.md |
| tar-pipe fragility for cross-pod copies | Nice-to-have | operational | none |
| llama-server "Invalid input batch." 500 | Important | 026 P0 | tools/server/server.cpp, src/llama-batch.cpp |
| 8-GPU 256e scaling sweep | Nice-to-have | when needed | REPORT-19 amendment |

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

---

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| CUDA_TURBOMIND family-alias buft | Important | 027 | ggml-cuda-turbomind.{cu,cuh}, llama-model.cpp |
| /models/dsv4-flash/ subdir convention | Nice-to-have | next manifest touch | manifests/*.yaml, SPRINT-025.md |
| tar-pipe fragility for cross-pod copies | Nice-to-have | operational | none |

# SPRINT-025 — Technical Critique: Full DSv4-Flash-256e Multi-GPU Landing

This document provides a technical critique of the `CLAUDE-DRAFT` and `CODEX-DRAFT` for SPRINT-025, evaluated against the reference `INTENT.md` and the existing codebase state.

---

## 1. Executive Summary

Both drafts correctly identify that **8× V100 (256 GiB)** is the only viable configuration for the full 156 GiB model and that **`LLAMA_SPLIT_MODE_LAYER`** is the logical P0 path. Both drafts significantly improve upon the `INTENT.md` by identifying a critical architectural blocker: the `TmLib` singleton in the CUDA_TURBOMIND implementation, which the Intent incorrectly assumed was "multi-device ready."

**CLAUDE-DRAFT** is the superior implementation plan, providing granular, phase-by-phase gates and a concrete refactor strategy for the singleton bug. **CODEX-DRAFT** offers better strategic insight into the tensor-parallel plumbing (specifically identifying the `deepseek4` guard in `src/llama-model.cpp`).

---

## 2. CLAUDE-DRAFT Evaluation

### Strengths
- **Granularity**: The P0–P6 phase structure is excellent, with explicit gates for each phase.
- **TmLib Refactor**: Provides a detailed C++ proposal for replacing the singleton with a `std::array` of slots, including the `dlopen` ref-counting nuance.
- **VRAM Precision**: Breaks down memory per GPU including scales, KV context variants (2K vs 8K), and workspace overhead.
- **NCCL Linkage**: Correctly identifies that standard CMake 3.22 lacks `FindNCCL.cmake` and proposes vendoring or using the llama.cpp version.

### Weaknesses
- **Missing Code Guard**: Fails to mention the `LLAMA_SPLIT_MODE_ROW` guard in `src/llama-model.cpp` which explicitly blocks the `deepseek4` architecture.
- **Overly Optimistic P2**: Assumes the `TmLib` refactor is 1-2 days without verifying if the underlying `libggml-turbomind.so` is thread-safe for multi-context initialization.

### Risk Analysis & Edge Cases
- **Strengths**: Identifies the risk of cross-device pointers (§7 Risk 5) and NCCL topology hangs (§7 Risk 7).
- **Gaps**: Does not explicitly address the "unbalanced layer split" mitigation beyond a passing mention of `-ts`.

### DoD Completeness
- High. The inclusion of `ldd` verification and specific TPS reporting makes the DoD actionable.

---

## 3. CODEX-DRAFT Evaluation

### Strengths
- **Strategic Guard Identification**: Explicitly identifies the `deepseek4` ROW-mode guard as a blocker, which is a major technical hurdle.
- **Placement Logic**: Proposes a "Fast path" vs "Better path" for expert placement, identifying the interaction between tensor overrides and pipeline parallelism.
- **Visibility Checks**: Strong focus on P0 visibility (device counting/enumeration) which is often where k8s-based multi-GPU setups fail.

### Weaknesses
- **Vague Implementation**: Phases are described at a high level. Lacks specific code snippets or line-reference targets for the refactor.
- **NCCL Assumptions**: Assumes `find_package(NCCL)` "already exists" and works, ignoring the CMake 3.22 version constraint.

### Risk Analysis & Edge Cases
- **Strengths**: Highlights the risk of tensor overrides disabling pipeline parallelism (§5).
- **Gaps**: Risk analysis is less technical/granular than CLAUDE's (e.g., missing specific CUDA error types).

### DoD Completeness
- Moderate. Lacks the granular technical check-offs (like `nm` symbol resolution) present in CLAUDE.

---

## 4. Technical Comparison & Error Correction

### 4.1 NCCL Build Wiring
- **Intent Error**: Intent assumes `find_package(NCCL)` is sufficient.
- **CLAUDE Correction**: Correctly notes that `cmake/Modules/FindNCCL.cmake` must be present/vendored for CMake 3.22.
- **Critique**: Both drafts correctly identify that NCCL must be added to the build image. CLAUDE's `ldd` and `nm` verification steps are the "gold standard" for this P1 gate.

### 4.2 LLAMA_SPLIT_MODE Plumbing
- **The "ROW" Blocker**: `CODEX` correctly identifies the `LLAMA_SPLIT_MODE_ROW` guard in `src/llama-model.cpp`. `CLAUDE` chooses `LAYER` but doesn't mention the code change required if `ROW` were pursued.
- **Recommendation**: The sprint should stick to `LAYER` as the primary target. If `ROW` is attempted (P6), it requires patching `src/llama-model.cpp` and `llama-model-loader.cpp`.

### 4.3 per-GPU CUDA_TURBOMIND Init
- **Intent Error**: Intent says "No structural change needed." This is **false**.
- **Draft Correctness**: Both drafts correctly identify that the current `TmLib` singleton and the `ggml_turbomind_init` call are single-device bound.
- **Comparison**: CLAUDE's refactor plan (§3.5) is much more robust, addressing the singleton flip that would otherwise cause "lethal" re-initialization thrashing during multi-GPU dispatch.

### 4.4 VRAM Accounting
- **The Numbers**: 156 GiB model / 8 GPUs = 19.5 GiB.
- **CLAUDE Model**: ~25 GiB (with 2K FP16 KV).
- **CODEX Model**: ~25 GiB (with q8_0 KV).
- **Critique**: CLAUDE's accounting is superior as it differentiates between the model weights and the packed scale overhead (~0.5 GiB per GPU), which is critical given the tight 32 GiB per-GPU limit.

---

## 5. Final Recommendations

1.  **Adopt CLAUDE's Phase Structure**: Use CLAUDE's P0-P6 structure for the implementation plan.
2.  **Incorporate CODEX's Guard Awareness**: Ensure the sprint acknowledges the `src/llama-model.cpp` guard if transitioning to ROW mode.
3.  **Prioritize P2 Refactor**: The per-device `TmLib` array is the "make or break" for this sprint. CLAUDE's P0.3 smoke test (C program for multi-device init) should be the very first technical action.
4.  **VRAM Policy**: Default to **2K context / FP16 KV** initially (CLAUDE's plan) but have the **q8_0 KV** fallback (CODEX's plan) ready if fragmentation exceeds 10%.

**Verdict**: Merge CLAUDE's technical depth with CODEX's strategic awareness of the `deepseek4` guard.

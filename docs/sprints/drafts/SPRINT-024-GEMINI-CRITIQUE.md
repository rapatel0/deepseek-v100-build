# SPRINT-024 — GEMINI CRITIQUE

## Executive Summary
Both drafts correctly identify the core bottleneck (kernel-launch overhead at M=1) and the primary solution (grouped MoE dispatch via `ggml_turbomind_mul_mat_grouped`). They both respect the sm70 constraints and the required performance gates. However, Claude's draft contains a significant technical error regarding scale stride calculation, while Codex provides a more robust implementation roadmap but is slightly more conservative on the initial performance path (FP16 boundary).

---

## 1. Draft Analysis: CLAUDE

### Strengths
- **Architecture Clarity**: §3.3 provides a very clear breakdown of device-side metadata requirements and scratch memory estimation (~250 KiB), which is helpful for resource planning.
- **Phased Value**: The use-case table in §2 effectively communicates how each phase adds value even if the sprint is truncated.
- **Verification Strategy**: The "P0 — Plumbing audit" is a strong addition, ensuring the baseline is locked and the thesis is verified before significant work begins.

### Weaknesses
- **Technical Error (Scales)**: §3.2 incorrectly derives the scale stride as `(K / group_size) * sV`. Per `api.cc`, the scale layout `Vdesc` (after swap) is row-major `[K/group_size, N]` with `ld = N`. The stride must be `N`, not related to the row count `K/group_size`.
- **Metadata Lifecycle**: Suggests caching `StridedPtr` arrays in `extra` struct as a P4 optimization. Given the launch-bound nature, this should likely be a P1 consideration to avoid unnecessary host-to-device traffic on the hot path.

### Gaps in Risk Analysis / Missing Edge Cases
- **Alignment Risk**: Misses the risk of misaligned `StridedPtr` writes on the host if not using `__align__(16)` correctly in the local mirror struct, which would cause `uint4` load failures on device.
- **Edge Case**: Does not explicitly address the "empty expert" case (zero tokens routed to an expert), which requires `expert_offsets` to handle stable prefix-sums without causing OOB reads.

### DoD Completeness
- **High**: Includes `REPORT-18` and specific match-rate gates for quality.
- **Missing**: Does not specify a gate for "prompt TPS" (prefill), only decode.

---

## 2. Draft Analysis: CODEX

### Strengths
- **Implementation Detail**: §4 provides a very surgical step-by-step for `ggml_cuda_mul_mat_id` integration, correctly identifying the need to stop slicing `src0`.
- **Correctness Focus**: P3 explicitly includes a "grouped-routing fixture" with non-uniform offsets and empty experts, which is a critical edge case.
- **ABI Awareness**: Correctly identifies the `void**` vs `StridedPtr*` ambiguity in the C ABI signature and proposes explicit mirrored structs.

### Weaknesses
- **Secondary FP16 Boundary**: Places the FP16 boundary work (P4) as secondary. Given the 1.45x perf gate (24 t/s), this "optional" work might be mandatory to overcome the DRAM traffic bottleneck once launch overhead is removed.
- **Lack of Baseline Audit**: Does not explicitly mandate a P0 baseline reproduction like Claude's draft, which is risky given the "random weights" nature of MIN models.

### Gaps in Risk Analysis / Missing Edge Cases
- **Synchronization**: Misses the risk of the `cudaMemcpy` in `api.cc:607` potentially serializing the stream if called repeatedly in a multi-slot context (though acceptable for single-slot).
- **Edge Case**: Does not address behavior when `num_experts == 1` (dense layers using MoE op), which should ideally fall back to the legacy path to avoid grouped overhead.

### DoD Completeness
- **Excellent**: Very specific numeric gates for both MIN-16e and MIN-32e.
- **Decision-Complete**: Explicitly mandates a "ship vs no-ship" decision based on the 21-24 t/s threshold.

---

## 3. Technical Corrections & Deep Dive

### StridedPtr Layout & Alignment
Both drafts correctly identify that `weights_packed` is not a `void**` but a `StridedPtr[]`.
- **Layout**: `void* ptr` (8B), `int stride` (4B), plus 4B implicit padding.
- **Alignment**: **Must** be `__align__(16)` because the sm70 kernel uses `(uint4&)ptr = __ldg(...)`. A 12-byte struct or unaligned 16-byte struct will cause a memory fault.

### Packed Stride (`packed_ld`) Derivation
- **Weights (B)**: For `HMMA_884` `OPERAND_B` `Pack_M=1` on sm70, the rule is indeed `packed_ld = K * 32`. 
    - *Clarification*: This is because `Packing_v2` interleaves 32 rows, expanding the logical row width to `K * 32`.
- **Scales (V)**: Claude's formula `(K / group_size) * sV` is **incorrect**. 
    - *Correction*: Scales use `Vdesc.ld = N` (the width of the scale matrix in its row-major packed form). There is no "Pack_M" expansion for scales on sm70.

### Grouped C ABI (`api.cc`)
- The `weights_packed` signature `const void* const*` is a type-safety "lie" in the C ABI. It must be passed a device pointer to a `StridedPtr` array.
- The `expert_offsets[num_experts]` read in `api.cc` is synchronous. While acceptable for single-slot decode, any future "multi-slot" extension (SPRINT-025) must move this to an async stream-local check or eliminate the host-side dependency.

### sm70 Turbomind APIs
- `ggml_turbomind_mul_mat_grouped` expects `Adesc.ld = K`.
- The grouped helper in `ggml-cuda` must ensure that `token_indices` is passed as `NULL` if `A` is already permuted/gathered, or provided if using indexed access. Codex's P2.3 choice to "choose one convention" is the correct architectural approach.

---

## 4. Final Recommendation
Merge the **Codex P1-P3 implementation roadmap** with **Claude's P0 baseline audit** and **Codex's numeric perf gates**. 
**Crucial Correction**: Ensure the implementation uses `stride = N` for scales and strictly enforces `__align__(16)` for the `StridedPtr` mirror struct. Move the FP16 boundary work (Claude P3 / Codex P4) to a "high priority" status to ensure the 24 t/s gate is hit.

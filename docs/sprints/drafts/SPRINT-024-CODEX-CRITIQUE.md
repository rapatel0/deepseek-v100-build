# SPRINT-024 — Codex Critique

## Overall

- The **Claude draft is clearly stronger**. It is closer to the intent, has a real phase structure, and understands the current `ggml-cuda` / TurboMind split much better.
- The **Gemini draft is too shallow to execute safely**. It misses key wiring work and contains concrete mistakes about the grouped ABI and `StridedPtr` usage.
- Best revision path: keep Claude's overall shape, but narrow the ship criteria so **grouped dispatch is the primary deliverable** and FP16-boundary work remains secondary, as the intent says.

## Claude draft

### Strengths

- Best alignment with the intent: grouped MoE dispatch first, real-model quality second, FP16-boundary as a follow-on.
- Strong repo awareness: it understands the current per-expert path in `ggml_cuda_mul_mat_id`, the `ggml_turbomind_tensor_extra` contents, and the existing `mul_mat_grouped` ABI.
- Verification is materially better than Gemini's: grouped-vs-legacy correctness, perf gates, quality checks, and profiling are all spelled out.
- Definition of Done is substantially more complete and reviewable.

### Weaknesses

- It is **over-scoped**. P1 grouped dispatch is the sprint's main value; P3 graph-builder work, P4 tuning, P5 reporting, and P6 process items make the sprint heavier than the intent requires.
- It is internally inconsistent on fallback behavior:
  - §3.1 says the grouped predicate is total and there is “no fallback”.
  - P1.3 later keeps the per-expert path as fallback.
- It still leans on host-side sort/metadata rebuild in P1.2, despite framing the architecture as “replayed on-device”. That is fine pragmatically, but the document should say so more directly.

### Gaps In Risk Analysis

- Missing explicit risk that **loader plumbing is incomplete today**. Current `ggml-cuda-turbomind.cu` resolves `ggml_turbomind_mul_mat`, but not `ggml_turbomind_mul_mat_grouped`.
- Missing explicit risk that **P3 touches graph construction, not just CUDA dispatch**. The FFN boundary is defined in the graph builder, so this is not a localized CUDA-only follow-on.
- Missing explicit risk that prefill may have a different bottleneck profile than decode even if grouped launch amortization works well at `M=1`.

### Missing Edge Cases

- No explicit test gate for `n_expert_used == 1`, even though the draft discusses it later as an open question.
- No explicit gate for experts with zero routed tokens beyond “should work”.
- No explicit test/fallback story for `scales_packed == NULL` / FP16 weights if the grouped helper is ever widened beyond the current quantized TurboMind set.
- Routing cardinality is inconsistent inside the draft: it uses **6 active experts** in the overview but later says “DSv4 uses top-8”. That should be resolved before the sprint is finalized.

### Definition Of Done Completeness

- Mostly strong.
- The main problem is that DoD makes **P3 mandatory** (`FP16 boundary across the FFN landed and active`). That conflicts with the intent, which marks FP16-boundary work as secondary.
- The DoD should explicitly allow: “ship grouped dispatch if perf/correctness/quality gates pass, even if P3 slips”.
- It should also include one concrete proof that the grouped path was actually exercised, for example a debug counter or log on first dispatch.

### Specific Technical Errors

- **Scale `StridedPtr.stride` is described incorrectly.**
  - Claude §3.2 says the scale stride should be `(K / group_size) * sV` and cached from `GetConverters(...)[1]->pack`.
  - That does **not** match the current sm70 API logic. In `ggml/vendor/turbomind/api.cc`, the grouped path sets `Vdesc.ld = 0` and expects the caller-supplied `StridedPtr.stride` to match the packed scale descriptor `ld`; in the single-path reconstruction, that value is the post-swap `s_desc.ld` / `Vdesc.ld`, **not** `K / group_size` (`api.cc:525-542`, `api.cc:655-669`).
  - This is the most important technical fix needed in the Claude draft.
- **The dependency section claims grouped dlsym wiring already exists, but it does not.**
  - Claude §9 says `ggml_turbomind_mul_mat_grouped` is already callable from `libggml-cuda` via `dlsym`.
  - Current code does not support that: `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` only defines `pfn_mul_mat` and resolves `ggml_turbomind_mul_mat` (`lines 37-58`, `85-91`).
  - Loader work is still part of P1, not a satisfied precondition.

## Gemini draft

### Strengths

- Concise and easy to scan.
- Keeps the main sprint thesis intact: grouped dispatch is the primary lever, with quality and FP16 boundary as follow-ons.
- The high-level perf target matches the intent.

### Weaknesses

- It is **too abstract to implement from**. It does not describe the current `ggml_cuda_mul_mat_id` flow, the existing host-side sort, or how regroup/scatter actually fits into the call path.
- It omits several concrete work items that are required in the current tree:
  - loader / `dlsym` wiring for `ggml_turbomind_mul_mat_grouped`
  - scratch allocation shape
  - permutation/unpermutation details
  - grouped-vs-legacy verification beyond one sentence
- It incorrectly targets `ggml/vendor/turbomind/api.cc` as a file to change, even though the sprint intent is to **wire the existing grouped ABI through ggml-cuda**.
- P3 is underspecified to the point of being misleading; moving the FP16 boundary is not just “modify FFN dispatch in `ggml-cuda.cu`”.

### Gaps In Risk Analysis

- No risk called out for wrong `StridedPtr` / packed-ld layout, which is the key correctness hazard.
- No risk called out for missing grouped symbol resolution in `ggml-cuda-turbomind.cu`.
- No risk called out for prefill behavior, host sync, or CUDA-graph incompatibility.
- No risk called out for non-MIN model availability or fallback if `AVG-16e` is missing.
- No risk called out for grouped-vs-per-expert divergence; it jumps directly to CPU-vs-GPU quality.

### Missing Edge Cases

- No explicit handling for experts with zero tokens.
- No explicit test matrix across `n_expert_used` values (`1/2/6/8/16`).
- No explicit prefill case (`M >> 1`), even though grouped dispatch changes the launch/computation balance there.
- No explicit handling of `token_indices == nullptr` vs gathered-input mode.
- No explicit handling of FP16/no-scale cases.
- No fallback path if grouped dispatch rejects or if the grouped symbol cannot be loaded.

### Definition Of Done Completeness

- Incomplete.
- It lacks:
  - proof that the grouped path is actually active
  - a grouped-vs-per-expert correctness gate on the same model
  - a fallback quality model if `AVG-16e` is unavailable
  - reporting/profiling outputs tied back to the sprint thesis
- It also hard-requires the FP16 boundary, which again is stricter than the intent.

### Specific Technical Errors

- **The `StridedPtr` description is wrong.**
  - Gemini §3.1 says each `StridedPtr` contains `weight`, `scales`, and `stride`.
  - That is false. `StridedPtr` is only:
    - `void * ptr`
    - `int stride`
  - See `research/lmdeploy/src/turbomind/kernels/gemm/matrix_ptr.h:9-13`.
  - The grouped ABI takes **two separate arrays**: `weights_packed` and `scales_packed` (`ggml/vendor/turbomind/include/ggml-turbomind-api.h:173-184`).
- **The draft puts the packed-ld fix in the wrong place.**
  - Gemini says to modify `ggml/vendor/turbomind/api.cc` so grouped matmul “uses `packed_ld` correctly”.
  - In the grouped API, `Bdesc.ld` and `Vdesc.ld` are intentionally set to `0`; the packed stride comes from the caller-supplied `StridedPtr` records (`api.cc:641-665`).
  - The correctness burden is therefore in the **ggml-cuda caller metadata**, not a new `api.cc` change.
- **P0 sizes `token_indices` incorrectly.**
  - It says to pre-allocate `token_indices` with size `N`.
  - `N` is the output-channel dimension of the expert weight. Token metadata should scale with **routed tokens**, i.e. `total_tokens = n_tokens * n_expert_used` (or equivalent grouped offsets), not `N`.
- **P3 targets the wrong layer of the stack.**
  - “Modify the FFN dispatch in `ggml-cuda.cu`” is not sufficient to move the FP16 boundary.
  - The FFN MoE flow is built in the graph builder (`src/llama-graph.cpp`, e.g. the `build_lora_mm_id` MoE path around `1429-1598`), so P3 needs graph-level changes plus helper changes.
- **Loader work is missing entirely.**
  - The current CUDA TurboMind shim only resolves `ggml_turbomind_mul_mat`; there is no grouped function pointer today in `ggml-cuda-turbomind.cu`.

## Recommendation

- Use the **Claude draft as the base**.
- Fix the two factual issues first:
  - scale `StridedPtr.stride` must track packed `Vdesc.ld`, not `(K / group_size) * sV`
  - grouped `dlsym` wiring is not already present
- Then reduce scope:
  - make grouped dispatch + correctness + perf + non-MIN quality the ship gate
  - keep FP16-boundary work as a secondary stretch, consistent with the intent

# SPRINT-024 — Grouped MoE dispatch landing (V100 sm70)

**Status:** DRAFT 2026-05-15
**Predecessor:** SPRINT-023
**Primary goal:** Replace per-expert MoE FFN launches with grouped turbomind dispatch and land a measurable decode-TPS gain on V100.

## Overview

SPRINT-023 proved the CUDA_TURBOMIND path is functionally integrated and about 3-3.6x faster than the old CPU-MoE baseline, but it also exposed the next bottleneck clearly: decode is now launch-bound at `M=1`. The current path still does one `ggml_turbomind_mul_mat` launch per active expert per layer, plus an FP32->FP16->FP32 cast pair around every call.

SPRINT-024 lands the grouped path that turbomind already exposes through `ggml_turbomind_mul_mat_grouped`. The core change is in `ggml_cuda_mul_mat_id`: keep the existing token-by-expert routing logic, but stop slicing `src0` and launching once per expert. Instead, build one grouped call per MoE linear per layer using device-side `token_indices`, `expert_offsets`, and per-expert `StridedPtr` tables for packed weights and scales.

This sprint is successful only if it is decision-complete. Either:

1. Grouped dispatch clears the perf gate and becomes the new default TURBOMIND MoE path on V100.
2. It improves throughput but misses the bar, in which case the sprint must end with a precise profile and the next extension identified.
3. It fails to move the bottleneck materially, in which case the sprint must say so clearly and stop.

## Use Cases

1. **Single-slot decode on MIN fixtures**
   `DSv4-Flash-MIN-{8e,16e,32e}` with `-ot exps=CUDA_TURBOMIND0` should exercise grouped MoE dispatch and improve token-generation TPS over the SPRINT-023 per-expert path.

2. **Deterministic regression against the existing TURBOMIND path**
   For the same prompt, seed, and model, grouped dispatch should match the legacy per-expert TURBOMIND outputs within FP16 tolerance and should not introduce NaNs or route-order corruption.

3. **Real-model quality verification**
   A non-`MIN-*` model that fits on a single V100, preferably `DSv4-Flash-AVG-16e`, should produce coherent deterministic output so the sprint is not validated only on broken-weight fixtures.

4. **Future multi-slot work**
   The grouped metadata path should become the base for SPRINT-025+ work on parallel decode slots or speculative decode. SPRINT-024 does not implement that work, but it should leave the dispatch surface ready for it.

## Architecture

### 1. Dispatch surface

The grouped path stays behind the existing `CUDA_TURBOMIND` opt-in surface. No new ggml op and no public ggml ABI change.

- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `ggml_cuda_mul_mat()` keeps the existing dense / single-expert turbomind path.
  - `ggml_cuda_mul_mat_id()` becomes the MoE switchpoint:
    - if `src0->buffer->buft` is not `CUDA_TURBOMIND`, behavior is unchanged
    - if `src0` is `CUDA_TURBOMIND` and `src0->type ∈ {GGML_TYPE_MXFP4, GGML_TYPE_F8_E4M3_B128}`, bypass the per-expert loop and call a new grouped helper

The mmvq/mmq/mmf early-return paths remain disabled for `CUDA_TURBOMIND` tensors. The grouped helper becomes the only optimized MoE path for those tensors.

### 2. Routing metadata

`ggml_cuda_mul_mat_id()` already computes the routing order needed for grouped execution:

- `ids_to_sorted_host`: token order packed by expert
- `ids_from_sorted_host`: inverse map for restoring output order
- `tokens_per_expert[e]`: number of routed token rows for expert `e`

SPRINT-024 formalizes this into grouped metadata:

- `token_indices_dev`
  - device `int32_t[total_routes]`
  - values are the token-row indices already emitted in `ids_to_sorted_host`
- `expert_offsets_dev`
  - device `int32_t[num_experts + 1]`
  - exclusive prefix sum of `tokens_per_expert`
  - `expert_offsets[0] = 0`
  - `expert_offsets[num_experts] = total_routes`

This preserves the current host-side sort implementation. The sprint does not attempt to move gating/sorting to CUDA; that would be a different sprint.

### 3. Packed per-expert pointer tables

The grouped turbomind path is not a raw `void*[]` pointer array in practice. For `ld == 0`, the sm70 grouped iterator resolves one per-expert record from a `StridedPtr` table:

```cpp
struct __align__(16) StridedPtr {
    void* ptr;
    int   stride;
};
```

SPRINT-024 should mirror that layout on the ggml-cuda side with a local helper struct and `static_assert(sizeof(...) == 16)`.

For each expert `e`, the grouped helper builds:

- `weight_ptrs_host[e] = { .ptr = src0->data + e * src0->nb[2], .stride = packed_ld }`
- `scale_ptrs_host[e]  = { .ptr = scales_dev + e * scales_per_expert, .stride = packed_scale_ld }`

Those host vectors are uploaded once per grouped launch to:

- `weight_ptrs_dev`
- `scale_ptrs_dev`

and then passed to `ggml_turbomind_mul_mat_grouped()` as opaque pointers.

### 4. Stride contract: packed `ld`, not logical `K`

This is the easiest place to reintroduce the P2 bug, so the sprint must treat it as a first-class correctness constraint.

For the sm70 path we care about, `conv_w` resolves to the `HMMA_884` operand-B pack. The grouped helper must set `StridedPtr.stride` to the packed leading dimension that the kernel iterator expects, not the unpacked logical `K`.

For the common `Pack_M=1` weight path on sm70:

- `packed_ld = K * 32` for `HMMA_884 | OPERAND_B | Pack_M=1` when the packed descriptor is row-major
- if the converter resolves to col-major after operand swapping, use the corresponding packed-rows value derived by the same `Packing_v2` logic already used in `api.cc`

The grouped helper must derive `packed_ld` exactly the same way `ggml_turbomind_mul_mat()` does today. Reusing the same formula is preferred over re-encoding ad hoc constants.

Equivalent rule for scales:

- `StridedPtr.stride` for `V` must be the packed scale descriptor `ld`
- for the current sm70 FP8 / MXFP4 path, that is the post-swap `Vdesc.ld`, not `K / group_size`

### 5. Output layout

Grouped execution writes expert-packed outputs to a contiguous temporary:

- `dst_sorted_fp16[total_routes, N]` in grouped expert order

The existing inverse gather remains valid:

1. grouped GEMM writes sorted output
2. optional FP16->FP32 conversion happens once on the sorted buffer
3. `get_rows_cuda(..., ids_from_sorted, ...)` restores `[token, expert_slot]` order into `dst`

The critical architectural change is that `src0` is no longer sliced into per-expert views for the turbomind path.

### 6. FP16 boundary

The grouped landing should work without changing graph dtypes, but SPRINT-024 keeps one scoped follow-on available:

- cast `src1_sorted` once to FP16 before the grouped call
- keep grouped output in FP16 until the last point where ggml requires FP32

This is explicitly secondary. The sprint should not entangle the main grouped landing with broad dtype-plumbing changes unless the perf profile says it is necessary.

## Implementation

### P0 — Baseline, instrumentation, and invariants

**Goal:** lock the current baseline and make grouped-path execution observable.

1. Record the exact SPRINT-023 comparison baseline:
   - `llama-bench -p 128 -n 32 -r 3`
   - `MIN-8e`, `MIN-16e`, `MIN-32e`
   - `-ot exps=CUDA_TURBOMIND0`
2. Add one-shot debug instrumentation for:
   - grouped path entered
   - `num_experts`, `total_routes`, `active_experts`
   - number of grouped launches per token/layer
3. Add assertions to grouped metadata construction:
   - `sum(tokens_per_expert) == total_routes`
   - `expert_offsets` monotonic
   - `expert_offsets[num_experts] == total_routes`
   - `extra->n_experts == ne02`
4. Add a helper that derives packed `B` and `V` strides from the same converter logic used in `api.cc`.

**P0 gate**

- Baseline table captured in the sprint doc or phase summary
- Instrumentation shows the legacy path is doing one turbomind launch per active expert
- Packed-ld helper reproduces the existing single-expert `api.cc` values for MXFP4 and FP8

### P1 — Grouped helper and turbomind ABI plumbing

**Goal:** make a single grouped turbomind call possible from ggml-cuda.

1. Extend `TmLib` / loader plumbing in `ggml-cuda-turbomind.cu` so `ggml_turbomind_mul_mat_grouped` is resolved and validated at startup.
2. Add `ggml_cuda_mul_mat_grouped_turbomind(...)` in `ggml-cuda-turbomind.{cu,cuh}`.
3. Add a local mirrored `StridedPtr` record for device upload:
   - 16-byte aligned
   - `void * ptr`
   - `int stride`
4. Implement grouped scratch allocation from the ggml CUDA pool for:
   - `A_fp16`
   - `D_fp16`
   - `token_indices_dev`
   - `expert_offsets_dev`
   - `weight_ptrs_dev`
   - `scale_ptrs_dev`
5. For each active expert, populate `weight_ptrs_host` and `scale_ptrs_host` using:
   - `src0->data + e * nb[2]`
   - `extra->scales_dev + e * scales_per_expert`
   - packed strides derived from `k_pack`, dtype, `N`, `K`, `group_size`
6. Upload the metadata and call `ggml_turbomind_mul_mat_grouped(...)`.

**P1 gate**

- A synthetic grouped call with `num_experts > 1` runs without crashing
- One grouped launch produces finite output for both FP8 and MXFP4 fixtures
- `weight_ptrs_dev` and `scale_ptrs_dev` are confirmed to be 16-byte records, not raw pointer arrays

### P2 — `ggml_cuda_mul_mat_id` integration

**Goal:** remove per-expert turbomind launches from the MoE path.

1. In `ggml_cuda_mul_mat_id()`:
   - keep the existing host-side token sort
   - keep `ids_from_sorted` for final restore
   - stop building per-expert `src0_slice` / `src1_slice` / `dst_slice` when `src0` is `CUDA_TURBOMIND`
2. Build `expert_offsets_host` directly from `tokens_per_expert`.
3. Reuse the already-sorted activation buffer:
   - `src1_sorted` remains the logical grouped input
   - grouped helper sees `A = src1_sorted`
   - `token_indices_dev` can point at the sorted token rows or be `NULL` if the helper passes pre-gathered rows; choose one convention and keep it consistent
4. Call the grouped helper once per MoE linear:
   - fused `w1w3`: one grouped launch
   - `w2`: one grouped launch
   - unfused path: three grouped launches total
5. Keep the legacy per-expert path behind a temporary debug switch until grouped correctness is stable.

**P2 gate**

- Debug counters show zero per-expert turbomind launches on the grouped path
- Grouped path executes exactly 2 or 3 MoE GEMM launches per layer, matching fused/unfused topology
- `ids_from_sorted` restore still yields correct tensor shape and no routing corruption

### P3 — Correctness and real-model verification

**Goal:** prove grouped math is equivalent to the existing path and not only fast on broken fixtures.

1. Extend `ggml/vendor/turbomind/test_correctness.cpp`:
   - same packed weights
   - same activations
   - compare grouped vs single-expert loop
   - validate both FP8 and MXFP4
2. Add a focused grouped-routing fixture:
   - `num_experts > 1`
   - non-uniform `expert_offsets`
   - at least one empty expert
3. End-to-end deterministic comparison on a real model that fits:
   - preferred: `DSv4-Flash-AVG-16e`
   - fallback: the smallest non-MIN quantized real-weight variant that fits in 32 GiB
4. Compare grouped TURBOMIND against the legacy TURBOMIND per-expert path first. CPU-MoE remains a secondary reference for overall quality, not the first math gate.

**P3 gate**

- `test_correctness` passes with the existing relative/ULP gate:
  - `rel <= 1e-3`
  - `max_abs <= max(2e-2, 2 x FP16_ULP(max_ref))`
- Deterministic grouped vs legacy TURBOMIND decode matches exactly for at least 32 generated tokens on 3 prompts with `temp=0`, or the mismatch is explained and bounded at the logits level before continuing
- Real model produces coherent output and no NaNs

### P4 — FP16 boundary cleanup (secondary, only if P1-P3 pass)

**Goal:** remove the extra hot-path cast work that grouped dispatch leaves behind.

1. Keep `src1_sorted` in FP16 once per MoE linear instead of converting per expert.
2. Keep grouped output in FP16 until the last required interface boundary.
3. Avoid broad graph rewrites:
   - only touch the MoE FFN routing path
   - keep dense layers and non-TURBOMIND tensors unchanged

**P4 gate**

- Additional decode gain of at least 5% on `MIN-16e`, or a clear profile showing the cast pair was not a meaningful bottleneck
- No new correctness drift relative to P3

If P4 measures below a 3% gain and complicates the graph materially, stop and defer it.

### P5 — Performance landing and profile

**Goal:** decide whether grouped dispatch is shippable on V100.

Benchmark:

- `llama-bench -p 128 -n 32 -r 3`
- `DSv4-Flash-MIN-{8e,16e,32e}`
- V100-SXM2-32GB, same build/toolchain as SPRINT-023

Required tables:

- prompt TPS
- decode TPS
- grouped launch count per decode token
- active experts per token
- VRAM footprint

**Hard perf gates**

- `MIN-16e` decode TPS >= **24.0 t/s**
- `MIN-32e` decode TPS >= **23.5 t/s**
- relative gain >= **1.45x** over SPRINT-023 grouped-disabled TURBOMIND baselines (`16.57` and `16.22` t/s)
- no prompt-TPS regression worse than **5%**

**Stretch gates**

- `MIN-16e` decode TPS >= **28.0 t/s**
- `MIN-32e` decode TPS >= **27.0 t/s**

**Interpretation**

- `>= 24 t/s`: ship grouped dispatch
- `21-24 t/s`: keep the code, profile the remaining bottleneck, and decide whether P4 or a small follow-on is enough
- `< 21 t/s`: grouped launch amortization did not land strongly enough; do not declare success without a precise explanation

Profiling focus:

- kernel launch count reduction
- time spent in FP32<->FP16 conversion
- host-side sort and `cudaMemcpyAsync` overhead for pointer tables / offsets
- grouped GEMM occupancy and DRAM activity

### P6 — Closeout and follow-through

**Goal:** finish with an unambiguous outcome.

If P5 clears the gate:

1. remove temporary debug switches
2. keep one low-noise runtime log or counter for future regression checks
3. document the new default numbers in the sprint summary

If P5 misses the gate:

1. capture a short failure analysis
2. identify whether the next lever is:
   - FP16 boundary
   - host-side sort / metadata movement
   - multi-slot decode
   - something unexpected
3. write the follow-up so SPRINT-025 can start from measured facts

**P6 gate**

- Sprint ends with either a ship decision or a precise follow-up target

## Files Summary

- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - grouped branch in `ggml_cuda_mul_mat_id`
  - removal of per-expert slicing for `CUDA_TURBOMIND`
  - grouped metadata construction

- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu`
  - loader wiring for `ggml_turbomind_mul_mat_grouped`
  - grouped helper implementation
  - packed stride derivation helper
  - transient upload of `StridedPtr` tables

- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh`
  - grouped helper declaration
  - local grouped-launch helper structs if shared with `.cu`

- `ggml/vendor/turbomind/api.cc`
  - only if needed for assertions, comments, or contract clarification around grouped `StridedPtr` handling
  - no API version bump unless the ABI truly changes

- `ggml/vendor/turbomind/include/ggml-turbomind-api.h`
  - only if comments need to say explicitly that grouped `weights_packed` / `scales_packed` are device pointers to `StridedPtr` records, not plain raw-weight pointers

- `ggml/vendor/turbomind/test_correctness.cpp`
  - grouped-vs-legacy correctness tests

- `docs/sprints/SPRINT-024-*.md`
  - phase summaries and final measurement report

## Definition of Done

1. `ggml_cuda_mul_mat_id()` uses grouped turbomind dispatch for `CUDA_TURBOMIND` MoE tensors and no longer launches once per expert.
2. Grouped metadata is explicit and validated:
   - `token_indices`
   - `expert_offsets`
   - per-expert packed-weight and packed-scale `StridedPtr` tables
3. `StridedPtr.stride` is derived from the packed descriptor `ld`, not logical `K`; for the common sm70 HMMA_884 operand-B pack-1 path this means `packed_ld = K * 32`.
4. Correctness tests pass for FP8 and MXFP4 with grouped routing, including empty-expert cases.
5. Real-model deterministic decode runs on a non-MIN model and produces coherent output.
6. V100 perf clears the hard gate:
   - `MIN-16e >= 24.0 t/s`
   - `MIN-32e >= 23.5 t/s`
7. Sprint summary states clearly whether grouped dispatch shipped, and if not, why not.

## Risks

1. **Wrong packed stride in `StridedPtr`**
   This is the most likely correctness bug. Reusing logical `K` instead of packed `ld` will recreate the P2 failure pattern.

2. **ABI ambiguity around grouped pointer tables**
   The grouped C ABI signature currently looks like raw pointer arrays, but the sm70 grouped iterator effectively expects `StridedPtr` records when `ld == 0`. The sprint must make that contract explicit in code comments and tests.

3. **Host-side metadata overhead erodes the launch win**
   Grouped dispatch reduces kernel launches, but if `token_indices`, `expert_offsets`, and per-expert pointer uploads dominate, the gain may flatten out below target.

4. **FP16 boundary work expands scope**
   If grouped dispatch alone misses the bar, it will be tempting to widen the sprint. P4 must stay optional and measured.

5. **Real-model availability**
   If no coherent non-MIN model fits on the hardware, end-to-end quality validation becomes weaker than desired.

## Security

This sprint does not add network-facing functionality, file parsing, or a new public service surface. The relevant security concerns are local correctness and memory safety:

- validate `num_experts`, `expert_offsets`, and pointer-table lengths before launch
- keep `expert_offsets` monotonic and bounded by `total_routes`
- ensure per-expert pointer tables reference only the allocated packed weight and scale regions
- avoid changing the turbomind API version unless the binary contract actually changes

No new secrets, auth paths, or sandbox escapes are involved.

## Dependencies

1. Existing SPRINT-023 turbomind integration:
   - `CUDA_TURBOMIND` buffer type
   - packed upload path
   - working `ggml_turbomind_mul_mat_grouped` export
2. V100 sm70 environment with the current turbomind kernel registry:
   - `Config_MXF4<kColMajor, 0>`
   - `Config_E4M3<kColMajor, 0>`
   - `Config_F16<kColMajor, 0>`
3. A real-weight DSv4 variant that fits in 32 GiB for quality checks
4. Existing benchmark flow:
   - `llama-bench`
   - `test_correctness`
   - optional `ncu` / trace tooling for profile capture

## Open Questions

1. Should the grouped helper pass `token_indices_dev` into turbomind indexed mode, or treat `src1_sorted` as already gathered and pass `NULL` indices? Both can work; one should be chosen and documented to avoid duplicate gather work.
2. Is the current host-side routing sort acceptable for this sprint, or does the metadata upload cost already force a device-side routing follow-on?
3. Which real model is the official quality gate for this sprint: `AVG-16e`, an IQ2 real-weight variant, or something else that is known to fit on the available V100?
4. If grouped dispatch lands at `21-24 t/s`, do we extend immediately into P4, or stop and schedule FP16-boundary cleanup separately?
5. Do we want to formalize the grouped ABI comment so `weights_packed` / `scales_packed` are described as device `StridedPtr` tables instead of generic `void*[]` arrays?

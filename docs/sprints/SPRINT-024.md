# SPRINT-024 — Grouped MoE dispatch + perf landing (V100 sm70)

**Status:** PLANNED 2026-05-15
**Predecessor:** SPRINT-023 (closed 16.2-16.6 t/s decode, 3.0-3.6× over CPU MoE baseline)
**Successor:** SPRINT-025 (multi-slot decode / speculative decoding)

---

## 1. Overview

SPRINT-023 wired turbomind's packed sm70 kernels under a `CUDA_TURBOMIND` buffer type and delivered the per-expert dispatch path. P5 measurement showed throughput is essentially flat in model size (16.2-16.6 t/s across MIN-8e → MIN-32e) — kernel-launch-bound at M=1. At decode rate × top-k active experts × layers × MoE-linears-per-layer that's roughly 4 100 `ggml_turbomind_mul_mat` launches per second.

SPRINT-024 cashes the launch-amortization check that SPRINT-023 deliberately left on the table. The C ABI `ggml_turbomind_mul_mat_grouped` already exists (`ggml/vendor/turbomind/api.cc`); SPRINT-023 P2.3 exercised it at `num_experts=1`. This sprint:

1. Plumbs grouped dispatch through `ggml_cuda_mul_mat_id`, collapsing `n_active_experts × n_moe_linears` launches per layer into 2-3.
2. Verifies output equivalence vs the legacy per-expert path (math gate) and vs CPU MoE on a real-weight model (quality gate).
3. Optionally tightens the FP16 activation boundary if profiling shows the cast pair is still on the hot path.
4. Optionally extends `-ot` to cover FP8 dense layers (stretch) once MoE perf is locked.

### Perf framing (per user interview)

Not a hard t/s ceiling. **Ship if grouped dispatch produces a measured TPS lift over SPRINT-023's 16.6 t/s baseline AND no regression elsewhere; quantify everything in REPORT-18.** This avoids prematurely calling success or failure on a thesis whose realised gain depends on the per-launch cost of the grouped kernel — which we can profile but not yet predict.

Implicit floor: any sprint where grouped path lands but decode TPS drops needs a stop-and-explain, not a quiet ship.

---

## 2. Use Cases

Each phase produces independent value if subsequent phases slip:

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | Baseline locked + instrumentation in place + `num>1` on sm70 verified + sync `cudaMemcpy` overhead measured. |
| P1 | Grouped helper + ABI plumbing land; standalone grouped call works for both quant types. |
| P2 | `mul_mat_id` routes through grouped path for CUDA_TURBOMIND tensors; legacy fallback retained behind a debug switch. |
| P3 | Correctness gate passes (grouped == legacy within FP16 ULP); real-weight model produces coherent output. |
| P4 | FP16 boundary cleanup if profile justified; otherwise stop-loss documented. |
| P5 | REPORT-18: full bench sweep, ship decision, followups for SPRINT-025. |
| P6 | Tag, memory updates, debug switch removed if shipped. |

Stretch (after P5 ship): F8_E4M3_B128 dense layers via the same CUDA_TURBOMIND surface.

---

## 3. Architecture

### 3.1 Where the grouped path slots in

Today `ggml_cuda_mul_mat_id` (`ggml/src/ggml-cuda/ggml-cuda.cu` ~line 2638) does this when `src0->buffer->buft == CUDA_TURBOMIND`:

1. Skip the mmvq/mmq/mmf fast-paths (gated by SPRINT-023 P4 predicate).
2. Sort tokens by expert host-side; emit `ids_to_sorted_host`, `ids_from_sorted_host`, `tokens_per_expert[]`.
3. Slice `src0` per expert and call `ggml_cuda_mul_mat` per expert. Each call lands in `ggml_cuda_mul_mat_turbomind` and emits one `ggml_turbomind_mul_mat` launch.

SPRINT-024 inserts a new branch before step 3:

```
ggml_cuda_mul_mat_id(ctx, dst)
  ├── src0 not CUDA_TURBOMIND? → existing fast-paths / fallback (unchanged)
  └── src0 is CUDA_TURBOMIND
       ├── grouped enabled (default-on)?
       │   └── ggml_cuda_mul_mat_grouped_turbomind(ctx, src0, src1, ids, dst)
       │       (one grouped launch per MoE linear)
       └── grouped disabled (debug switch via env GGML_TM_DISABLE_GROUPED)
           └── existing per-expert slicing (kept through P3 for bisection)
```

No new ggml op. No new buffer type. No public ABI change. The grouped C ABI was designed for this in SPRINT-023 P1.

### 3.2 Per-expert pointer arrays (StridedPtr)

`ggml_turbomind_mul_mat_grouped`'s `weights_packed` / `scales_packed` parameters are declared `const void* const*` in the C ABI but the kernel **reinterprets them** as device-side arrays of `StridedPtr` (16 bytes per expert) when the corresponding `Bdesc.ld == 0` / `Vdesc.ld == 0`. From `matrix_ptr.h:9-13`:

```cpp
struct __align__(16) StridedPtr {
    void* ptr;     //  8 B
    int   stride;  //  4 B
    //              4 B implicit pad to satisfy __align__(16)
};
```

The kernel's resolve path (`matrix_ptr.h:61-97`) loads one `StridedPtr` per `gemm_id` via a 16-byte `__ldg((const uint4*)param.ptr + g)`. Two consequences:

1. **Two separate arrays.** Weights and scales are NOT combined into one record. Each is its own `StridedPtr[num_experts]` on device.
2. **Stride field is the *packed* leading dimension**, not the logical K/N. The same value that goes into `Bdesc.ld` / `Vdesc.ld` for the single-expert path.

#### Stride values for the sm70 path

For `HMMA_884 | OPERAND_B | Pack_M=1` (the only sm70 packed config we ship today):

```
packed_b_ld = K * 32     // From Packing_v2::apply({m,k}) = {m/32, k*32}
                         //   then mk2cs<kRowMajor>(...).x = K * 32
                         //   (memory `turbomind_packed_b_ld_factor.md`)
```

For the scale (`OPERAND_V`):

```
packed_v_ld = post-swap Vdesc.ld   // For sm70 FP8/MXFP4 with conv_s->order = kColMajor
                                    // and OPERAND_V (not OPERAND_U), post-swap:
                                    // rows=K/group_size, cols=N, order=kRowMajor, ld=N.
                                    // → packed_v_ld = N
```

**Neither stride is to be hard-coded.** SPRINT-024 P1 builds a converter-derived helper (mirroring the math already in `api.cc:499-542`) and uses it for both single-expert and grouped paths. The helper signature lives in the same compilation unit so future changes have one source of truth.

#### Per-expert pointers

For each expert `e ∈ [0, n_experts)`:

```cpp
weight_ptrs[e].ptr    = (char *) src0->data + e * src0->nb[2];
weight_ptrs[e].stride = packed_b_ld;       // = K * 32 today
scale_ptrs [e].ptr    = (char *) extra->scales_dev + e * extra->scales_per_expert;
scale_ptrs [e].stride = packed_v_ld;       // = N today
```

**Assert** at first dispatch: `src0->nb[2] >= ggml_turbomind_packed_bytes_per_expert(...)` — guards against future changes to the per-expert allocation layout (F-06).

### 3.3 Pointer-table lifecycle (cached in `extra`)

Experts are static after model load. SPRINT-024 caches both StridedPtr arrays in `ggml_turbomind_tensor_extra` during the first dispatch, so subsequent forward passes skip the host build + H2D upload. New fields:

```cpp
struct ggml_turbomind_tensor_extra {
    // existing
    int    k_pack;
    void * scales_dev;
    size_t scales_bytes;
    size_t scales_per_expert;
    int    group_size;
    int    n_experts;
    // new in SPRINT-024
    void * weight_ptrs_dev;   // device StridedPtr[n_experts] — cached at first dispatch
    void * scale_ptrs_dev;    // device StridedPtr[n_experts] — same
    int    packed_b_ld;       // cached from converter, used on every dispatch
    int    packed_v_ld;       // cached from converter
};
```

Lifetime: same as the parent buffer. Freed alongside the existing `scale_allocs` / `extra_allocs` in `free_buffer`.

### 3.4 Routing metadata (per layer, per forward pass)

`ggml_cuda_mul_mat_id` already builds:

- `ids_to_sorted_host` — token order packed by expert (length `n_tokens × n_expert_used`)
- `ids_from_sorted_host` — inverse map
- `tokens_per_expert[e]`

The grouped helper consumes these as:

- `token_indices_dev` — device int32[total_routes] — copy of `ids_to_sorted_host`
- `expert_offsets_dev` — device int32[n_experts + 1] — exclusive prefix sum of `tokens_per_expert`
- All three uploaded via the ggml CUDA pool (transient per call)

### 3.5 Layer GEMM count

Per MoE FFN layer there are 2 GEMMs when `w1`/`w3` are fused, 3 unfused. Grouped dispatch is **per-MoE-linear**, not per-layer:

- Today: `~n_active_experts × n_linears_per_layer` launches per layer = `6 × 3 ≈ 18` per layer unfused.
- After P2: `n_linears_per_layer` launches per layer = `3` per layer unfused (or 2 fused).

At 43 layers × decode rate, that's ~6× fewer launches at unfused and ~9× at fused.

### 3.6 Sync `cudaMemcpy` at api.cc:607

Each `ggml_turbomind_mul_mat_grouped` call currently does a blocking `cudaMemcpy` to read `expert_offsets[num_experts]` for `total_tokens`. If grouped dispatch fires N times per token (one per MoE linear × number of layers), this creates N stream sync points per token — a latent perf footgun that survives launch-count-only analysis.

P0.4 measures the cost. P1 either:

(a) Eliminates the read by passing `total_tokens` as an explicit C-ABI parameter (small ABI extension; bump `ggml_turbomind_api_version`), or

(b) Keeps it but ensures the call happens once per layer in a stream-ordered way that the surrounding graph already syncs around.

(a) is preferred — gives us a deterministic, async path.

### 3.7 FP16 boundary (P4, secondary)

Today the cast pair in `ggml_cuda_mul_mat_turbomind` runs FP32→FP16 (A) then FP16→FP32 (D) on every dispatch. After P2 the cast still runs once per grouped call (2-3× per layer instead of 18×) but is still hot. P4 moves the boundary upstream so the cast is once per token per MoE block:

- Cast `src1_sorted` to FP16 once before the grouped call(s) for the layer.
- Keep grouped output in FP16 until the inverse gather; cast back at the last step.

Only touches the MoE FFN path, not the full graph.

### 3.8 Stretch — F8_E4M3_B128 dense layers

Pack pipeline already supports F8_E4M3_B128 (SPRINT-023 P2.3). The only missing wiring is dispatch through `ggml_cuda_mul_mat` (not `mul_mat_id`) when the user pins dense tensors with `-ot 'pattern=CUDA_TURBOMIND0'`. SPRINT-023 P4 already has the predicate at the top of `ggml_cuda_mul_mat`; SPRINT-023 already verified that single-matmul works on these types.

What's missing: a `-ot` regex that matches the dense tensors of interest (`attn_q.weight`, `attn_kv_a.weight`, `attn_kv_b.weight`, `attn_o.weight` for DSv4). Stretch tests on the AVG-16e or IQ2-64e model with `-ot 'attn_(q|kv_(a|b)|o).weight=CUDA_TURBOMIND0'`.

---

## 4. Implementation

### P0 — Baseline + instrumentation + invariants (1 day)

**Goal:** lock the comparison point and make the grouped path observable before any changes.

1. **P0.1** — Rebuild `sprint-022-dsv4-integration` HEAD; re-run `llama-bench -p 128 -n 32 -r 3` on MIN-8e / MIN-16e / MIN-32e. Confirm 16.2-16.6 t/s baseline reproduces within 5%. Capture as baseline table for REPORT-18. **Halt if it doesn't reproduce.**
2. **P0.2** — Add an env-gated debug counter in `ggml_cuda_mul_mat_turbomind` (and later the grouped helper): print N grouped vs N per-expert launches, total_tokens, active_experts per token. Plumbed through `GGML_TM_VERBOSE=1`.
3. **P0.3** — Standalone `num_experts > 1` smoke on the existing C ABI: extend `ggml/vendor/turbomind/test_correctness.cpp` to pack the same weight `N` times (`N ∈ {2, 6, 8}`) and call `ggml_turbomind_mul_mat_grouped` directly with an identity routing. Verify finite output, no NaN, no rc != 0. **Halt if the sm70 grouped path doesn't accept `num > 1`** — the whole sprint thesis depends on it.
4. **P0.4** — Microbench the sync `cudaMemcpy` at api.cc:607 by timing 1000 back-to-back grouped calls with `n_active_experts = 6`. Document the per-call overhead. If > 50 µs, mandates the C ABI extension in P1.

**P0 Gate**:
- ✅ Baseline reproduced within 5%
- ✅ Instrumentation prints expected counters under `GGML_TM_VERBOSE=1`
- ✅ `num > 1` grouped call succeeds on sm70 for both FP8 and MXFP4
- ✅ Sync-memcpy cost documented; P1 path chosen (ABI extend vs leave)

### P1 — Grouped helper + ABI plumbing (3 days)

**Goal:** make a single grouped turbomind call possible from ggml-cuda, end to end.

1. **P1.1** — `dlsym` `ggml_turbomind_mul_mat_grouped` in `ggml-cuda-turbomind.cu`. Add `pfn_mul_mat_grouped` field to `TmLib`. Resolve at startup; assert non-NULL.
2. **P1.2** — Build the converter-derived `packed_ld` helper (single source of truth used by both single-expert and grouped paths). Locate either in `api.cc` (exported via a new C-ABI function) or as a private static in `ggml-cuda-turbomind.cu` that replicates the math. Decision in P1 review; default to the latter to avoid ABI bump.
3. **P1.3** — Extend `ggml_turbomind_tensor_extra` with `weight_ptrs_dev`, `scale_ptrs_dev`, `packed_b_ld`, `packed_v_ld`. Populate in `set_tensor` (one-shot at upload); free in `free_buffer`.
4. **P1.4** — Optional ABI extension if P0.4 mandates: add `total_tokens` as an explicit parameter to `ggml_turbomind_mul_mat_grouped`. Bump `ggml_turbomind_api_version`. Keep backward-compat shim that reads `expert_offsets[num_experts]` only if `total_tokens < 0`.
5. **P1.5** — Add `ggml_cuda_mul_mat_grouped_turbomind` signature to `ggml-cuda-turbomind.cuh`:
   ```cpp
   void ggml_cuda_mul_mat_grouped_turbomind(
       ggml_backend_cuda_context & ctx,
       const struct ggml_tensor * src0,   // expert weights, 3D, ne[2] = n_experts
       const struct ggml_tensor * src1,   // activations FP32 row-major
       const struct ggml_tensor * ids,    // routing ids
       struct ggml_tensor       * dst);   // output FP32 row-major
   ```
6. **P1.6** — Implement in `ggml-cuda-turbomind.cu`:
   - Build `tokens_per_expert[]`, `ids_to_sorted_host[]`, `ids_from_sorted_host[]`, `expert_offsets_host[]` host-side (same as existing per-expert path).
   - Upload `expert_offsets_dev`, `token_indices_dev` via `ggml_cuda_pool_alloc`.
   - Allocate `A_fp16 [total_routes × K]` and `D_fp16 [total_routes × N]` from the pool.
   - Permute + cast in a single fused kernel: read `src1` rows by `ids_to_sorted`, write `A_fp16` contiguously.
   - Call `ggml_turbomind_mul_mat_grouped(A_fp16, /*token_indices=*/nullptr, expert_offsets_dev, n_experts, weight_ptrs_dev, scale_ptrs_dev, ggml_type, N, K, group_size, k_pack, D_fp16, stream)`. `nullptr` for token_indices because A is pre-permuted.
   - Cast + unpermute: read `D_fp16`, scatter to `dst` by `ids_from_sorted_dev`.
7. **P1.7** — Synthetic harness in `test_correctness.cpp`: pack the same weight as N experts, run grouped vs the existing single-expert path on the same activations. Gate: same FP16-ULP tolerance as SPRINT-023 P2.3 (`rel ≤ 1e-3`, `max_abs ≤ max(2e-2, 2 × FP16_ULP(max_ref))`). Test `N ∈ {2, 6, 8}` and an empty-expert fixture (one expert routed 0 tokens).

**P1 Gate**:
- ✅ `dlsym` resolves `ggml_turbomind_mul_mat_grouped`
- ✅ Standalone grouped call produces finite output on FP8 and MXFP4 for `N ∈ {2, 6, 8}`
- ✅ `test_correctness.cpp` extension passes (grouped == single-expert within FP16 ULP)
- ✅ Empty-expert fixture passes
- ✅ Pointer-table caching exercised (counter shows H2D upload happens once per buffer)

### P2 — `mul_mat_id` integration (2 days)

**Goal:** route MoE through the grouped helper end-to-end.

1. **P2.1** — In `ggml_cuda_mul_mat_id`, after the existing TURBOMIND predicate but before the per-expert slicing loop: branch to `ggml_cuda_mul_mat_grouped_turbomind`. Per-expert slicing path remains, gated behind `GGML_TM_DISABLE_GROUPED=1`.
2. **P2.2** — Validate launch counts at runtime under `GGML_TM_VERBOSE`: expect exactly 2 or 3 turbomind launches per MoE-FFN layer (matching fused/unfused `w1w3`/`w2` topology), zero per-expert launches.
3. **P2.3** — Smoke load MIN-16e with `-ot 'exps=CUDA_TURBOMIND0'`; verify generation runs to completion with grouped path, no crashes.

**P2 Gate**:
- ✅ Debug counter shows zero per-expert launches under default config
- ✅ Generation runs to completion on MIN-16e
- ✅ `GGML_TM_DISABLE_GROUPED=1` still works (legacy bisection path)

### P3 — Correctness + non-MIN quality verification (2 days)

**Goal:** prove grouped is equivalent to legacy, and prove the path produces coherent output on real weights.

1. **P3.1** — Greedy decode 32 tokens, `temp=0`, fixed seed, fixed prompt set of 10 prompts (mix of chat / code / math) on `DSv4-Flash-MIN-16e` with three configurations:
   - `GGML_TM_DISABLE_GROUPED=1` (per-expert TURBOMIND — SPRINT-023 P4 path)
   - default (grouped TURBOMIND — SPRINT-024 P2 path)
   - `-ot 'exps=CPU'` (CPU MoE — SPRINT-022 baseline)

   Gate: **grouped vs legacy turbomind** ≥ 95% leading-token match (these should be near-bit-identical: same kernel, different launch granularity). **grouped vs CPU MoE** ≥ 75% leading-token match (FP16 quantization noise dominates).
2. **P3.2** — Same procedure on **DSv4-Flash-AVG-16e** (real weights, 18 GiB). Primary quality gate.
3. **P3.3** — Same procedure on **DeepSeek-V4-Flash-IQ2-64e** (28 GiB). Secondary signal; if it doesn't fit with full `-ot exps=`, use partial offload (`-ncmoe` for the layers that don't fit on TURBOMIND).
4. **P3.4** — Sanity check NaN/Inf in output logits at 8 random positions across the run. None.

**P3 Gate**:
- ✅ grouped vs legacy turbomind ≥ 95% on MIN-16e
- ✅ grouped vs CPU MoE ≥ 75% on AVG-16e (primary)
- ✅ IQ2-64e produces coherent output (looser gate: ≥ 60% match, gibberish absent)
- ✅ No NaN/Inf in checked logit positions

### P4 — FP16 boundary cleanup (secondary, stop-loss)

**Goal:** remove redundant FP32↔FP16 cast pair if it's still on the hot path after grouped landed.

1. **P4.1** — `ncu` profile on a 16-token decode of MIN-16e under the grouped path. Measure `cudaLaunchKernel` overhead, HMMA active %, time spent in the cast kernels (`convert.cu` produces them).
2. **P4.2** — If cast pair is < 3% of decode time, **stop** — document in REPORT-18, defer F-02 to SPRINT-025.
3. **P4.3** — Otherwise:
   - Cast `src1_sorted` to FP16 once before the layer's first grouped call.
   - Keep grouped output in FP16 across `w1w3` → swiglu → `w2` (if SwiGLU also supports FP16 input/output).
   - Cast back to FP32 at the inverse-gather step.
4. **P4.4** — Re-bench MIN-16e. Gate: ≥ 5% additional decode TPS over P2 numbers; else stop and document.

**P4 Gate** (only fires if P4.3 ran):
- ✅ Additional ≥ 5% decode TPS on MIN-16e
- ✅ No new correctness drift vs P3 (re-run the P3.1 grouped-vs-legacy gate)

### P5 — Measurement + REPORT-18 (1-2 days)

**Goal:** decision-complete narrative on the launch-amortization thesis.

1. **P5.1** — `llama-bench -p 128 -n 32 -r 3` on MIN-8e / MIN-16e / MIN-32e:
   - SPRINT-023 baseline (re-run for fresh comparison) — three rows
   - SPRINT-024 grouped path — three rows
   - SPRINT-024 grouped + FP16 boundary (if P4 ran) — three rows
2. **P5.2** — `ncu` metric pack on the new grouped path:
   - `cudaLaunchKernel` overhead per dispatch (was the dominant cost — did it actually drop?)
   - HMMA active % (the math regime — should rise)
   - DRAM bytes/sec (the bandwidth regime)
3. **P5.3** — Write `docs/sprints/SPRINT-024-REPORT-18.md`. Required sections:
   - Bench table (with deltas vs SPRINT-023 P5)
   - Launch-count delta (counted via instrumentation)
   - Decision: SHIP / EXTEND / STOP, with rationale
   - If SHIP: SPRINT-025 starting line
   - If EXTEND: precise next-lever identified, scoped, sized
   - If STOP: explanation of why launch amortization didn't pay off

**P5 Gate**:
- ✅ All three bench rows captured
- ✅ ncu metric pack captured
- ✅ REPORT-18 ends with a SHIP / EXTEND / STOP verdict
- ✅ Followups doc filed for SPRINT-025 if applicable

### P6 — Close-out (0.5 day)

**Goal:** lock in the ship state or the diagnosis.

1. If SHIP:
   - Remove `GGML_TM_DISABLE_GROUPED` debug switch (or keep behind `#ifdef` for future bisection).
   - Update the SPRINT-022 baseline numbers in `docs/sprints/SPRINT-024-REPORT-18.md` to the new defaults.
   - Memory updates: add a `grouped_moe_landed.md` entry.
2. If EXTEND:
   - Keep debug switch, document its purpose.
   - File `SPRINT-024-FOLLOWUPS.md` with measured next-lever target.
3. If STOP:
   - Keep grouped path behind opt-in env (default-off).
   - File `SPRINT-024-FOLLOWUPS.md` with the failure analysis.
4. Tag `sprint-024-close` regardless.

### P7 — Stretch — F8_E4M3_B128 dense layers (0.5-1 day, only if P5 shipped + time)

**Goal:** extend the win to non-expert linears.

1. **P7.1** — Smoke-load AVG-16e with `-ot 'attn_(q|kv_a|kv_b|o).weight=CUDA_TURBOMIND0,exps=CUDA_TURBOMIND0'`. Verify dense tensors pack + dispatch.
2. **P7.2** — Compare decode TPS vs MoE-only TURBOMIND config. Document delta.
3. **P7.3** — Add to REPORT-18 as an appendix.

---

## 5. Files Summary

### Modified

| Path | Change |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` | Grouped helper impl; loader plumbing for `mul_mat_grouped`; pointer-table caching in `set_tensor`; FP16 cast plumbing for P4. |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` | Grouped helper signature; extended `ggml_turbomind_tensor_extra`. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_mul_mat_id` predicate routing to grouped helper; debug switch for legacy fallback. |
| `ggml/vendor/turbomind/test_correctness.cpp` | Grouped vs single-expert ULP gate; empty-expert fixture; N ∈ {2,6,8} sweep. |

### New

| Path | Purpose |
|---|---|
| `docs/sprints/SPRINT-024-REPORT-18.md` | Measurement narrative + ship decision. |
| `docs/sprints/SPRINT-024-P{0,1,2,3,4,5,6}-summary.md` | Per-phase summaries (same pattern as SPRINT-023). |
| `docs/sprints/SPRINT-024-FOLLOWUPS.md` | If EXTEND or STOP outcomes apply. |

### Possibly modified (P1.4 conditional)

| Path | Change |
|---|---|
| `ggml/vendor/turbomind/include/ggml-turbomind-api.h` | Add `total_tokens` parameter to `ggml_turbomind_mul_mat_grouped`; bump `GGML_TURBOMIND_API_VERSION`. |
| `ggml/vendor/turbomind/api.cc` | Honor explicit `total_tokens`; fall back to sync memcpy only if `< 0`. |

---

## 6. Definition of Done

1. ✅ `ggml_cuda_mul_mat_id` routes CUDA_TURBOMIND MoE tensors through the grouped path by default.
2. ✅ `dlsym` for `ggml_turbomind_mul_mat_grouped` succeeds at startup.
3. ✅ `ggml_turbomind_tensor_extra` cached StridedPtr arrays + packed lds; H2D upload happens once per tensor.
4. ✅ `StridedPtr.stride` is derived from converter resolution, not hard-coded; both `packed_b_ld` and `packed_v_ld` correct for FP8 and MXFP4 on sm70.
5. ✅ Empty-expert fixture passes.
6. ✅ Correctness gate: grouped vs legacy turbomind ≥ 95% on MIN-16e greedy-decode 32-token completions.
7. ✅ Quality gate: grouped TURBOMIND vs CPU MoE ≥ 75% leading-token match on DSv4-Flash-AVG-16e.
8. ✅ IQ2-64e produces coherent output (≥ 60% match).
9. ✅ Debug counter exposes launch counts under `GGML_TM_VERBOSE=1`.
10. ✅ REPORT-18 contains: bench table, launch-count delta, SHIP / EXTEND / STOP verdict.
11. ✅ No regression in `ggml/vendor/turbomind/test_correctness.cpp`.
12. ✅ Tag `sprint-024-close`.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Wrong `packed_*_ld` value in StridedPtr | Medium | High | Single converter-derived helper; assert against single-expert path values at P1.7 |
| 2 | `cudaMemcpy` sync at api.cc:607 dominates the launch budget | Medium | Medium | P0.4 measures; P1.4 extends ABI if > 50 µs |
| 3 | `num_experts > 1` on sm70 turbomind registry has hidden ragged-batch bug | Low-Medium | High | P0.3 standalone smoke before any integration work |
| 4 | Empty-expert (zero tokens routed) crashes in `Bdesc.num` or `Adesc.offsets` resolve | Medium | Medium | P1.7 fixture; P2 generation-loop smoke catches in practice |
| 5 | Per-launch metadata upload (`expert_offsets`, `token_indices`) erodes the launch win | Low | Medium | Profile in P0.4; switch to one-shot dev allocation pool if needed |
| 6 | DSv4-Flash-AVG-16e unavailable or doesn't fit | Low | Medium | IQ2-64e fallback (already in DoD); LOG if no real-weight model fits within 32 GiB |
| 7 | FP16 boundary work (P4) leaks into the rest of the graph | Medium | Medium | Stop-loss: < 3% gain → defer; stay within MoE FFN sandwich |
| 8 | Stretch dense FP8 (P7) breaks pack pipeline for non-expert tensors | Low | Low | One-shot smoke; revert `-ot` regex if any regression |

---

## 8. Security

No new network-facing surface. Local-only correctness and memory safety:

- Validate `num_experts`, `expert_offsets[num_experts]`, pointer-table lengths before launch (already done by `api.cc`; SPRINT-024 adds asserts on the caller side).
- Keep `expert_offsets` monotonic; bound by `total_routes`.
- Ensure cached pointer arrays in `extra` reference only allocated weight + scale regions (caught at `free_buffer` if the buffer was dropped before the extra; SPRINT-024 P1 keeps the existing lifetime tracking).
- No new secrets, no new sandbox surface.

ABI extension in P1.4 (conditional) bumps `ggml_turbomind_api_version` — surface change is internal to this private fork only.

---

## 9. Dependencies

1. SPRINT-023 turbomind integration: `CUDA_TURBOMIND` buffer type, pack pipeline, per-expert dispatch (legacy fallback).
2. V100 sm70 environment, CUDA 12.2, gcc 11.4, cmake 3.22 — same as SPRINT-023.
3. Existing turbomind sm70 kernel registry: `Config_E4M3<kColMajor, 0>`, `Config_MXF4<kColMajor, 0>` — P0.3 verifies these accept `num > 1`.
4. Real-weight model: DSv4-Flash-AVG-16e (preferred) + IQ2-64e (secondary).
5. Benchmark tooling: `llama-bench`, `ggml/vendor/turbomind/test_correctness.cpp`, `ncu` for P4/P5 profiling.

---

## 10. Open Questions

1. **`packed_ld` helper location** — `api.cc` (with new C-ABI accessor) vs private static in `ggml-cuda-turbomind.cu`? Decision in P1.2. Default to the latter; revisit if the same math appears in a third call site.
2. **ABI extension in P1.4 vs leaving the sync memcpy** — depends on P0.4 measurement. > 50 µs/call → extend; else leave.
3. **`token_indices` passing convention** — `nullptr` (pre-permuted A) vs explicit indices. P1.6 picks the pre-permute path; documented in code.
4. **Stretch dense FP8** — does the AVG-16e GGUF actually have dense FP8 tensors named `attn_*.weight`? Verify before P7.
5. **CUDA graph compatibility** — does the grouped path play nicely with `GGML_CUDA_USE_GRAPHS`? Probably yes (no stream sync if P1.4 lands) but verify.

---

## 11. Outcome contract

Per user interview: this sprint ships if **grouped dispatch produces a measured TPS lift over SPRINT-023's 16.6 t/s baseline with no regression elsewhere**, quantified in REPORT-18. The decision is explicit and recorded; "ship anyway with a perf regression" is not on the table.

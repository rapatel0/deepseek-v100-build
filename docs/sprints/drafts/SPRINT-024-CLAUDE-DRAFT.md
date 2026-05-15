# SPRINT-024 — Grouped MoE dispatch + FP16 boundary + quality verification (V100 sm70)

**Status:** DRAFT 2026-05-15
**Predecessor:** SPRINT-023 (closed at 16.2–16.6 t/s decode, 3.0–3.6× over CPU MoE baseline; throughput flat in model size — launch-bound at M=1)
**Successor:** SPRINT-025 (multi-slot / speculative decode, if grouped MoE doesn't lift the ceiling far enough)

---

## 1. Overview

SPRINT-023 wired turbomind's packed-weight sm70 kernels under a `CUDA_TURBOMIND`
buffer type and delivered the per-expert dispatch path. P5 measurement showed
the bottleneck is now per-kernel launch overhead at M=1: throughput is flat in
model size across MIN-8e → MIN-32e at ~16.5 t/s decode, with ~4 100 small
`ggml_turbomind_mul_mat` launches/sec (6 active experts × 43 layers × 16 t/s).

SPRINT-024 cashes the launch-amortization check that SPRINT-023 deliberately
left on the table. The C ABI `ggml_turbomind_mul_mat_grouped` already exists
(`ggml/vendor/turbomind/api.cc:566`) and was exercised at num_experts=1 during
P2.3 debug. This sprint:

1. **P1** wires the grouped C ABI through `ggml_cuda_mul_mat_id` — collapses
   N_active_experts launches per MoE layer into one.
2. **P2** verifies output quality on a non-MIN model (MIN-* fixtures have
   intentionally broken expert weights).
3. **P3** moves the FP16 activation boundary upstream so the FP32↔FP16 cast
   pair runs once per token instead of once per (expert × layer).
4. **P4** profiles + tunes any remaining hot spots.
5. **P5** measurement + REPORT-18.
6. **P6** close-out.

**Hard perf gate:** decode TPS on MIN-16e (or MIN-32e) ≥ **24 t/s**
(≥ 1.45× over SPRINT-023's 16.6 t/s baseline). Stretch **30 t/s**.

---

## 2. Use Cases

Each phase produces independent value if subsequent phases slip:

| Phase | Useful output if sprint stops here |
|---|---|
| P1 | Grouped dispatch lands; decode TPS lifts ≥ 1.4× even without quality test or FP16-boundary work. |
| P2 | Quality verification on real expert weights — closes the SPRINT-022 question "does the V100 path produce coherent tokens." |
| P3 | FP16 boundary moves upstream — additional ~5–10% TPS, plus useful for any future fused-FFN work. |
| P4 | ncu metric pack + REPORT-18 baseline data: HMMA active %, launch overhead per dispatch — informs SPRINT-025 multi-slot design. |
| P5 | TPS table vs SPRINT-023, gate evaluation, recommendations. |
| P6 | Tag, follow-ups, memory updates. |

---

## 3. Architecture

### 3.1 Where the grouped path slots in

The current `ggml_cuda_mul_mat_id` (in `ggml/src/ggml-cuda/ggml-cuda.cu:2638`)
does this when `src0->buffer->buft` is `CUDA_TURBOMIND`:

1. Skip the mmvq/mmq/mmf fast-paths (lines 2654–2683): turbomind tensors don't
   match those kernels' format.
2. Sort tokens by expert host-side (lines 2701–2730).
3. Issue `ggml_cuda_mul_mat` per expert with a slice (lines 2748–2793) — each
   call lands in `ggml_cuda_mul_mat_turbomind` and emits one
   `ggml_turbomind_mul_mat` launch.

SPRINT-024 P1 inserts a new branch *before* step 2 when the per-token-sort
metadata can be replayed on-device as the grouped contract:

```
ggml_cuda_mul_mat_id(ctx, dst)
  ├── src0 is on CUDA_TURBOMIND buffer?
  │   ├── No  → existing fast-path / per-expert slicing (unchanged)
  │   └── Yes
  │       ├── new path: ggml_cuda_mul_mat_grouped_turbomind(ctx, src0, src1, ids, dst)
  │       │   ├── permute_tokens_by_expert    (device kernel)
  │       │   ├── build per-expert StridedPtr array on device
  │       │   ├── ggml_turbomind_mul_mat_grouped(...)
  │       │   └── unpermute_tokens
  │       └── (no fallback — predicate is total)
```

No new ggml op, no new buffer type, no C ABI change. The C ABI was designed
for this in SPRINT-023 P1.

### 3.2 Per-expert pointer array layout (StridedPtr)

Critical detail from SPRINT-023 P2.3 + memory `turbomind_packed_b_ld_factor.md`:

When `Bdesc.ld == 0` (the grouped contract), turbomind's `resolve<T, kBlocked>`
in `research/lmdeploy/src/turbomind/kernels/gemm/matrix_ptr.h:62` does:

```cpp
StridedPtr ptr{param.ptr, param.stride};
if (ptr.stride == 0) {
    (uint4&)ptr = __ldg((const uint4*)param.ptr + g);
}
```

i.e. it loads 16 bytes (`{void* ptr; int stride; int pad;}`) per gemm_id `g`
from `param.ptr`. So `weights_packed` is **not** a `void**` — it's a
device-side array of `StridedPtr` records.

**StridedPtr layout** (`research/lmdeploy/src/turbomind/kernels/gemm/matrix_ptr.h:9`):

```cpp
struct __align__(16) StridedPtr {
    void* ptr;     //  8 B — pointer to packed expert weight
    int   stride;  //  4 B — packed leading dimension (NOT logical K)
    // 4 B implicit pad to satisfy __align__(16)
};
```

**The stride field is the *packed* ld**, not the logical K. For HMMA_884
OPERAND_B with `Pack_M=1` (the sm70 path) the rule from
`turbomind_packed_b_ld_factor.md` is:

```
packed_ld = K * 32   // OPERAND_B Pack_M=1 packs cols by ×32 along K
```

For the FP8 / MXFP4 / FP16 cases the `MatrixLayout::ld` is bits per row /
`bitsof<T>` in the resolve math (line 74) — for our HMMA_884 OPERAND_B path
the dispatcher passes `stride = K * 32` so element-pointer arithmetic on
expert offsets resolves correctly when `param.offsets` is non-NULL.

For the scale array `scales_packed` the analogous rule applies but with
`Vdesc.ld` derived from `K / group_size` rather than `K`. The scale stride
in each `StridedPtr` is `(K / group_size) * sV` where `sV` is the V-operand
pack factor reported by `tmg::GetConverters(...)[1]->pack`. The dispatcher
queries this once per type at init and caches it.

### 3.3 Device-side metadata

Per layer, per forward pass, the grouped path needs:

| Buffer | Type | Length | Source | Lifetime |
|---|---|---|---|---|
| `A_fp16` | half | `total_tokens × K` | cast of permuted `src1` rows | scratch (per call) |
| `D_fp16` | half | `total_tokens × N` | output of grouped GEMM | scratch (per call) |
| `expert_offsets` | int32 | `num_experts + 1` | prefix-sum of `tokens_per_expert` | scratch (per call) |
| `token_indices` | int32 | `total_tokens`, may be NULL | original row indices for gather; NULL if we pre-permute `A` | scratch (per call) |
| `weights_packed` | `StridedPtr` | `num_experts` | rebuilt from `src0->extra` + base ptr | scratch (per call), one-shot init OK if base pointers stable |
| `scales_packed` | `StridedPtr` | `num_experts` | rebuilt from `src0->extra` (`scales_dev + i * scales_per_expert`) | as above |

Total scratch: O(num_experts × 32 B + total_tokens × (K+N) × 2 B + total_tokens × 4 B).
At decode (M=1, total_tokens = `n_expert_used` = 6), this is dominated by the
activation buffers and is small (~250 KiB at K=N=7168).

### 3.4 Build-time tensor-extra reuse

`ggml_turbomind_tensor_extra` (in `ggml-cuda-turbomind.cuh:35`) already
holds:

- `k_pack` — encoded `{b_pack[0:12], v_pack[12:24]}`
- `scales_dev` — base of the per-expert scales buffer
- `scales_per_expert` — byte stride between per-expert scales
- `group_size`
- `n_experts`

The grouped dispatch helper rebuilds `weights_packed[i].ptr =
src0->data + i * src0->nb[2]` and `scales_packed[i].ptr =
extra->scales_dev + i * extra->scales_per_expert`. These pointers are stable
for the life of the tensor — we can cache the device-side StridedPtr array
in the extra struct at first dispatch and reuse it on every forward pass.

To keep P1 simple we build a fresh StridedPtr array per call into a pool
allocation; P4 considers caching if profiling shows the build is non-trivial.

### 3.5 FP16 boundary (P3)

Current path (after SPRINT-023):

```
mul_mat_id → permute (FP32) → ggml_cuda_mul_mat_turbomind
                                    cast FP32→FP16 (per expert call)
                                    ggml_turbomind_mul_mat (FP16 in, FP16 out)
                                    cast FP16→FP32 (per expert call)
             ← unpermute (FP32)
```

After P3, the cast pair runs once per token at the FFN router input/output
instead of `2 × n_active_experts × n_layers` times per token:

```
upstream-of-ffn cast FP32→FP16  (once per token per layer)
mul_mat_id → permute (FP16) → ggml_cuda_mul_mat_grouped_turbomind
                                    ggml_turbomind_mul_mat_grouped  (FP16 in, FP16 out)
             ← unpermute (FP16)
downstream cast FP16→FP32  (once per token per layer)
```

This is bounded — we don't push FP16 across the whole graph, just across the
FFN sandwich. The activation byte traffic to/from DRAM halves on the hot path.

### 3.6 Quality verification (P2)

Per `dsv4_flash_min_models_are_garbage.md`: MIN-* fixtures cannot be used for
quality. Two candidate models:

- **DSv4-Flash-AVG-16e** (18 GiB, real weights — per user) — preferred.
- **DeepSeek-V4-Flash-IQ2-64e** (28 GiB) — fallback if AVG-16e is unavailable.

Gate: greedy decode 32 tokens against CPU MoE baseline on a small fixed
prompt set; ≥ 75 % leading-token match across the set. Same model + seed +
temperature 0 on both legs.

---

## 4. Implementation

### P0 — Plumbing audit + microbench (1 day, ship-blocker for P1+)

1. **P0.1** — Build current `main` of branch `sprint-022-dsv4-integration` on
   gpu-01, re-run `llama-bench -p 128 -n 32 -r 2` on MIN-8e / MIN-16e / MIN-32e.
   Confirm 16.2–16.6 t/s baseline reproduces within 5 %. If not, halt and
   diagnose — the gate is relative to this measured number.
2. **P0.2** — `ncu --kernel-name "sm70_*" --metrics
   sm__cycles_active.avg.pct_of_peak_sustained_elapsed,
   dram__bytes_read.sum,gpu__time_duration.sum,launch__grid_size` on a 16-token
   decode (`-p 0 -n 16`) of MIN-16e. Capture per-launch `cudaLaunchKernel`
   overhead and HMMA active %. Document in `tools/tc-grid/docs/REPORT-18.md`
   first draft.
3. **P0.3** — Sanity-check `ggml_turbomind_mul_mat_grouped` with num_experts=1
   in a standalone unit (extend `ggml/vendor/turbomind/test_correctness.cpp`)
   against `ggml_turbomind_mul_mat`. Must be bit-identical (same `Gemm::Run`
   path under both wrappers).

**P0 Gate**:
- ✅ Baseline reproduced within 5 %
- ✅ Launch overhead ≥ 60 % of per-dispatch time documented (or we halt and
  rethink — the whole sprint thesis depends on this)
- ✅ Grouped C ABI at num_experts=1 passes correctness vs single-expert ABI

### P1 — Grouped MoE dispatch (4 days)

1. **P1.1** — Add the dispatch helper signature to
   `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh`:

   ```cpp
   void ggml_cuda_mul_mat_grouped_turbomind(
       ggml_backend_cuda_context & ctx,
       const struct ggml_tensor * src0,   // expert weights (3D, ne[2] = n_experts)
       const struct ggml_tensor * src1,   // activations (FP32 row-major)
       const struct ggml_tensor * ids,    // routing ids
       struct ggml_tensor       * dst);   // output (FP32 row-major)
   ```

   Place in same compilation unit as the existing per-expert helper to share
   FP16/FP32 cast plumbing.

2. **P1.2** — Implement in `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu`:
   - Read `ids` host-side (same memcpy pattern as the existing
     `ggml_cuda_mul_mat_id` at lines 2712–2714).
   - Build `tokens_per_expert[]` and `expert_offsets[]` host-side (same as
     existing path, lines 2716–2730), then upload `expert_offsets` to device
     via the ggml CUDA pool.
   - Allocate `A_fp16` device buffer of `total_tokens × K × sizeof(half)` and
     `D_fp16` of `total_tokens × N × sizeof(half)`. Use `ggml_cuda_pool_alloc`.
   - Permute + cast: a single fused kernel
     `permute_and_cast_f32_to_f16` reads `src1` rows in the order dictated by
     `ids_to_sorted` and writes contiguous `A_fp16`. Reuses the
     `ids_to_sorted_host` logic from `ggml_cuda_mul_mat_id`.
   - Build `StridedPtr` arrays (host first, copy to device pool):
     ```cpp
     for (int i = 0; i < n_experts; ++i) {
         w_arr[i].ptr    = (char*)src0->data + i * src0->nb[2];
         w_arr[i].stride = K * 32;            // OPERAND_B Pack_M=1 on sm70
         s_arr[i].ptr    = (char*)extra->scales_dev + i * extra->scales_per_expert;
         s_arr[i].stride = (K / extra->group_size) * sV;  // sV cached at init
     }
     ```
   - Call `ggml_turbomind_mul_mat_grouped(A_fp16, /*token_indices=*/nullptr,
     expert_offsets_dev, n_experts, w_dev, s_dev, ggml_type, N, K, group_size,
     extra->k_pack, D_fp16, stream)`.
   - Cast + unpermute: `cast_and_unpermute_f16_to_f32` reads `D_fp16` and
     scatters to `dst` rows by `ids_from_sorted`.
   - Add one-time `fprintf(stderr, "[cuda-turbomind] grouped dispatch active, n_experts=%d, total_tokens=%d\n", ...)` behind a `GGML_TM_VERBOSE` env check.

3. **P1.3** — Hook into `ggml_cuda_mul_mat_id`
   (`ggml/src/ggml-cuda/ggml-cuda.cu`, ~line 2654): if `is_turbomind` is
   true, dispatch to the new helper and return. The existing per-expert
   slicing path (lines 2685–2793) stays as a fallback for when the new
   helper rejects (e.g., unsupported type, num_experts == 0).

4. **P1.4** — Extend `ggml/vendor/turbomind/test_correctness.cpp`:
   pack the same weight as `N` experts (identical data), run the
   single-expert ABI against the grouped ABI with the same activations,
   compare D within **FP16 ULP** at production tolerance
   (relative ≤ 2e-2 + 2-ULP soft floor — same gate as P2.3).

   Pin N ∈ {2, 6, 8, 16} to cover top-k (DSv4 uses top-8) and the 16e/32e
   load.

5. **P1.5** — Run `llama-bench -p 128 -n 32 -r 3` on MIN-8e / MIN-16e /
   MIN-32e. Capture decode TPS; assess against the P1 perf gate below.

**P1 Gate**:
- ✅ `test_correctness.cpp` extended cases pass
- ✅ Generation runs to completion on MIN-16e with grouped path
- ✅ Decode TPS on MIN-16e ≥ **22 t/s** (intermediate gate; the hard 24 t/s
  gate at sprint level is evaluated after P3 too)

### P2 — Real-model quality verification (2 days, parallel to P1)

1. **P2.1** — Locate / fetch DSv4-Flash-AVG-16e GGUF; if absent, fall back
   to DeepSeek-V4-Flash-IQ2-64e. Document choice in REPORT-18.
2. **P2.2** — Fixed prompt set of 10 prompts (mixed: chat, code, math).
   Record token IDs from 32-token greedy decode (seed=0, temp=0) on:
   - `-ot exps=CPU` (CPU MoE reference)
   - `-ot exps=CUDA_TURBOMIND0` per-expert path (SPRINT-023 baseline)
   - `-ot exps=CUDA_TURBOMIND0` grouped path (SPRINT-024 P1)
3. **P2.3** — Compute per-prompt leading-token match rate (max prefix of
   identical tokens / 32). Aggregate.

**P2 Gate**:
- ✅ ≥ 75 % aggregate leading-token match between CPU and grouped TURBOMIND
- ✅ ≥ 95 % match between per-expert TURBOMIND and grouped TURBOMIND (these
  should be near-identical — both call the same kernel, just different
  launch granularity)

If the per-expert / grouped match is < 95 %, that's a correctness regression
and P1 needs to be revisited before P3 starts.

### P3 — FP16 activation boundary across the FFN (3 days)

1. **P3.1** — Identify the FFN routing graph in `ggml_cuda_mul_mat_id`'s
   callers (the ggml graph builder). Find the immediate predecessor /
   successor casts around the `MUL_MAT_ID` op for `ffn_*_exps` tensors.
2. **P3.2** — Add an FP16 fast-path in `ggml_cuda_mul_mat_grouped_turbomind`:
   if `src1->type == GGML_TYPE_F16` already, skip the FP32→FP16 cast.
   Mirror on the output side: if `dst->type == GGML_TYPE_F16`, skip the
   FP16→FP32 cast.
3. **P3.3** — In the ggml graph builder for the DeepSeek-V4 MoE FFN (file
   path TBD during P3 — likely `src/llama-model.cpp` near the FFN routing),
   emit FP16 tensors for the activation that feeds into the experts when
   the `exps` tensor is on `CUDA_TURBOMIND`. Use the existing
   `ggml_cast(ctx, x, GGML_TYPE_F16)` machinery.

   **Scope guard**: this is a graph-builder change limited to the MoE FFN
   branch. Do NOT push FP16 across the attention block or the dense layers
   — that's an explicit non-goal (see §10).

4. **P3.4** — Test: rerun the P2 quality gate. The FP16-boundary change
   should preserve the leading-token match within ± 5 %. If a regression
   appears, the cast was masking an upstream numerical issue and we need
   to investigate.
5. **P3.5** — Rerun `llama-bench`. Expected: additional 5–10 % decode TPS.

**P3 Gate**:
- ✅ P2 quality match holds within ± 5 %
- ✅ Decode TPS on MIN-16e ≥ **24 t/s** (the hard sprint gate)

### P4 — Profile + tune (2 days)

1. **P4.1** — `ncu` metric pack on the grouped path: `sm70_*` HMMA active %,
   DRAM read/write %, `cudaLaunchKernel` per dispatch, occupancy. Compare to
   P0.2 baseline.
2. **P4.2** — If launch count is still ≥ 1 500/s after grouped+FP16, profile
   the cast / permute kernels. Likely candidates for fusion:
   - `permute_and_cast` and the up-projection's input gather
   - `cast_and_unpermute` and the down-projection's output scatter
3. **P4.3** — Cache the device-side `StridedPtr` arrays per-tensor (in
   `ggml_turbomind_tensor_extra`) if the per-call rebuild shows up in
   profile. The pointers are stable for the lifetime of the tensor.
4. **P4.4** — If we hit the stretch gate (30 t/s) before P4 work, P4 can be
   trimmed to documentation only.

**P4 Gate**: no regression; recorded ncu numbers ready for REPORT-18.

### P5 — Measurement + REPORT-18 (1 day)

1. **P5.1** — Decode TPS table: SPRINT-023 baseline vs SPRINT-024 grouped vs
   SPRINT-024 grouped+FP16 on MIN-8e / MIN-16e / MIN-32e, three reps each.
2. **P5.2** — Quality table: leading-token match rates from P2 on real-weight
   model.
3. **P5.3** — ncu deltas from P0.2 → P4.1.
4. **P5.4** — REPORT-18 narrative; sprint-level memory updates if any new
   sm70 gotchas surfaced.

**P5 Gate**: REPORT-18 written; perf gate evaluated.

### P6 — Close-out (0.5 day)

1. Commit, push, tag `sprint-024-close`.
2. Write `SPRINT-024-FOLLOWUPS.md` capturing anything deferred to SPRINT-025
   (likely multi-slot decode + speculative).
3. Update sprint memory entries.

---

## 5. Files Summary

### New files

| Path | Purpose |
|---|---|
| `docs/sprints/SPRINT-024.md` | This sprint (final), after draft review |
| `docs/sprints/SPRINT-024-P0-baseline.md` | P0 baseline + ncu data |
| `docs/sprints/SPRINT-024-FOLLOWUPS.md` | Carry-overs to SPRINT-025 |
| `tools/tc-grid/docs/REPORT-18.md` | Sprint close report |

### Modified files

| Path | Change |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` | Add `ggml_cuda_mul_mat_grouped_turbomind` declaration; extend `ggml_turbomind_tensor_extra` if P4.3 caching lands (add `void* w_arr_dev; void* s_arr_dev;`) |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` | Implement grouped helper, permute+cast fused kernels, StridedPtr build, dlsym `ggml_turbomind_mul_mat_grouped` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Branch in `ggml_cuda_mul_mat_id` to call the grouped helper when `is_turbomind` (~line 2654); skip P3 FP16 fast-path |
| `ggml/vendor/turbomind/test_correctness.cpp` | New test cases: grouped vs per-expert equivalence at N ∈ {2, 6, 8, 16} |
| `src/llama-model.cpp` (P3.3) | FP16 activation tensors for FFN router input/output when `exps` is `CUDA_TURBOMIND` (path/symbol TBD during P3) |
| `tools/llama-bench/llama-bench.cpp` | If a new measurement flag needed (e.g., toggle grouped vs per-expert via env) |

### Unchanged (frozen surfaces)

- `ggml/vendor/turbomind/include/ggml-turbomind-api.h` — C ABI is stable.
  `GGML_TURBOMIND_API_VERSION` stays at 1.
- `ggml/vendor/turbomind/api.cc` — `ggml_turbomind_mul_mat_grouped` already
  does the work. No edits unless P4 surfaces a bug.

---

## 6. Definition of Done

### Ship-blockers (must pass to close the sprint)

1. ✅ Grouped dispatch helper implemented; `ggml_cuda_mul_mat_id` routes
   TURBOMIND tensors to it.
2. ✅ `test_correctness.cpp` grouped-vs-per-expert equivalence passes at
   N ∈ {2, 6, 8, 16} within FP16 ULP relative tolerance.
3. ✅ Decode TPS on MIN-16e ≥ **24 t/s** under `llama-bench -p 128 -n 32 -r 3`
   on V100-SXM2-32GB.
4. ✅ Quality verification on a non-MIN model: ≥ 75 % leading-token match vs
   CPU MoE baseline on the 10-prompt set under greedy decode.
5. ✅ ≥ 95 % leading-token match between per-expert TURBOMIND and grouped
   TURBOMIND on the same prompt set (correctness sanity).
6. ✅ FP16 boundary across the FFN landed and active (P3); cast pair runs
   once per token per layer, not 6× per token per layer.
7. ✅ No regression on `ggml/vendor/turbomind/test_correctness.cpp` legacy
   cases.
8. ✅ REPORT-18 captures the TPS table, quality table, and ncu deltas.
9. ✅ Memory entries updated for any new sm70 / packing gotchas.

### Explicit non-gates

- ❌ **Stretch 30 t/s.** A nice-to-have, not a ship-blocker. Hit means we
  punt SPRINT-025 multi-slot; miss means SPRINT-025 is mandatory.
- ❌ **Perplexity sweep.** Out of scope; needs multi-GPU and a real eval set.
- ❌ **256e full model.** Won't fit on a single V100 (156 GiB). Quality on a
  smaller real-weight variant is the gate.
- ❌ **Multi-slot / speculative decode.** Strategic, SPRINT-025+.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Grouped ABI passes the num_experts=1 unit test but breaks at num_experts > 1 | Medium | High | P1.4 covers N ∈ {2, 6, 8, 16}; if it breaks, the failure mode (column wrap mod 32 / NaN scales / wrong offsets) maps to known sm70 packing memory entries |
| 2 | StridedPtr stride field wrong → corrupted reads (signature: first 32 N-cols match, rest garbage; or per-expert offset wrong) | High | High | Memory `turbomind_packed_b_ld_factor.md` documents the rule; P1.4 catches; also dump first/last expert's data to host and bit-compare at the gate |
| 3 | Launch-overhead is not the bottleneck — grouped path gives <1.4× | Medium | High (sprint thesis) | P0.2 measures launch overhead first; if it's < 60 % we halt and reconsider before doing P1 work. Fallback: P3+P4 still ship; sprint closes with whatever lift we get and SPRINT-025 plans differently |
| 4 | FP16 boundary surfaces numerical instability in attention or norms | Medium | Medium | P3 scope is strictly the FFN router input/output; gate is the P2 quality match within ± 5 %. If the gate fails, revert P3.3 and ship grouped-only |
| 5 | DSv4-Flash-AVG-16e not available; IQ2-64e doesn't fit on 32 GB V100 with TURBOMIND offload | Medium | Medium | Fallback: use a smaller DeepSeek variant (MoE-7B / Qwen-MoE) for the quality gate; document delta and note that production-model quality verification carries over to SPRINT-025 |
| 6 | `cudaMemcpy` for `total_tokens` in `ggml_turbomind_mul_mat_grouped` (api.cc:607) is synchronous and serializes the stream | Low | Medium | Acceptable at decode (one sync per layer = 43/token); if profiling shows it dominates, lift to a stream-local atomic counter and remove the sync |
| 7 | Per-expert StridedPtr build cost is non-trivial at decode | Low | Low | P4.3 caches in `ggml_turbomind_tensor_extra` (pointers are stable for tensor lifetime) |
| 8 | Scope creep into SPRINT-025 multi-slot territory | Medium | Medium | Hard non-goal in §10; reviewers reject if scope drifts |
| 9 | ggml-cuda graph capture path breaks because the grouped helper has host syncs (cudaMemcpyAsync + cudaStreamSynchronize for ids) | Medium | Medium | The existing per-expert path already has these (lines 2685–2686 + 2713–2714); we match the same pattern. If graph capture is desired in a later sprint, that's a SPRINT-025 follow-up |
| 10 | The grouped path's correctness depends on `decode_pack(k_pack_value & 0xFFFu)` matching the convertor used at pack time — if `set_tensor` packed at different `Pack_M` than what the dispatch infers, we get garbage | Low | High | The pack value is stored in `extra->k_pack` at `set_tensor`; the dispatch helper reads it from the same source. Asserts in test_correctness verify this on real data |

---

## 8. Security

- **No new attack surface.** The C ABI surface is unchanged; we add an
  internal caller of an existing entry point.
- **No new dlopen path.** Same `libggml-turbomind.so` from the same directory
  as `libggml-cuda.so`. `dlsym("ggml_turbomind_mul_mat_grouped")` is a new
  symbol lookup but the library was built with `-fvisibility=default` for
  the API symbols.
- **Memory safety.** The dispatcher computes `total_tokens` from
  `expert_offsets[num_experts]` (device read via `cudaMemcpy` in api.cc:607);
  if upstream produces garbage offsets we get a kernel-side OOB read.
  Mitigation: assert `0 <= total_tokens <= n_tokens * n_expert_used` host-side
  before launch.
- **No deserialization.** Same as SPRINT-023.

---

## 9. Dependencies

### Upstream

- `nisparks/experiment/deepseek-v4-dynamic-graph` (unchanged)
- `research/lmdeploy/` patched checkout (unchanged; same `cuda-patches/0006-*.patch`)

### Hardware

- gpu-01 V100-SXM2-32GB
- 251 GiB host RAM
- CUDA 12.2 toolchain (build pod `llamacpp-build`)

### Sprint preconditions

- ✅ SPRINT-023 closed: `libggml-turbomind.so` builds, per-expert dispatch works
- ✅ `ggml_turbomind_mul_mat_grouped` callable from libggml-cuda via dlsym
- ✅ `ggml_turbomind_tensor_extra` populated by `set_tensor` for MoE tensors
- ✅ `test_correctness.cpp` infrastructure exists
- ⏳ DSv4-Flash-AVG-16e GGUF available locally (or accept IQ2-64e fallback);
  verify before P2 starts

---

## 10. Open Questions

1. **Hard perf gate magnitude.** Intent doc proposes ≥ 24 t/s on MIN-16e
   (≥ 1.45× over 16.6 t/s baseline). Conservative model is ~1.5× from launch
   amortization alone. Is 24 the right number, or should we set 28 (≥ 1.7×)
   and accept higher miss rate?
2. **F-02 / P3 scope.** FP16 boundary across the whole FFN is bounded;
   pushing further (across attention) is out of scope here but will SPRINT-025
   want it? Decision to defer here, but flag in followups.
3. **F-03 quality model.** AVG-16e (preferred) vs IQ2-64e — pick one for the
   gate or both? Both increases scope; one is sufficient.
4. **Should we cache StridedPtr arrays per-tensor (P4.3)?** Adds complexity
   to `ggml_turbomind_tensor_extra` and lifetime management. Defer until
   profiling demands it.
5. **`cudaMemcpy` for `total_tokens` in api.cc:607.** Stream sync per layer is
   ~43 syncs/token. Likely fine at decode; lifts cleanly if needed. Decision:
   leave as-is for SPRINT-024; revisit if P4.1 ncu shows it dominating.
6. **Should grouped path fall back to per-expert if num_experts == 1 (dense
   layers)?** The C ABI handles num_experts == 1 (and the P0.3 test pins
   this), but per-expert may be lower overhead. Decision: dispatch grouped
   only when `ne02 > 1`; else use the existing single-launch path.
7. **Behavior when prefill (M >> 1) hits the grouped path.** Each expert sees
   a non-trivial token batch; turbomind ceilings start to apply. Probably a
   net win. Document and verify in P1.5 measurements.

---

## 11. What this sprint is NOT

- New kernel work. `Config_E4M3` and `Config_MXF4` from SPRINT-023 are the
  kernels; we're just collapsing launches.
- A perplexity / quality sweep against the real 256e model. That needs
  multi-GPU and is SPRINT-025+ territory.
- Multi-slot decode or speculative decoding (F-04). Strategic, SPRINT-025+.
- New buffer types or C ABI changes. CUDA_TURBOMIND and the API stay frozen.
- A push for FP16 end-to-end across the whole model graph. We touch the FFN
  router boundary only.
- A profile-and-rewrite cycle on the cast / permute kernels — P4 may fuse
  but won't redesign.

---

## 12. Sequencing — what if P1 underperforms?

| P1.5 outcome | Sprint plan |
|---|---|
| Decode ≥ 24 t/s on MIN-16e | P3 still lands for the 5–10 % follow-on; P4 trims to documentation; close on schedule |
| Decode 20–23 t/s | P3 becomes the critical path; gate is met iff P3 + P4 add the missing ≥ 1 t/s. If after P3 we're still below 24, sprint closes with `partial-gate-miss` status, REPORT-18 documents the actual ceiling and SPRINT-025 mandates multi-slot |
| Decode < 20 t/s | Launch overhead was not the dominant cost. P3 still ships (independent value), but the launch-amortization thesis was wrong. SPRINT-025 = pivot to investigate the actual hot path (probably the cast / DRAM traffic), and the M=1 ceiling is closer than expected |
| Correctness regression | P3 + P4 deferred until P1 passes the unit test; ship as a single-phase sprint if necessary |

---

## 13. Estimated effort

| Phase | Days |
|---|---:|
| P0 — Baseline reproduce + ncu + grouped sanity | 1 |
| P1 — Grouped dispatch helper + correctness | 4 |
| P2 — Real-model quality gate (parallel) | 2 |
| P3 — FP16 boundary across FFN | 3 |
| P4 — Profile + tune | 2 |
| P5 — REPORT-18 | 1 |
| P6 — Close-out | 0.5 |
| **Total** | **13.5 days** |

Per `feedback_effort_estimation_undocumented_hardware.md`: signals here are
**lower** than SPRINT-023 — the C ABI is stable, the StridedPtr layout is
documented in memory, the cast / permute kernels are conventional. Realistic
worst case ~20 days; the 3× multiplier doesn't fully apply because the
opaque-hardware work was done in SPRINT-023. P3's graph-builder change is
the highest-uncertainty item.

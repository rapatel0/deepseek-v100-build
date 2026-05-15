# SPRINT-019 — Deferred items

Items proposed in drafts or critiques, or surfaced from prior sprints,
that are explicitly out of scope for SPRINT-019. Each captures the
work specifically enough that a future planner can pick it up without
re-deriving context.

## Turbomind `gemm_bench` standalone build

- **What**: Build turbomind's `gemm_bench` target standalone (commented
  out in upstream `gemm/CMakeLists.txt:108-137`). Provides an
  independent third-party reference point at the same shape, separate
  from our CUTLASS Gemm70 ceiling.
- **Why deferred**: 1–2 days build engineering to reactivate disabled
  upstream targets and reconstruct transitive deps (gemm2, core,
  cublas, quantization_kernels, gpt_kernels). The same need is mostly
  served by CUTLASS Gemm70 in-sprint.
- **Target sprint**: SPRINT-020, **only if** SPRINT-019 misses the 50
  TF goal and we want a second-source comparison before declaring the
  v11/v12 family ceiling.
- **Prerequisites**: SPRINT-019 close report needs an unresolved gap.
- **Files**: `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`
  + transitive dep tree.

## INT4 BN=256 spill fix via two-pass FRAG_N

- **What**: Split FRAG_N=4 into two outer-product passes of FRAG_N=2
  each. Inner K-loop runs twice, re-loading A from SMEM each pass.
  c_frag count drops from 32 to 16, fits the 255-reg cap.
- **Why deferred**: Pattern is parallel to SPRINT-019 P4 (large-BM
  c_frag SMEM spill). Pickup only after P4 succeeds and we want to
  back-port the technique to v3 INT4. From `SPRINT-016-FOLLOWUPS.md`.
- **Target sprint**: SPRINT-020, conditional on P4 success.
- **Prerequisites**: SPRINT-019 P4 ships a working c_frag rotation pattern.
- **Files**: would create `tools/tc-grid/kernels/v3_bn256_kernels.cuh`;
  modify `tools/tc-grid/src/launch_int4.cu`.

## v5 persistent-CTA grid-stride revisit

- **What**: v5 used persistent CTAs but underperformed v3 at sprint-016.
  With v12's reduced register footprint potentially unlocking 3+ CTAs/SM,
  the persistent pattern may now help. From `SPRINT-016-DEFERRED.md`.
- **Why deferred**: Conditional on SPRINT-019 P1 (FP16 acc) succeeding
  and ncu showing occupancy headroom (waves_per_multiprocessor > 2.0).
- **Target sprint**: SPRINT-020, conditional on P1 success.
- **Prerequisites**: P1 ships AND
  `launch__waves_per_multiprocessor > 2.0` at the champion shape.
- **Files**: `tools/tc-grid/kernels/v5_kernels.cuh` (already exists).

## Cache-policy `Stream` (`ld.global.cs`) revisit with restricted scope

- **What**: SPRINT-017 tried `__ldcs` on both W_qs and W_scales; regressed
  -30% because W_scales has high L1 reuse. A restricted version using
  `__ldcs` ONLY on W_qs (which truly has no reuse) may give the +0–3%
  turbomind expects from Stream policy.
- **Why deferred**: Negative result from sprint-017 burned this lever;
  only revisit with very targeted scope after the major levers settle.
- **Target sprint**: SPRINT-020 tail or stretch.
- **Prerequisites**: SPRINT-019 main levers shipped; new champion
  identified; ncu shows non-trivial L1 contention.
- **Files**: `tools/tc-grid/kernels/v12_kernels.cuh` load_tile().

## tc-grid harness CSV quoting

- **What**: Quote the `dist` field in tc-grid's CSV output (e.g.,
  `"U(-1,1)"`) so embedded commas don't break field-by-comma splits.
  From `SPRINT-016-FOLLOWUPS.md`.
- **Why deferred**: Ergonomics; not blocking SPRINT-019 measurements.
  Parsing scripts in this sprint handle the existing format.
- **Target sprint**: Future ergonomics pass.
- **Files**: `tools/tc-grid/src/main.cu` (CSV output format).

## `scripts/ledger.py` for sprint state

- **What**: Implement the ledger sync referenced by the sprint-plan
  workflow, OR remove the references from the workflow. From
  `SPRINT-016-FOLLOWUPS.md`.
- **Why deferred**: Missing tool noted but not blocking; the sprint
  workflow proceeds without it. Process automation, not perf work.
- **Target sprint**: When project conventions tighten.
- **Files**: Process / tooling.

## MoE-aware dispatcher integration into DSv4 inference

- **What**: Once per-M (and possibly per-shape) champions are
  identified at sprint-019 close, wire them into the live DSv4
  inference path (lmdeploy or pytorch backend).
- **Why deferred**: Systems-integration sprint, not kernel-tuning.
  Different files, different risk profile.
- **Target sprint**: SPRINT-021 (after SPRINT-020's possible
  architectural break).
- **Prerequisites**: SPRINT-019 close with documented per-M champion table.
- **Files**: outside `tools/tc-grid/` — touches DSv4 model code.

## §6.6 multi-shape MoE validation (deferred to in-sprint)

- **Status**: Originally deferred in SPRINT-019-INTENT.md. **Promoted
  to in-sprint as P6** per user interview decision. Listed here for
  audit-trail completeness only.

## Wholesale port of turbomind sm_70 GEMM library

- **What**: Replace our v11/v12 kernel family with a port of turbomind's
  full sm_70 GEMM library (iterator_sm70.h, mainloop_sm70.h, the
  shipped tile registry in sm70_884_4.cu).
- **Why deferred**: Major architectural break. Only worth doing if
  SPRINT-019 confirms the v11/v12 family hits its ceiling well below
  the CUTLASS reference AND we want to capture turbomind's specific
  optimizations (phase-table swizzle, dispatcher choices we haven't
  reproduced).
- **Target sprint**: SPRINT-020, conditional on SPRINT-019 outcome.
- **Prerequisites**: SPRINT-019 REPORT-13 documents the v12 ceiling
  with ncu evidence.
- **Files**: would create `tools/tc-grid/kernels/turbomind_*.cuh` family.

## Direct CUTLASS extension with custom dequant prologue

- **What**: Use CUTLASS 3.x's `MmaTensorOpComputeWithMask` or equivalent
  extension point to inject our INT8 dequant as a prologue.
- **Why deferred**: CUTLASS 2.11.0 (current pin) doesn't expose this
  cleanly; would require CUTLASS 3.x upgrade. The MixedInputGemm pattern
  is also CUTLASS 3.x+.
- **Target sprint**: SPRINT-021+.
- **Prerequisites**: SPRINT-019 outcome warrants the CUTLASS dep upgrade.
- **Files**: CMakeLists.txt FetchContent_Declare cutlass GIT_TAG bump;
  new `kernels/cutlass3_int8_kernels.cuh`.

## 96x128 tile shape exploration

- **What**: Turbomind ships a 96×128 tile (sm70_884_4.cu line 22). We
  didn't include 96 in any grid sweep because it's a non-power-of-2 BM.
- **Why deferred**: Grid-search in SPRINT-019 P1.4/P2.3/etc. will
  include 96 if competitive at 96x128. If it doesn't emerge as a winner,
  it stays deferred as a curiosity.
- **Target sprint**: Future tile-tuning pass.
- **Prerequisites**: None.
- **Files**: `tools/tc-grid/src/main.cu` kTiles[].

## Lower-bound exploration: M ∈ [1, 8) granular dispatch

- **What**: SPRINT-019 P3 tests M ∈ {1, 8, 32} but the per-M
  dispatcher just uses M-thresholds. For M=2 vs M=3 vs M=7, separate
  KSPLIT or kernel choice might matter.
- **Why deferred**: Granular M-dispatch is a deployment concern, not a
  kernel-tuning concern. SPRINT-019 fixes the broad gap at M ∈ {1, 8, 32}.
- **Target sprint**: SPRINT-021 (deployment-integration).

---

## Summary table

| Item | Target Sprint | Prerequisite / Blocker |
|---|---|---|
| Turbomind gemm_bench standalone | SPRINT-020 (conditional) | SPRINT-019 misses 50 TF |
| INT4 BN=256 spill fix | SPRINT-020 (conditional) | SPRINT-019 P4 ships rotation pattern |
| v5 persistent-CTA revisit | SPRINT-020 (conditional) | SPRINT-019 P1 succeeds; waves > 2.0 |
| Cache-policy Stream (W_qs only) | SPRINT-020 tail / stretch | SPRINT-019 main levers settle |
| tc-grid CSV quoting | Future ergonomics pass | None |
| scripts/ledger.py | When conventions tighten | None |
| MoE-aware dispatcher in DSv4 | SPRINT-021 | SPRINT-019 per-M table |
| Turbomind GEMM wholesale port | SPRINT-020 (conditional) | SPRINT-019 confirms v12 ceiling |
| CUTLASS 3.x extension | SPRINT-021+ | CUTLASS 3.x dep upgrade approved |
| 96x128 tile exploration | Future tile-tuning | None |
| Granular M ∈ [1, 8) dispatch | SPRINT-021 (deployment) | SPRINT-019 broad fix shipped |

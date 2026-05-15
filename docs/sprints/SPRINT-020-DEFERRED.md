# SPRINT-020 — Deferred items

Items proposed in drafts or critiques, or carried from prior sprints,
that are explicitly out of scope for SPRINT-020.

## Wholesale turbomind kernel port into `tools/tc-grid/kernels/`

- **What**: Replace v12 family with turbomind-derived kernels in
  `tools/tc-grid/kernels/turbomind_*.cuh`. Full integration of
  `sm70_884_{4,8,16}.cu`, `mainloop_sm70.h`, `iterator_sm70.h` into
  the tc-grid kernel tree.
- **Why deferred**: SPRINT-020 takes a bench-first approach to AVOID
  committing to this large rewrite until evidence (gemm_bench numbers
  via tc-grid bridge) justifies it. The port is what SPRINT-021 becomes
  if SPRINT-020 hits the breakthrough threshold.
- **Target sprint**: SPRINT-021, conditional on SPRINT-020 breakthrough
  branch (≥ 44 TF or asymmetric ≥ 10%).
- **Prerequisites**: SPRINT-020 P3 evidence shows the breakthrough
  threshold is met.
- **Files**: would create `tools/tc-grid/kernels/turbomind_*.cuh` family
  + replace v12 dispatch in `launch_int8.cu`.

## CUTLASS 3.x extension with custom dequant prologue

- **What**: Use CUTLASS 3.x's `MmaTensorOpComputeWithMask` or
  equivalent to inject INT8 dequant as a prologue. Bumps CUTLASS pin
  from 2.11.0 to 3.x.
- **Why deferred**: CUTLASS 3.x focuses on sm_80+. Even if it supports
  sm_70, the dep upgrade is heavy and only worth it if SPRINT-020 +
  SPRINT-021 (potential port) both fail to close the gap.
- **Target sprint**: SPRINT-022+, conditional on neither SPRINT-020 nor
  SPRINT-021 reaching 50 TF.
- **Prerequisites**: Both SPRINT-020 ceiling-proof AND SPRINT-021 port
  (if executed) hit walls.
- **Files**: `tools/tc-grid/CMakeLists.txt` (CUTLASS pin),
  `tools/tc-grid/kernels/cutlass3_int8_kernels.cuh` (new).

## INT4 BN=256 spill fix via two-pass FRAG_N

- **What**: Split FRAG_N=4 into two outer-product passes of FRAG_N=2
  each. From SPRINT-016-FOLLOWUPS / SPRINT-019-DEFERRED.
- **Why deferred**: Was conditional on SPRINT-019 P4 (c_frag SMEM
  spill) success. P4 closed as NEGATIVE result (BM=192/256 v12_ms3
  lose at every M because kernel is mio-bound). Spill-rotation
  optimization would worsen mio_throttle. Stays deferred.
- **Target sprint**: SPRINT-022+ if mio_throttle is mitigated.
- **Prerequisites**: An mio_throttle attack (e.g., Swizzle<3,3,3>
  revisit) reduces SMEM bank contention.
- **Files**: would create `tools/tc-grid/kernels/v3_bn256_kernels.cuh`.

## v5 persistent-CTA grid-stride revisit

- **What**: v5 was abandoned in sprint-016. With v12's reduced register
  footprint, persistent CTAs may now help.
- **Why deferred**: Was conditional on `waves_per_multiprocessor > 2.0`
  per SPRINT-019-DEFERRED. SPRINT-019 measured waves/SM at 5.6 already
  (same as v11) — no occupancy headroom from the register relief; the
  freed registers are spent on rmem buffers (P2 work). Condition not
  met.
- **Target sprint**: SPRINT-022+ if a future sprint frees waves/SM.
- **Files**: `tools/tc-grid/kernels/v5_kernels.cuh`.

## Cache-policy `Stream` (`ld.global.cs`) revisit with restricted scope

- **What**: Apply `__ldcs` ONLY on W_qs (no reuse), leave W_scales on
  L1. From SPRINT-019-DEFERRED.
- **Why deferred**: Mio_throttle is the new wall; this lever attacks
  L1 capacity, not SMEM bandwidth. Wrong tier of attack.
- **Target sprint**: SPRINT-022+, when L1 contention is on the
  critical path again.
- **Files**: `tools/tc-grid/kernels/v12_kernels.cuh` load_tile().

## tc-grid harness CSV `dist` field quoting

- **What**: Quote the `dist` field in CSV output (e.g., `"U(-1,1)"`)
  so embedded commas don't break naive parsers.
- **Why deferred**: Ergonomics, not blocking. SPRINT-019's
  `scripts/bench-median.sh` parser handles the unquoted format.
- **Target sprint**: Future ergonomics pass.
- **Files**: `tools/tc-grid/src/main.cu`.

## `scripts/ledger.py` for sprint state

- **What**: Implement the ledger sync referenced by the sprint-plan /
  sprint-execute workflows, OR remove the references. From
  SPRINT-016-FOLLOWUPS.
- **Why deferred**: Missing tool noted but not blocking; workflows
  proceed without it. Process automation.
- **Target sprint**: When conventions tighten.

## Pytorch backend integration (alternative to lmdeploy)

- **What**: Wire v12_ms3 + v12s into a pytorch-based DSv4 inference
  path as an alternative to lmdeploy's turbomind backend.
- **Why deferred**: SPRINT-020 picked lmdeploy turbomind as the
  primary runtime target. Pytorch path is a different infrastructure
  with different correctness gates.
- **Target sprint**: SPRINT-022+ if pytorch deployment becomes a
  production target.

## 96×128 tile shape exploration

- **What**: Turbomind ships a 96×128 tile (`sm70_884_4.cu` line 22).
  Non-power-of-2 BM wasn't included in SPRINT-019 grid sweeps.
- **Why deferred**: Specific tile-tuning curiosity; not a sprint
  primary lever.
- **Target sprint**: Future tile-tuning pass; would surface naturally
  if SPRINT-021 port exposes the full turbomind tile set.

## Granular M ∈ [1, 8) dispatch

- **What**: M=2 vs M=3 vs M=7 dispatch differences. SPRINT-020 P0.3
  uses M-thresholds; granular dispatch is a deployment-side concern.
- **Why deferred**: Deployment infrastructure concern, not kernel
  tuning. P5 ceiling-proof branch validates at end-to-end model level.
- **Target sprint**: SPRINT-022+ (deployment optimization).

## VISION.md document

- **What**: Long-horizon sprint sequencing document via `/vision`.
- **Why deferred**: SPRINT-016–019 have proceeded without one. The
  architectural-decision branch at SPRINT-020 P4 is the right moment
  to consider sequencing for SPRINT-021+, but `/vision` is itself a
  workflow that needs the user's invocation.
- **Target sprint**: After SPRINT-020 close; before SPRINT-021 if
  taking breakthrough branch.

## nsys PNG screenshots

- **What**: Render Nsight Systems timeline PNGs for P2/P3 ncu work.
- **Why deferred**: gpu-01 pod is headless (per SPRINT-019-FOLLOWUPS).
  Kernel-summary CSV used as text proxy.
- **Target sprint**: When a GUI workstation or `nsys export --type png`
  workflow is wired in.

---

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Wholesale turbomind port | SPRINT-021 (conditional) | SPRINT-020 breakthrough branch |
| CUTLASS 3.x extension | SPRINT-022+ | Neither SPRINT-020 nor SPRINT-021 closes 50 TF gap |
| INT4 BN=256 spill fix | SPRINT-022+ | mio_throttle mitigation lands first |
| v5 persistent-CTA revisit | SPRINT-022+ | waves/SM headroom appears |
| Cache-policy Stream (W_qs only) | SPRINT-022+ | L1 contention is on critical path again |
| tc-grid CSV `dist` quoting | Future ergonomics | None |
| scripts/ledger.py | When conventions tighten | None |
| Pytorch backend integration | SPRINT-022+ | Pytorch becomes production target |
| 96×128 tile shape exploration | Future tile-tuning | None |
| Granular M ∈ [1, 8) dispatch | SPRINT-022+ | Deployment optimization phase |
| VISION.md (`/vision` workflow) | Pre-SPRINT-021 | User invocation |
| nsys PNG export | Whenever GUI is available | Headless pod limitation |

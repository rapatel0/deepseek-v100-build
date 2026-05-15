# SPRINT-020 — Turbomind decision sprint for V100 INT8 GEMM

## Overview

SPRINT-019 proved that the `v12` family can move the V100 INT8 path from
`35.07 TF` to `38.98 TF` at `M=2048, N=K=7168`, and that `v12s_ks8`
closes the decode-style gap at `M=64` with `21.55 TF`. It also changed the
limiting factor: `v12_ms3` is no longer gmem-latency-bound, it is now
SMEM-bandwidth-bound. That makes SPRINT-020 an architectural sprint, not
another local tile-tuning sprint.

This draft recommends the **turbomind port path** as the primary plan.
Reason: the repo already contains a production-grade SM70 GEMM stack in
`research/lmdeploy/src/turbomind/kernels/gemm/` with registry, dispatch
cache, SplitK support, and MoE call sites. That is the shortest route to
either a real breakthrough or a defensible ceiling proof. A CUTLASS
extension is more speculative and would require fresh template surgery.
Deployment-only work is valuable, but it accepts the `38.98 TF` ceiling
before independently testing a different mainloop.

Headline sprint targets:

- Reproduce the current champions exactly: `38.98 TF` at `M=2048` for
  `v12_ms3` and `21.55 TF` at `M=64` for `v12s_ks8`.
- Build and run a Turbomind SM70 benchmark on the same V100 and the same
  DSv4-relevant shapes.
- Achieve one of two outcomes:
  - **Breakthrough:** `>= 44.0 TF` median-of-5 at
    `M=2048, N=K=7168`, or `>= 10%` win over `v12_ms3` on the asymmetric
    DSv4 shapes.
  - **Ceiling proof:** Turbomind, measured apples-to-apples, stays
    `<= 41.0 TF` at `M=2048, N=K=7168`, with the limiting counters and
    kernel structure documented.

## Use Cases

- **Large-M production GEMM**: determine whether DSv4 prefill-style work
  should stay on `tools/tc-grid/kernels/v12_kernels.cuh` or move to a
  Turbomind-derived kernel family.
- **Decode / small-M serving**: preserve `v12s` as the default unless the
  new path demonstrates equal-or-better `M=64` behavior.
- **Asymmetric MoE shapes**: measure the shapes SPRINT-019 could not cover
  cleanly because `tc-grid` still assumes `N == K` in its CLI surface.
- **Runtime selection**: if Turbomind wins, use its existing
  `registry.cu` + `dispatch_cache.{h,cu}` path instead of inventing a new
  dispatcher from scratch.

## Architecture

Current state:

- `tools/tc-grid/kernels/v12_kernels.cuh` contains the shipped kernel
  family: `mm_int8_lut_v12`, `mm_int8_lut_v12_ms3`, and `mm_int8_lut_v12s`.
- `tools/tc-grid/src/launch_int8.cu` dispatches versions `50`, `51`, and
  `60` for those kernels.
- `tools/tc-grid/src/main.cu` is still the benchmark control plane, but its
  CLI is oriented around `--nk` and therefore under-serves the asymmetric
  MoE cases.

Target state:

- `tc-grid` remains the apples-to-apples measurement harness.
- Turbomind SM70 GEMM is brought up in two layers:
  - **Layer 1: native bench bring-up** in
    `research/lmdeploy/src/turbomind/kernels/gemm/test/gemm_bench.cu`,
    backed by `kernel/sm70_884_{4,8,16}.cu`,
    `arch/config_sm70_s884.h`, `mainloop_sm70.h`,
    `iterator_sm70.h`, and `registry.cu`.
  - **Layer 2: tc-grid bridge** that lets `tc-grid` invoke Turbomind
    GEMM on the exact same shapes, distributions, and tolerance checks
    used by `v12_ms3` and `v12s`.

The design choice is deliberate: benchmark first, then bridge, then decide
whether to port code into `tools/tc-grid/kernels/` or to accept Turbomind's
`gemm2` stack as the long-term runtime owner.

## Implementation

### P0 — Lock the comparison contract

1. Reproduce the SPRINT-019 baseline on the live V100 with
   `scripts/bench-median.sh` and record:
   - `v12_ms3<128,128,16,4,16,1> = 38.98 TF` at `M=2048`
   - `v12s<64,128,32,4,8,1,8> = 21.55 TF` at `M=64`
2. Clear the deferred sanitizer debt on the shipped atomic path:
   `compute-sanitizer --tool memcheck,racecheck,initcheck` for
   `v12s` with `KSPLIT={2,3,5,8,16}`.
3. Extend `tools/tc-grid/src/main.cu` to support
   `--n-list` and `--k-list` while preserving `--nk` as the square-shape
   shorthand.
4. Record the SPRINT-020 shape catalog in a new doc artifact:
   `7168x7168`, `7168x18944`, `18944x7168`, `2048x7168`,
   `4096x4096`, `8192x8192`.

Files expected:

- `tools/tc-grid/src/main.cu`
- `tools/tc-grid/include/tc_grid.h`
- `tools/tc-grid/docs/REPORT-13.md` references only, no rewrite
- `tools/tc-grid/docs/shape-catalog-SPRINT-020.csv` or equivalent

### P1 — Bring up Turbomind SM70 benchmarking

1. Re-enable a guarded benchmark target in
   `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`.
   Prefer an option such as `TM_ENABLE_GEMM_BENCH` rather than always-on
   NVBench.
2. Un-comment or replace the disabled `gemm_bench` target so it builds on
   the existing CUDA 12.2 / V100 environment without pulling unrelated
   Turbomind test baggage.
3. Extend `research/lmdeploy/src/turbomind/kernels/gemm/test/models.h`
   with DSv4-flash-style entries that map to the catalog above, especially
   the `7168/18944` expert MLP shapes.
4. Verify that the SM70 registrations in
   `research/lmdeploy/src/turbomind/kernels/gemm/registry.cu` expose the
   `sm70_884_4`, `sm70_884_8`, and `sm70_884_16` families on the V100.

Gate:

- `gemm_bench` runs on V100 and emits valid timings for at least
  `M in {64, 256, 1024, 2048}` on one square shape and one asymmetric shape.

Files expected:

- `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`
- `research/lmdeploy/src/turbomind/kernels/gemm/test/gemm_bench.cu`
- `research/lmdeploy/src/turbomind/kernels/gemm/test/models.h`

### P2 — Build the tc-grid to Turbomind bridge

1. Add a new launcher, preferably
   `tools/tc-grid/src/launch_turbomind_int8.cu`, that adapts
   `tc_grid::LaunchSpec` to `turbomind::gemm::Operation`,
   `MatrixLayout`, and `Gemm::Run`.
2. Wire that launcher into `tools/tc-grid/CMakeLists.txt` behind a
   dedicated option such as `TCGRID_ENABLE_TURBOMIND_GEMM`.
3. Register one or more Turbomind-backed rows in `tools/tc-grid/src/main.cu`
   with a new version band, for example `70+`, so the results land in the
   same CSV as `v12_ms3`.
4. Reuse `tc-grid`'s existing data generation and tolerance logic so the
   comparison is on identical activations, identical quantized weights,
   and identical `rel/p99/maxabs` reporting.

Gate:

- Turbomind-backed `tc-grid` rows run cleanly for the square baseline at
  `M=2048, N=K=7168`.
- The bridge produces either:
  - `>= 41.0 TF`, which is enough to continue the port path, or
  - a clear incompatibility report that justifies an immediate pivot.

Files expected:

- `tools/tc-grid/CMakeLists.txt`
- `tools/tc-grid/src/launch_turbomind_int8.cu`
- `tools/tc-grid/src/main.cu`
- `tools/tc-grid/include/tc_grid.h`

### P3 — Measure the real decision surfaces

1. Run median-of-5 square-shape comparisons at
   `M={64,256,1024,2048,4096}` for:
   - `v12_ms3`
   - `v12s`
   - Turbomind bridge rows
2. Run asymmetric-shape comparisons at:
   - `M=2048, N=18944, K=7168`
   - `M=2048, N=7168, K=18944`
   - `M=64, N=18944, K=7168`
3. Capture Nsight Compute counters for the winning Turbomind row and the
   current `v12_ms3` champion using the same metric pack as SPRINT-019.
4. Export Turbomind's chosen launch specs through
   `Gemm::Export` / `DispatchCache::Export` so the measured winners become
   reusable runtime artifacts instead of one-off observations.

Decision rule:

- Continue with Turbomind as the sprint winner if it hits `>= 44.0 TF`
  at `M=2048, N=K=7168`, or if it is `>= 10%` faster than `v12_ms3` on
  the asymmetric `18944/7168` MoE shapes while staying within tolerance.
- Treat the result as a ceiling proof if it stays `<= 41.0 TF` on the
  square baseline and does not open a meaningful asymmetric-shape win.

Files expected:

- `tools/tc-grid/docs/ncu/SPRINT-020-*.csv`
- `tools/tc-grid/docs/SPRINT-020-turbomind-vs-v12.csv`
- `research/lmdeploy/src/turbomind/kernels/gemm/gemm.cu`
- `research/lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.{h,cu}`

### P4 — Runtime follow-through if Turbomind wins

1. Keep `tc-grid` as the lab harness, but move the runtime path to the
   existing Turbomind GEMM owner rather than duplicating dispatch logic.
2. Validate the real call sites:
   - `research/lmdeploy/src/turbomind/models/llama/LlamaLinear.cu`
   - `research/lmdeploy/src/turbomind/models/llama/moe_ffn_layer.cc`
3. Import the measured dispatch cache into the runtime path and confirm
   that the correct kernels are reused rather than re-measured on every run.
4. Write a small integration note that maps `tc-grid` winner labels to the
   Turbomind runtime descriptors actually selected by `registry.cu`.

Gate:

- End-to-end Turbomind runtime uses the measured dispatch cache and does
  not regress correctness on the DSv4-flash MoE path.

### P5 — Fallback close if Turbomind loses

If P2 or P3 fails the thresholds, do not force a partial port. Close the
sprint by documenting the ceiling proof and immediately queue the next
implementation sprint around deployment integration:

1. Land the `tc-grid` `N != K` CLI work.
2. Land the sanitizer debt cleanup.
3. Encode per-shape dispatch rules for the `v12_ms3` / `v12s` split.
4. Use the measured Turbomind result as the evidence for not choosing the
   port or CUTLASS branches.

## Files Summary

- `tools/tc-grid/src/main.cu`
  Add `--n-list` / `--k-list`, register Turbomind-backed rows, preserve
  current `v12` row naming.
- `tools/tc-grid/include/tc_grid.h`
  Extend the control surface for asymmetric shape sweeps and add the new
  launcher prototype.
- `tools/tc-grid/src/launch_int8.cu`
  Keep current `v12` dispatch intact; only add bridge routing if needed.
- `tools/tc-grid/src/launch_turbomind_int8.cu`
  New adapter from `tc-grid` specs to Turbomind `gemm2`.
- `tools/tc-grid/CMakeLists.txt`
  Optional link path for Turbomind GEMM.
- `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`
  Re-enable a guarded benchmark target.
- `research/lmdeploy/src/turbomind/kernels/gemm/test/gemm_bench.cu`
  Native Turbomind benchmark bring-up.
- `research/lmdeploy/src/turbomind/kernels/gemm/test/models.h`
  Add DSv4-relevant shape entries.
- `research/lmdeploy/src/turbomind/kernels/gemm/registry.cu`
  Confirm SM70 kernel registration and runtime visibility.
- `research/lmdeploy/src/turbomind/models/llama/LlamaLinear.cu`
  Runtime validation point if the Turbomind path wins.
- `research/lmdeploy/src/turbomind/models/llama/moe_ffn_layer.cc`
  MoE integration validation point if the Turbomind path wins.

## Definition of Done

- `tc-grid` supports asymmetric `N` and `K` sweeps without breaking square
  `--nk` usage.
- `v12s` clears `memcheck`, `racecheck`, and `initcheck` for the atomic
  `KSPLIT` coverage set.
- A Turbomind SM70 benchmark target builds and runs on the V100.
- The `tc-grid` to Turbomind bridge runs the square baseline and at least
  two DSv4 asymmetric shapes.
- One of these conclusions is backed by CSV + ncu evidence:
  - Turbomind is the new preferred path because it reaches `>= 44.0 TF`
    at `M=2048, N=K=7168`, or opens a `>= 10%` win on the asymmetric MoE
    shapes.
  - Turbomind does not materially beat `v12_ms3`, and the sprint closes
    with a documented ceiling proof plus a deployment-integration handoff.
- If Turbomind wins, the dispatch cache export/import path is exercised in
  `LlamaLinear.cu` or an equivalent runtime entry point.

## Risks

1. **Build-system churn in LMDeploy/Turbomind.**
   Mitigation: gate `gemm_bench` behind a new option and avoid broad
   `BUILD_TEST` changes.
2. **Apples-to-oranges comparisons between tc-grid and Turbomind.**
   Mitigation: make the bridge reuse `tc-grid` data generation, reference
   GEMM, and tolerance evaluation.
3. **Turbomind wins only on shapes that are not production-critical.**
   Mitigation: promote asymmetric `18944/7168` shapes to first-class gates,
   not appendix measurements.
4. **The bridge consumes too much sprint time.**
   Mitigation: set the hard pivot at the end of P2. If the bridge is not
   working by then, close with the benchmark evidence and switch to
   deployment integration.
5. **Sanitizer debt on `v12s` hides an existing correctness issue.**
   Mitigation: clear sanitizer work before using `v12s` as the production
   fallback in any decision memo.

## Security

- No external service or model-facing API change is required for the
  preferred path; the work is local CUDA/C++ benchmarking and runtime
  selection.
- The main security concern is memory safety in new launcher glue. Require
  `compute-sanitizer` and fail closed on shape/layout mismatch.
- Dispatch cache import/export should be treated as trusted local build
  artifacts only; do not add a remote-loading path in this sprint.

## Dependencies

- V100 `sm_70` access on `gpu-01`
- CUDA 12.2 toolchain
- Paused DCGM exporter before `ncu`
- Existing `CUTLASS v2.11.0` pin for comparison only
- NVBench availability if `gemm_bench` is re-enabled
- Turbomind libraries under `research/lmdeploy/src/turbomind/`
- Existing `scripts/bench-median.sh` protocol and SPRINT-019 metric pack

## Open Questions

**Q1. Which architectural path should SPRINT-020 choose?**

Preferred path: **turbomind port, benchmark-first, then bridge into
`tc-grid`.**

Why this is the best path:

- The repo already contains the candidate kernel family:
  `research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_{4,8,16}.cu`
  with its real scheduler and dispatch code.
- It tests a genuinely different architecture from `v12_ms3` without first
  committing to a large rewrite in `tools/tc-grid/kernels/`.
- If it wins, the runtime integration path already exists through
  `LlamaLinear.cu` and `moe_ffn_layer.cc`.
- If it loses, the team gets a strong ceiling proof and can stop chasing
  local GEMM rewrites.

Alternative A: **CUTLASS extension.**

- Scope would center on `tools/tc-grid/kernels/cutlass_int8_kernels.cuh`,
  `tools/tc-grid/src/launch_int8.cu`, and likely a new local wrapper rather
  than editing vendored `_deps/cutlass-src/` directly.
- This path should only be chosen if the expected target is `>= 48 TF` at
  `M=2048`, because anything smaller does not justify the template and
  maintenance cost.
- It is the highest-risk option because SPRINT-018's CUTLASS result used a
  pre-dequant ceiling path, not the fused production problem we actually
  need to ship.

Alternative B: **deployment integration.**

- Scope would center on `tools/tc-grid/src/main.cu`,
  `tools/tc-grid/src/launch_int8.cu`, and the runtime owner chosen for the
  DSv4 serving path.
- This is the right fallback if Turbomind fails to show a material win.
- It should not be the primary SPRINT-020 choice because it would accept
  the `v12_ms3` ceiling without independently testing another architecture.

**Q2. Should the sprint port Turbomind kernels into `tools/tc-grid/kernels/`
or simply bridge `tc-grid` to `gemm2`?**

Bridge first. A direct port is only justified if the bridge wins and there
is a clear maintainability reason not to reuse `gemm2`.

**Q3. What is the minimum “real win” threshold?**

Recommended threshold: `>= 44.0 TF` on the square baseline, or `>= 10%`
over `v12_ms3` on the DSv4 asymmetric shapes. Anything smaller is not a
big enough delta to justify replacing the current champion.

**Q4. What counts as a ceiling proof?**

Recommended answer: Turbomind on the same V100, measured from the same
`tc-grid` harness, remains `<= 41.0 TF` at `M=2048, N=K=7168` and shows no
meaningful asymmetric-shape win.

**Q5. If Turbomind wins, what should own runtime dispatch?**

Recommended answer: Turbomind's existing `registry.cu` and
`dispatch_cache.{h,cu}` should own it. Do not create a second
hand-maintained per-shape dispatcher if a measured cache can be imported.

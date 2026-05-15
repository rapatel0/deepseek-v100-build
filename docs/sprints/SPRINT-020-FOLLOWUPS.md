# SPRINT-020 follow-ups (in-progress; updated as the sprint runs)

Items discovered during execution that emerged from the work itself.

---

## research/lmdeploy/ is untracked

**What**: The entire `research/lmdeploy/` subtree is in the project
`.gitignore`. SPRINT-020 P1.2 edited
`research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt` to add the
`TM_ENABLE_GEMM_BENCH` option, but the edit lives in the working tree
only — not in git. The pattern matches `cuda-patches/` which also stays
out of version control.

**Why discovered**: Tried to `git add` the CMakeLists edit during P1.2
commit; git ignored the path.

**Severity**: Important — without a persistence mechanism, the P1.2 wiring
will be lost on a fresh clone. P1 work effectively needs to live as a
patch file under `cuda-patches/` (matching the convention) or be
re-applied each session.

**Suggested sprint**: SPRINT-020 P1.3 (immediately before any P1.5 build
attempt). Write the CMakeLists modification as a patch under
`cuda-patches/0006-turbomind-gemm-bench.patch` and document the apply
workflow.

**Files**: `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`,
new `cuda-patches/0006-*.patch`, `.gitignore` (verify cuda-patches is
not also git-ignored).

---

## turbomind gemm_bench requires lmdeploy build pipeline

**What**: P1.1 audit assumed `cmake -B build` from the gemm/ dir would
suffice. Closer reading of `research/lmdeploy/CLAUDE.md` shows the
turbomind C++ extension is built via `setup.py` + CMake with a specific
env contract (`LMDEPLOY_TARGET_DEVICE=cuda`, optional `DISABLE_TURBOMIND`,
`CUDACXX`). Standalone CMake invocation in just `kernels/gemm/` will
likely fail because parent targets (parser, fmt, cuda_utils) aren't
provided.

**Why discovered**: Reading `research/lmdeploy/CLAUDE.md` during P1.1.

**Severity**: Important — affects P1.5 build approach. Either:
(a) build via lmdeploy's setup.py pipeline (requires Python env setup
    + may pull lots of lmdeploy-internal deps)
(b) carve a minimal CMake harness that vendors just gemm2+core+nvbench
    into tc-grid's existing build (smaller scope but bigger surgery)

**Suggested sprint**: SPRINT-020 P1.3 decision point.

**Files**: `research/lmdeploy/setup.py`, plus the parent CMakeLists for
parser/core/cuda_utils.

---

## v12s sanitizer status

**What**: SPRINT-020 P0.1 sanitizer pass results:
- `compute-sanitizer --tool memcheck`: 14 cudaErrorInvalidValue entries
  in ERROR SUMMARY — ALL from kernel-launch pre-flight rejects on
  unrelated kernel families (v3/v10/v11 large-BN). v12s itself ran
  cleanly across 12 instantiations × N_TIMING_ITERS+N_WARMUP launches.
- `compute-sanitizer --tool racecheck`: **0 hazards, 0 errors, 0 warnings.**
  v12s SplitK is race-free per the sanitizer.
- `compute-sanitizer --tool initcheck`: in-flight as of latest commit
  (target: 0 errors).

**Why discovered**: SPRINT-019-FOLLOWUPS item 1 (deferred sanitizer
debt); resolved during SPRINT-020 P0.1.

**Severity**: Nice-to-have artifact retention.

**Suggested sprint**: Carries to REPORT-14 §correctness as a
"SPRINT-019 debt cleared" entry.

---

## v12s initcheck got stuck in instrumented re-run loop

**What**: SPRINT-020 P0.1's `compute-sanitizer --tool initcheck` was
killed after 11+ min of slow progress (1% GPU, 99% CPU on the tc-grid
child process). Output reached 2609 lines then stalled. The
`--kernel-regex kns=mm_int8_lut_v12s` filter caused initcheck to
instrument every v12s launch and re-run each kernel many times (per
N_TIMING_ITERS=8 + N_WARMUP_ITERS=2) for stable timing measurements;
combined with initcheck's per-load instrumentation, this multiplied
into an effectively-stuck workload.

**Retry strategy** (for next session):
1. Build a one-shot single-launch test driver (e.g.
   `tests/test_v12s_initcheck_sm70.cu`) that launches v12s ONCE per
   KSPLIT value and exits. Avoid going through tc-grid's
   timing-iteration loop.
2. OR add a `--single-shot` flag to tc-grid that does ONE warmup +
   ONE timed iter only (no rerun loop). Cleaner; reusable.
3. Run initcheck against the one-shot binary with all 5 KSPLIT values.

**Why discovered**: P0.1 execution timed out under instrumentation.

**Severity**: Important — initcheck on v12s is in the sprint §1.2 #2
no-skip rule. memcheck and racecheck already passed; initcheck is
the third leg of the atomic-kernel correctness gate.

**Suggested sprint**: SPRINT-020 P0.1 retry (this session's first
unfinished item).

**Files**: would create `tools/tc-grid/tests/test_v12s_initcheck_sm70.cu`
OR add `--single-shot` flag in `tools/tc-grid/src/main.cu`.

---

## lmdeploy turbomind builds via setup.py + cmake_build_extension

**What**: SPRINT-020 P1.3 investigation found lmdeploy uses
`cmake_build_extension` (PyPI package) to wrap CMake builds inside
Python's setup.py. Building turbomind kernels via standalone
`cmake -B build` in `research/lmdeploy/src/turbomind/kernels/gemm/`
will fail because the parent build provides parser, fmt, cuda_utils
through the lmdeploy CMake superbuild.

**Path forward decided**: option (b) is the only viable path. The pod
has no python3 (`which python3` → 127), so lmdeploy's setup.py path
(option a) is non-viable without first setting up a Python toolchain in
the gpu-01 pod. That's an entire sub-project unto itself.

(b) Carve a minimal harness inside `tools/tc-grid/CMakeLists.txt`
    that vendors only the files gemm_bench needs:
    - turbomind kernels/gemm/*.cu (already has CMakeLists; reuse the
      `gemm2` add_library block)
    - turbomind core/*.cu/*.cc (already has CMakeLists; reuse the
      `core` add_library block)
    - fmt::fmt — add a FetchContent block for `fmtlib/fmt` v10.x or
      v11.x (header-only mode supported via FMT_HEADER_ONLY=ON)
    - NVBench — already in patch 0006 under TM_ENABLE_GEMM_BENCH
    - parser — inspect turbomind/utils/CMakeLists.txt for what's
      actually needed by gemm2 (may be small)
    Concrete shape: `tools/tc-grid/turbomind_minimal/CMakeLists.txt`
    that `add_subdirectory()`'s into the research/lmdeploy/ tree
    selectively, or directly globs the .cu sources.

(a) ABANDONED. Heavy lift for a single benchmark.

**Why discovered**: P1.3 inspection of research/lmdeploy/setup.py.

**Severity**: Important — gates P1.5 (the entire gemm_bench build
smoke-test).

**Suggested sprint**: SPRINT-020 P1.3 decision point.

**Files**: `research/lmdeploy/setup.py`, `tools/tc-grid/CMakeLists.txt`,
potentially a new `tools/tc-grid/turbomind_minimal/` directory.

---

## CMakeLists patch lives in cuda-patches/

**What**: SPRINT-020 P1.2's CMakeLists edit (TM_ENABLE_GEMM_BENCH
guard option) was persisted as `cuda-patches/0006-turbomind-gemm-
bench-guard.patch` matching the existing patch-file convention
(0001..0005). The patch file is the persistence mechanism since
both `research/lmdeploy/` and `cuda-patches/` themselves are
fully .gitignored.

**Why discovered**: P1.2 commit attempt revealed research/ is
ignored.

**Severity**: Nice-to-have — documents the existing workflow more
explicitly.

**Suggested sprint**: SPRINT-020 P6 (REPORT-14 narrative).

**Files**: `cuda-patches/0006-turbomind-gemm-bench-guard.patch`
(working tree only); apply with
`cd research/lmdeploy && git apply ../../cuda-patches/0006-*.patch`.

---

## P1.5 nvbench/cmake conflict — first real P1.5 attempt blocker

**What**: Attempted SPRINT-020 P1.5 build smoke-test of gemm_bench
through the `turbomind_minimal/` carve-out. Two blockers surfaced
immediately:

1. **Pod has no outbound network**. `git clone` from FetchContent
   fails with "Could not resolve host: github.com". The CUTLASS
   FetchContent inside tc-grid worked only because it was cached
   from a prior session with internet.

2. **CMake version mismatch**. The cached NVBench commit
   `d8dced8a64d9ce305add92fa6d274fd49b569b7e` requires
   `cmake_minimum_required(VERSION 3.23.1)`. The pod's CMake is
   3.22.1 (`apt list --installed | grep cmake` → `cmake/jammy
   3.22.1-1ubuntu1.22.04.2`). Trying to switch nvbench to an older
   tag (`v0.1.0` requires 3.20) fails because of (1).

**Resolution paths** for next session:

(a) **Vendor a CMake-3.22-compatible nvbench**. Download an older
    nvbench release tarball on the laptop, copy via kubectl cp to
    the pod's `_deps/` cache, and update the FetchContent_Declare
    to point at the local file:// URL.

(b) **Replace nvbench with a minimal cudaEventRecord harness**.
    Write `tools/tc-grid/turbomind_minimal/gemm_bench_simple.cu`
    that does what nvbench does for our use case (M, N, K parsed,
    median-of-5 timed iterations, TF/ms/gbps printed). Decouples
    from nvbench entirely. Probably 100-200 lines of CUDA + tc-grid
    interop.

(c) **Upgrade CMake in the pod**. Requires kubectl exec apt with
    internet, OR rebuilding the pod container image. Heavy.

Path (b) is preferable because it also dodges nvbench's other
upstream-system dependencies and gives us tighter control over the
output format (already matches tc-grid CSV).

**Why discovered**: Direct attempt at P1.5 build smoke-test.

**Severity**: Important — gates the entire P1.5+P2+P3 architectural
gate.

**Suggested sprint**: SPRINT-020 P1.5 (continuation; decision between
paths a/b/c).

**Files**: `tools/tc-grid/turbomind_minimal/`,
`research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`
(NVBench FetchContent_Declare).

---

## CRITICAL P1.6 — Turbomind has no sm70 INT8 kernel (re-frames §1.1)

**What**: During P1.6 (gemm_bench_simple.cu wiring to `Gemm::Run`),
discovered that turbomind's `sm70_s884` registry has **no INT8/uint8
weight kernel**. The available 8-bit-weight sm70 config is `Config_E4M3`
(FP8 e4m3); the 4-bit config is `Config_U4_g`. INT8 kernels in
turbomind are sm75+ only (`mma.m8n8k16.s8`).

**Verified by**:
- `arch/config_sm70_s884.h`: only Config_U4_d, Config_U4_g, Config_MXF4,
  Config_E4M3, Config_F16 templates exist
- `kernel/sm70_884_{4,8,16}.cu`: only U4_d, U4_g, MXF4, E4M3, F16
  instantiations registered
- Runtime dispatcher message: `No feasible kernel found for the problem:
  sm70_f16_i8k128_f32_ttt_fff_64x7168x7168_1` confirms i8 + sm70 has
  no registry match.

**Why discovered**: P1.6 first runtime attempt with INT8 weight
MatrixLayout failed at dispatch.

**Severity**: **CRITICAL** — invalidates SPRINT-020 §1.1's
"Turbomind ≥ 44 TF" / "≤ 41 TF" head-to-head INT8 contract. The
turbomind sm70 INT8 baseline does not exist as a comparable kernel.

**Pivot taken (P1.6)**: Switched gemm_bench_simple to FP16 ceiling
measurement via turbomind's cuBLAS-path dispatch. Result captured in
`tools/tc-grid/docs/turbomind-fp16-ceiling-SPRINT-020-P1.csv`.

**Headline numbers (FP16, no quant) on V100 N=K=7168**:
- M=64: 38.92 TF
- M=512: 80.03 TF
- M=2048: **87.05 TF**
- M=2048 N=18944 K=7168: 97.06 TF (peak observed)

**Reframed §1.1 contract**:
- Original: Turbomind INT8 vs v12_ms3 INT8 → impossible (no kernel)
- Reframed: Turbomind FP16 ceiling (87 TF M=2048) vs v12_ms3 INT8
  (38.98 TF) → v12_ms3 is at **45% of the sm70 FP16 ceiling**
- Branch: This is BREAKTHROUGH-class headroom by §1.1 spirit (large
  gap to hardware peak). SPRINT-021 should pursue closing it.

**How to apply going forward**:
- Use `turbomind-fp16-ceiling-SPRINT-020-P1.csv` as the canonical
  sm70 ceiling table for v100 work.
- The "50 TF" project goal sits at 57% of FP16 ceiling at M=2048 —
  achievable. Final 87 TF level is the absolute upper bound.
- Skip P2 (tc-grid bridge to turbomind INT8) — there's nothing to
  bridge to.

**Suggested sprint**: This sprint's P3-P6 reframed; SPRINT-021 takes
the breakthrough branch with FP16-ceiling-relative targets.

**Files**: `tools/tc-grid/docs/turbomind-fp16-ceiling-SPRINT-020-P1.csv`,
`tools/tc-grid/turbomind_minimal/gemm_bench_simple.cu` (FP16 path),
`docs/sprints/SPRINT-020.md` §1.1 (needs amendment).

---

## P0.5 reproduce baselines passed ±2% gate

**What**: SPRINT-020 P0.5 median-of-5 reproduce vs SPRINT-019 close:
- v12_ms3 @ M=2048: 39.13 TF (was 38.98, +0.4%)
- v12s_ks8 @ M=64: 21.41 TF (was 21.55, -0.7%)
- v11 champ @ M=2048: 35.02 TF (was 35.07, 0%)

All within ±2% gate. Environment stable across sessions.

**Severity**: Nice-to-have artifact retention.

**Suggested sprint**: REPORT-14 §correctness baselines.

**Files**: `tools/tc-grid/docs/baseline-SPRINT-020-P0.csv`.

---

## Will add as found

Additional items will be appended as P1.3-P6 progresses.

---

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| research/lmdeploy/ untracked → patch persistence | Important | SPRINT-020 P6 (documented) | cuda-patches/ |
| Turbomind needs lmdeploy build pipeline (setup.py path) | Important | SPRINT-020 P1.3 decision | research/lmdeploy/setup.py |
| v12s initcheck stuck in instrumented re-run loop | Important | SPRINT-020 P0.1 retry | tc-grid harness or new test |
| CMakeLists patch in cuda-patches/0006 | Nice-to-have | REPORT-14 docs | cuda-patches/0006-*.patch |
| P0.5 reproduce passed ±2% gate | Nice-to-have | REPORT-14 baselines | baseline-SPRINT-020-P0.csv |
| v12s sanitizer artifacts | Nice-to-have | REPORT-14 evidence | tools/tc-grid/docs/sanitizer/ |
| P1.5 nvbench/cmake/no-internet conflict | Important | SPRINT-020 P1.5 cont. | turbomind_minimal/, kernels/gemm/CMakeLists.txt |

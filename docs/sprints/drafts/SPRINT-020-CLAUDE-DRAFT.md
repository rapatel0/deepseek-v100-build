# SPRINT-020-CLAUDE-DRAFT — Hybrid: turbomind gemm_bench ceiling check + v12 family deployment integration

**Status:** DRAFT (Claude). **Predecessor:** SPRINT-019 (closed
`3829e75b2`, v12_ms3 = 38.98 TF M=2048, v12s_ks8 = 21.55 TF M=64).

**Headline goal:** answer the SPRINT-019 architectural-decision-point
question with **evidence**, not opinion, AND clear the deployment-readiness
debt that's been accruing since SPRINT-016. Specifically:

1. **Ceiling proof or break**: stand up turbomind's `gemm_bench` standalone
   on gpu-01 and run it at the v12_ms3 champion shape(s). Either it
   exceeds v12_ms3 by ≥ 5 TF at M=2048 (→ commits SPRINT-021 to a
   port) or it lands within ±2 TF (→ v12 family is provably at peak
   for the V100 INT8 GEMM class).
2. **Deployment readiness**: ship the three SPRINT-019 Important
   follow-ups (v12s sanitizer, asymmetric N≠K CLI, per-(M, shape)
   dispatcher) AND validate v12_ms3 against the DSv4-flash MoE expert
   layer catalog AND wire the per-(M, shape) champion table into the
   DSv4 inference path (lmdeploy turbomind backend; pytorch path
   tracked as Open Q5).

**Discipline:** SPRINT-019's no-skip rule carries forward with the
recalibrated v12-family correctness gate (`rel ≤ 1e-2 ∧ p99 ≤ 1.0
∧ maxabs ≤ 5.0`). Median-of-5 measurement. Corrected ncu
`-k regex:...` template (FOLLOWUPS item 4). DCGM-exporter
pre-flight before every ncu run.

Cross-references:
- [SPRINT-020-INTENT.md](./SPRINT-020-INTENT.md) — intent + interview
- [SPRINT-019.md](../SPRINT-019.md) — predecessor
- [SPRINT-019-FOLLOWUPS.md](../SPRINT-019-FOLLOWUPS.md) — items P0
  consumes
- [REPORT-13.md](../../../tools/tc-grid/docs/REPORT-13.md) — ceiling
  rationale; §9 forward paths
- [TURBOMIND-INSIGHTS.md](../../../tools/tc-grid/docs/TURBOMIND-INSIGHTS.md)
  — prior turbomind read; §counterfactual #3 notes the gemm_bench
  resurrection is "1–2 days of build work"
- Memory: `v100_splitk_atomic_pattern`,
  `v100_3stage_register_budget_rule`, `feedback_pre_dequant_defeats_int8`,
  `feedback_effort_estimation_undocumented_hardware` (3× gut estimates
  on opaque tooling — applies HARD to P1 build engineering).

---

## 1. Overview

SPRINT-019 closed with v12_ms3 at 38.98 TF M=2048 (78% of the 50 TF
goal) and v12s_ks8 at 21.55 TF M=64. REPORT-13 §3 names the wall:
v12_ms3 is **mio_throttle-bound** (25.73%) — SMEM bandwidth, not gmem
latency. The `long_scoreboard` stall that dominated v11 was killed
(31% → 0.6%), but the new bottleneck is intrinsic to the v12-family
mainloop. REPORT-13 §9 lists three forward paths and explicitly defers
the choice to SPRINT-020.

This sprint commits to **path 3 (hybrid)** from the intent:

```
P0 — Foundation cleanup (sanitizer + N≠K CLI + dispatch.h + ncu template)
P1 — turbomind gemm_bench standalone build (ceiling reference)
P2 — gemm_bench vs v12_ms3 head-to-head; decision gate for SPRINT-021
P3 — Per-(M, shape) dispatcher + DSv4-flash MoE catalog sweep
P4 — DSv4 inference integration (lmdeploy turbomind backend)
P5 — End-to-end DSv4-flash sample-generation correctness
P6 — Close-out: REPORT-14 + memory updates + SPRINT-020-FOLLOWUPS
```

The architectural-break question (intent §5 Q1, success-criteria path 1)
is answered by P1 + P2 **without** committing to a port. If gemm_bench
reveals headroom, SPRINT-021 is the port sprint. If not, v12 family is
provably at peak.

The deployment-integration question (intent §5 Q5, success-criteria
path 2) is answered by P3 + P4 + P5. These are gated by P0's foundation
cleanup (the N≠K CLI is a hard prerequisite for the MoE catalog).

### 1.1 Headline targets

| Goal | Target |
|---|---|
| **Ceiling answer** | gemm_bench TF at M=2048, N=K=7168 captured; v12_ms3 vs gemm_bench delta documented with ncu-evidenced gap analysis |
| **v12s correctness debt cleared** | `compute-sanitizer --tool {memcheck,racecheck,initcheck}` clean on v12s_ks8 production shape AND adversarial KSPLIT ∈ {2,3,5,8,16} |
| **Asymmetric MoE coverage** | v12_ms3 + v12s benchmarked at 6 DSv4-flash shapes (7168×18944, 18944×7168, 2048×7168, 4096×4096, 7168×7168, 8192×8192) × 8 M values, median-of-5; CSV committed |
| **Per-(M, shape) dispatch rule** | Encoded in `tools/tc-grid/include/dispatch.h`; threshold-adjacent M ∈ {63,65,255,257,1023,1025} verified; correct shape-dependent kernel selection at all 6 shapes |
| **DSv4 integration** | v12_ms3 + v12s invokable from `lmdeploy/turbomind/` INT8 path on the DSv4-flash model; sample-generation correctness vs baseline within rel ≤ 1e-2 token-distribution agreement |
| **No perf regression** | Every shipped (M, shape) cell ≥ v12_ms3 standalone TF (median-of-5) — no degradation from integration overhead |

### 1.2 No-skip rule (carried forward from SPRINT-019 §1.2)

Every commit gate satisfies all 10 items. Applies per phase, per
kernel/integration change, no exceptions. The v12-family gate
(`rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0`) replaces v11's tighter
gate for any v12-derived kernel. v10/v11 stay on the tighter gate.

For **integration phases (P4, P5)**: the gate extends to end-to-end
sample-generation correctness (token distribution KL ≤ 0.05 over a 500-
sample reference set; see §4.5).

### 1.3 What this sprint is NOT

- **NOT** a wholesale turbomind port. P1's scope is *building gemm_bench
  as an external benchmark*, not extracting kernels into our tree.
  Total expected build effort: 8–16 hr (3× the TURBOMIND-INSIGHTS §counterfactual
  #3 1–2 day estimate per `feedback_effort_estimation_undocumented_hardware`).
- **NOT** a CUTLASS 2.11 extension. Defers to SPRINT-021 if P2 surfaces
  headroom.
- **NOT** an INT4 / FP4 revisit. Out of scope.
- **NOT** a new kernel sprint. v12_ms3 + v12s are the production
  champions. P2's outcome may queue a kernel sprint, but it doesn't
  produce one.

---

## 2. Use cases

### 2.1 Production workloads addressed

- **DSv4-flash inference, MoE expert dispatch**: P3's MoE catalog and
  P4's `lmdeploy` integration close the path from kernel champion to
  actual model serving. Per-(M, shape) dispatch handles the variable
  expert geometries (7168×18944 FFN-up, 18944×7168 FFN-down, 2048×7168
  attn-out).
- **DSv4-flash inference, decode-style batches (M ∈ [1, 64])**: v12s_ks8
  ships via the integrated dispatcher; sanitizer-cleared.
- **DSv4-flash inference, prefill / large-batch (M ∈ [1024, 4096])**:
  v12_ms3 ships via the integrated dispatcher.

### 2.2 Non-production use cases

- **gemm_bench standalone**: long-lived external reference for V100
  INT8 GEMM ceiling. Stays in `external/lmdeploy-build/` (pod-local
  build artifact); not vendored into tc-grid.
- **`tools/tc-grid/include/dispatch.h`**: reusable per-(M, shape) rule
  table. Becomes the contract surface between tc-grid (tuner) and
  DSv4 inference (consumer).
- **Asymmetric N≠K tc-grid CLI**: unblocks any future kernel work that
  needs to sweep non-square shapes (every MoE workload).

### 2.3 Out of scope (still deferred to SPRINT-021+)

- Wholesale turbomind GEMM port (conditional on P2 finding headroom).
- CUTLASS 2.11 extension with custom INT8-dequant prologue (same).
- CUTLASS 3.x dependency upgrade (sm_80+ only; not actionable on V100).
- sB SMEM swizzle revisit (FOLLOWUPS item 5) — only relevant if P1/P2
  doesn't already moot it; carried as a stretch if time permits in P2.
- INT4 BN=256 spill fix (SPRINT-019-DEFERRED; condition still not met).
- v5 persistent-CTA revisit (condition not met).
- 96×128 tile exploration.
- BM=192/256 c_frag SMEM-spill rotation (FOLLOWUPS item 6;
  mio_throttle-conditional).
- Pytorch-engine DSv4 integration path (Open Q5; selecting `lmdeploy
  turbomind` for SPRINT-020).

### 2.4 Canonical ncu metric set (carried from SPRINT-019 §2.4)

Same 13 metrics, M ∈ {2048, 4096} for any large-M comparison, M ∈ {64,
256} for small-M comparison. **Use the corrected `-k regex:...`
template** (SPRINT-019-FOLLOWUPS item 4); the `--kernel-id ::name:1`
template is broken.

```
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_atom.sum
launch__registers_per_thread
launch__shared_mem_per_block_static
launch__waves_per_multiprocessor
```

### 2.5 Standardized profiling protocol (corrected)

```bash
ncu -k 'regex:mm_int8_lut_v12(_ms3|s)?' \
    --metrics <13-metric-set> \
    --csv \
    ./build/tc-grid --m-list 2048 --n-list 7168 --k-list 7168 \
                    --dist uniform_small \
    > tools/tc-grid/docs/ncu/SPRINT-020-P<N>-<variant>-M2048.csv
```

**Asymmetric N≠K invocation** (post-P0):

```bash
./build/tc-grid --m-list 2048 --n-list 18944 --k-list 7168 \
                --dist uniform_small
```

**DCGM-exporter pre-flight** (mandatory before every ncu run):

```bash
kubectl get nodes gpu-01 \
  -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.deploy\.dcgm-exporter}'
# expected: paused
```

**Median-of-5**: reuse `scripts/bench-median.sh` from SPRINT-019 P0.

---

## 3. Architecture

### 3.1 Current state (sprint-019 close)

```
tools/tc-grid/
├── kernels/
│   ├── v12_kernels.cuh          ── 3 sibling templates (v12 / v12_ms3 / v12s)
│   ├── mma_sm70.cuh             ── corrected FP16-acc PTX wrapper
│   ├── cutlass_int8_kernels.cuh ── 85 TF pre-dequant ceiling reference
│   └── …v10*, v11*, v3*…        ── retained as historical
├── src/
│   ├── launch_int8.cu           ── version 50/51/60 dispatch (v12/v12_ms3/v12s)
│   ├── main.cu                  ── kTiles[] registration; --m-list / --nk / --dist
│   ├── data_gen.cu              ── A/B/C allocation (assumes N=K today)
│   └── reference.cu             ── CPU baseline
└── docs/                        ── REPORT-1..13, V11-DESIGN.md, V12-DESIGN.md, …

research/lmdeploy/                ── lmdeploy + turbomind source tree (read-only)
└── src/turbomind/kernels/gemm/
    ├── kernel/sm70_884_{4,8,16}.cu   ── sm_70 INT8/INT4/FP8 kernel instantiations
    ├── arch/config_sm70_s884.h       ── tile config types
    ├── arch/mma_sm70.h               ── m8n8k4 wrapper (compare with our mma_sm70.cuh)
    ├── mainloop_sm70.h               ── mainloop variants (compare with v12_kernels)
    ├── test/gemm_bench.cu            ── 87-line nvbench harness
    └── CMakeLists.txt                ── 138-line build; gemm_bench currently
                                          commented out per TURBOMIND-INSIGHTS §counterfactual #3

lmdeploy/turbomind/                ── Python wrapper (target for P4 integration)

deps:
└── cutlass-src @ 2.11.0           ── stay on 2.x; 3.x is sm_80+

Production champions (SPRINT-019 dispatch table, M-only):
  M=1    → v12_ms3 32x128x16_w4         ── 0.20 TF
  M=8    → v12_ms3 64x128x32_w4         ── 1.57 TF
  M=32   → v12_ms3 64x128x32_w4         ── 6.22 TF
  M=64   → v12s    64x128x32_w4 ks8     ── 21.55 TF
  M=256  → v12_ms3 64x128x16_w4         ── 29.00 TF
  M=1024 → v12_ms3 128x128x16_w4        ── 38.96 TF
  M=2048 → v12_ms3 128x128x16_w4        ── 38.98 TF
  M=4096 → v12_ms3 128x128x16_w4        ── 38.61 TF
```

### 3.2 Target architecture (sprint-020 close)

```
tools/tc-grid/
├── include/
│   └── dispatch.h               ── NEW: per-(M, shape) champion table
│                                    + entry-point picker (used by both
│                                    tc-grid AND the DSv4 binding)
├── src/
│   ├── main.cu                  ── --n-list / --k-list / --shape-list flags;
│   │                               outer N×K loop alongside M
│   └── data_gen.cu              ── asymmetric N≠K allocation paths
├── docs/
│   ├── grid-sweep-SPRINT-020-PHASE-{0..5}.csv
│   ├── ncu/SPRINT-020-P{0..5}-*.csv
│   ├── REPORT-14.md             ── close report + ceiling answer
│   └── BENCH-CEILING-V100.md    ── gemm_bench vs v12_ms3 head-to-head
└── tests/
    ├── test_v12s_full_racecheck.cu     ── NEW: full-shape sanitizer test
    └── test_dispatch_threshold.cu      ── NEW: M-threshold dispatch test

external/lmdeploy-build/         ── NEW: pod-local turbomind build out-of-tree
  build/                             (gitignored; CSV outputs committed under
  bin/gemm_bench                      tc-grid/docs/turbomind-bench/)

lmdeploy/turbomind/              ── MODIFIED: INT8 path can dispatch to
  ds_v4_int8_dispatch.{h,cu}        tc-grid's dispatch.h-derived champion table
                                    (pybind11 surface mirrored)

DSv4 inference path:
  serving/dsv4_flash_int8.py     ── NEW: end-to-end DSv4-flash inference using
                                    v12_ms3/v12s for INT8 expert matmuls
```

### 3.3 Why this phase order (P0 → P1 → P2 → P3 → P4 → P5 → P6)

1. **P0 first** because: (a) v12s sanitizer debt is a hard
   correctness-blocking item per SPRINT-019 §1.2 #2 and must clear before
   any v12s-derived production work; (b) the N≠K CLI is a hard
   prerequisite for P3's MoE catalog — DSv4 expert shapes are
   asymmetric; (c) `dispatch.h` is the contract surface for both P3's
   per-(M, shape) rule AND P4's lmdeploy integration — building it
   early keeps both phases consistent; (d) the corrected ncu template
   must replace the broken `--kernel-id ::name:1` in sprint docs before
   any P1/P2 measurement.
2. **P1 next** because gemm_bench is the only thing in the sprint with
   significant build-engineering risk. Front-loading it allows it to
   slip into P2's slack without blocking P3/P4/P5.
3. **P2 between P1 and P3** because P2's outcome is *informational*
   (does SPRINT-021 build a port?) but doesn't gate P3/P4/P5
   deployment work. P2 must run before P6 (REPORT-14 needs the answer)
   but doesn't block the integration track.
4. **P3 → P4 → P5** in order because: P3 builds the dispatch rule and
   per-(M, shape) data; P4 wires it through `lmdeploy/turbomind/`;
   P5 validates end-to-end on actual DSv4-flash inference. Each
   depends on the previous.
5. **P6 last** as close-out.

The two tracks (P1-P2 ceiling, P3-P4-P5 integration) are **independent**
after P0 — they can be parallel if a second worker is available, but
the plan assumes serial execution.

---

## 4. Implementation

Every phase uses the SPRINT-019 five-tier verification structure where
applicable, with **Tier 6 (end-to-end model correctness)** added for
P4-P5:

- **Tier 1 — isolated CPU-reference correctness** (`tests/test_*.cu`)
  with `compute-sanitizer --tool {memcheck,racecheck,initcheck}` as
  applicable.
- **Tier 2 — production integration**: `launch_int8.cu` / `dispatch.h`
  / pybind11 surface.
- **Tier 3 — full M-sweep bit-compare** against v10 across the extended
  M-list and (post-P0) the 6-shape MoE catalog.
- **Tier 4 — performance characterization**: median-of-5 tc-grid + ncu
  at M ∈ {64, 256, 2048, 4096} + CUTLASS Gemm70 ratio.
- **Tier 5 — Nsight Systems timeline** (kern_sum CSV proxy per
  SPRINT-019-FOLLOWUPS item 3; PNG only if GUI is wired).
- **Tier 6 — end-to-end model correctness** (P4-P5 only): DSv4-flash
  sample-generation token-distribution KL vs reference baseline.

### Phase P0 — Foundation cleanup (SPRINT-019 follow-ups + ncu template fix)

**Goal:** clear the three SPRINT-019 Important follow-ups (FOLLOWUPS
items 1, 2, 7) and the broken ncu template (item 4) BEFORE any
kernel-level work. This unblocks every subsequent phase.

**Pre-conditions:**
- `tcg-dev` pod live on gpu-01; CUDA 12.2.2; driver pinned.
- DCGM-exporter paused on gpu-01.
- `_deps/cutlass-src/` populated.
- Working tree clean post-`3829e75b2`.

**Steps:**

1. **P0.1 — `git tag sprint-020-baseline`** at the current HEAD
   (`3829e75b2`). Snapshot v12 family + dispatch state.

2. **P0.2 — v12s compute-sanitizer race + initcheck**
   (FOLLOWUPS item 1):
   - Build `tools/tc-grid/tests/test_v12s_full_racecheck.cu`. Replays
     the v12s_ks8 64×128×32_w4 launch at the production shape (M=64,
     N=K=7168) AND the small-K stress shape (M=64, N=128, K=1024).
   - Adversarial KSPLIT ∈ {2, 3, 5, 8, 16} (covers power-of-two AND
     non-power-of-two boundaries per SPRINT-019 §P3.1).
   - `compute-sanitizer --tool memcheck` AND `--tool racecheck` AND
     `--tool initcheck` — all three mandatory for atomic kernels.
   - **Repeated-launch initcheck**: reset scratch buffer between
     launches; verify no stale data observed.
   - **Decision gate P0.2**: all sanitizers clean across all KSPLIT
     values → proceed. Else: investigate (likely a scratch buffer
     reset bug); v12s does NOT ship to DSv4 integration until clean.

3. **P0.3 — tc-grid asymmetric N≠K CLI**
   (FOLLOWUPS item 2):
   - Modify `tools/tc-grid/src/main.cu`: add `--n-list` and `--k-list`
     flags parallel to `--m-list`. `--nk` remains as the
     shorthand for N=K (back-compatible).
   - Outer loop becomes M × N × K cartesian product unless a
     `--shape-list "n1xk1,n2xk2,..."` flag pins paired (N, K).
   - Modify `tools/tc-grid/src/data_gen.cu`: A/B/C buffer allocation
     accepts independent N and K.
   - Reference CPU path (`reference.cu`) verified for one asymmetric
     shape (e.g., 4×1×16 with N=8, K=16 → known-good).
   - Smoke test: `./build/tc-grid --m-list 64 --n-list 18944 --k-list
     7168 --dist uniform_small` produces correct rel/p99 within v12
     gate.

4. **P0.4 — `tools/tc-grid/include/dispatch.h` skeleton**
   (FOLLOWUPS item 7):
   - Define `struct DispatchKey { int M; int N; int K; }`,
     `struct DispatchEntry { int version; int bm, bn, bk, w, atoms_m,
     atoms_n; int ksplit; }`, and `DispatchEntry pick(DispatchKey)`.
   - Initial table populated from SPRINT-019 close (M-only, the
     current dispatch state). P3 extends with per-(M, shape) entries.
   - Include `dispatch.h` from `launch_int8.cu`'s dispatcher; tc-grid
     still allows explicit `--version` override, but the default path
     consults `dispatch.h::pick()`.
   - **Threshold-adjacent verification**: re-run M ∈ {63, 65, 255,
     257, 1023, 1025} at N=K=7168 (the SPRINT-019 P7.1 set); confirm
     dispatch.h returns the same champion as SPRINT-019.

5. **P0.5 — sprint-template ncu fix**
   (FOLLOWUPS item 4):
   - Update SPRINT-020 (this doc) §2.5 to use `-k regex:...`. ✅ done.
   - Add a short note to `tools/tc-grid/docs/REPORT-14.md` skeleton
     so future sprints reference the corrected template.

6. **P0.6 — Vision-doc decision flag** (intent §5 Q6): no `/vision`
   pass in this sprint. The hybrid path is locally optimal given
   the SPRINT-019 evidence; long-horizon sequencing is best decided
   AFTER P2 surfaces the ceiling answer. Tracked as
   SPRINT-020-FOLLOWUPS item 1.

**Tier coverage:**
- Tier 1 mandatory for P0.2 (sanitizers).
- Tier 3 mandatory for P0.3 (asymmetric shape correctness).
- No Tier 4 perf gate (P0 is foundation, not optimization).

**Decision gate P0:**
- ✅ all four substeps complete; sanitizers clean; asymmetric CLI
  verified bit-correct; `dispatch.h` skeleton in place; threshold
  M values verified → proceed to P1 + P3 in parallel (or serial).
- ❌ sanitizer issue in P0.2 → halt the integration track; debug
  v12s before P3 sees it.

**ETA:** 6–10 hr (foundation work, no novel algorithms).

---

### Phase P1 — turbomind gemm_bench standalone build

**Goal:** stand up `research/lmdeploy/src/turbomind/kernels/gemm/test/
gemm_bench.cu` as an out-of-tree binary on the `tcg-dev` pod, runnable
against the same shapes as tc-grid. **No code is ported into tc-grid**;
gemm_bench stays in `external/lmdeploy-build/` as a sibling reference.

**Risk** (per `feedback_effort_estimation_undocumented_hardware`): the
TURBOMIND-INSIGHTS §counterfactual #3 estimate of "1–2 days" likely
underestimates. Signals: gemm_bench is commented out in the upstream
CMake (a sign it's been broken in main); depends on `nvbench` (extra
dep); transitively depends on the full turbomind GEMM tree (operand.h,
test/models.h, test/testbed.h, ~30+ headers). Expect 8–16 hr (3×
multiplier).

**Pre-conditions:**
- P0 complete.
- `research/lmdeploy/` checked out at a known revision (capture
  `git -C research/lmdeploy rev-parse HEAD`).
- nvcc 12.2 (already in pod).

**Steps:**

1. **P1.1 — Out-of-tree CMake bootstrap**:
   - Create `external/lmdeploy-build/CMakeLists.txt` that
     `add_subdirectory(${TURBOMIND_ROOT})` and exposes only the
     `turbomind::gemm` target plus `gemm_bench` executable.
   - Vendor or fetch `nvbench` (CMake `FetchContent`); pin to a tag
     compatible with CUDA 12.2.
   - Re-enable the gemm_bench target in upstream `CMakeLists.txt`
     either via patch (tracked in `cuda-patches/lmdeploy/gemm-bench.patch`)
     or via local CMake override.
   - sm_70 target arch only (`-arch=sm_70`); skip sm75/sm80/sm90 to
     minimize build time and surface.

2. **P1.2 — First successful build**:
   - Goal: `cmake --build external/lmdeploy-build/build --target
     gemm_bench` produces `external/lmdeploy-build/build/bin/gemm_bench`.
   - Document build flags + nvbench version in
     `tools/tc-grid/docs/BENCH-CEILING-V100.md` §1 (new file).
   - **Time budget**: if > 12 hr cumulative on P1.1+P1.2, scope-reduce.
     Drop nvbench, replace `gemm_bench.cu` with a stripped 30-line
     harness that invokes one `Registry::Run()` cycle and times with
     `cudaEvent_t`. The "1–2 days" estimate from TURBOMIND-INSIGHTS
     §counterfactual #3 assumed it'd build cleanly; if it doesn't,
     reduce scope rather than blow the schedule.

3. **P1.3 — Sanity smoke**:
   - Run `./gemm_bench --axis bs=2048 --axis tp=1 --axis e_num=0 --axis
     e_tok=1 --axis idx=<dsv4-like-config>` (resolve `idx` from
     `test/models.h` config table to the closest 7168×7168 shape).
   - Confirm a TF number prints AND `compute-sanitizer --tool
     memcheck` is clean on one launch (turbomind code may be
     production-grade but we don't know what extra cleanups they do).

4. **P1.4 — Output capture**:
   - Run gemm_bench across M ∈ {64, 256, 2048, 4096} at the closest
     turbomind config to N=K=7168 (the v12_ms3 baseline shape).
   - Capture **median-of-5** TF per M.
   - Commit raw output (textual) under `tools/tc-grid/docs/turbomind-
     bench/gemm_bench-SPRINT-020-P1-base.txt`.

**Tier coverage:**
- Tier 1: `compute-sanitizer --tool memcheck` on first launch.
- Tier 4: median-of-5 perf capture.
- Tier 5: nsys kern_sum CSV for one launch (compare kernel runtimes
  with v12_ms3's nsys).

**Decision gate P1:**
- ✅ gemm_bench builds + runs + produces a TF number AND the launch is
  memcheck-clean → P1 closes; P2 proceeds.
- ⚠️ gemm_bench builds but TF numbers are nonsense (e.g., 0.5 TF) or
  the harness measures the wrong thing → fall back to scope-reduced
  harness (P1.2 escape hatch). Document the deviation.
- ❌ gemm_bench cannot be built within 16 hr (cumulative) → P2's
  ceiling answer becomes "structurally unknown; turbomind reference
  was not reachable in-sprint." REPORT-14 logs this as the answer
  (which is still an answer for SPRINT-021 planning — it says "first
  step of any port attempt is owning gemm_bench infrastructure;
  budget at least a sprint for that alone").

**ETA:** 8–16 hr (3× the TURBOMIND-INSIGHTS estimate). Hard time-box.

---

### Phase P2 — gemm_bench vs v12_ms3 head-to-head + ceiling answer

**Goal:** convert P1's numbers into a definitive answer for the
SPRINT-019 architectural decision (intent §5 Q1 / success-criteria
path 1). Either:
- gemm_bench at M=2048, N=K=7168 ≥ v12_ms3 + 5 TF → **headroom
  proven; SPRINT-021 commits to a turbomind port or CUTLASS
  extension**; OR
- gemm_bench within ±2 TF of v12_ms3 → **v12 family at peak; SPRINT-021
  commits to next-tier work (CUTLASS 3.x dep upgrade investigation
  if H100 becomes available, OR further deployment polish)**.

**Steps:**

1. **P2.1 — Head-to-head sweep**:
   - Run gemm_bench AND tc-grid (v12_ms3 champion) at M ∈ {64, 256,
     2048, 4096}, N=K=7168, median-of-5.
   - Output: `tools/tc-grid/docs/turbomind-bench/comparison-
     SPRINT-020-P2.csv` with columns: M, v12_ms3 TF, gemm_bench TF,
     CUTLASS Gemm70 TF, Δ vs v12_ms3, Δ vs CUTLASS.

2. **P2.2 — ncu diff at M=2048**:
   - Profile gemm_bench at M=2048 with the §2.4 metric set.
   - Profile v12_ms3 at M=2048 (re-use SPRINT-019 P2 numbers or
     re-capture).
   - Diff: which stall is lower in gemm_bench? Expected hypotheses:
     - If gemm_bench's `mio_throttle.pct` is much lower → confirms
       SMEM bandwidth is the v12 wall AND that the dequant prologue
       is the lever (path: SPRINT-021 builds a custom dequant in v13
       or CUTLASS extension).
     - If `long_scoreboard` is much lower (already 0.6% in v12_ms3) →
       turbomind has additional gmem-overlap that we missed; smaller
       gain to chase.
     - If `hmma_cycles_active.pct` rose AND v12_ms3's was already
       32% → confirms the tensor pipe is genuinely the limit at our
       shape (peak FP16-acc on V100 is ~125 TF; we'd be at ~30%
       utilization, which is 38 TF — explains the v12 ceiling).

3. **P2.3 — Write `tools/tc-grid/docs/BENCH-CEILING-V100.md`**:
   - §1 Build environment (nvbench version, turbomind commit,
     CMake flags).
   - §2 Headline table (P2.1).
   - §3 ncu diff (P2.2).
   - §4 Verdict: headroom present? If yes, name the lever (which
     stall dropped most in gemm_bench).
   - §5 SPRINT-021 recommendation: port turbomind / extend CUTLASS /
     accept ceiling.

4. **P2.4 — Optional stretch: sB SMEM swizzle revisit**
   (FOLLOWUPS item 5):
   - If P1 + P2 close in < 12 hr (under budget), AND P2.2 confirms
     mio_throttle is the wall, AND gemm_bench shows headroom, then:
     prototype Swizzle<3,3,3> on sB in v12_ms3 (REPORT-12 §4.2 had
     +14% at BK=32, -1% at BK=16; v12_ms3's new perf profile may
     flip the BK=16 sign).
   - **Hard rule**: ONLY proceed under all three conditions. The
     no-skip rule still applies (tier 1–4); don't ship a swizzle
     prototype without sanitizer + grid sweep + median-of-5.
   - If shipped, this becomes the FIRST SPRINT-020 perf win and may
     close 5–10% of the ceiling gap.

**Tier coverage:**
- Tier 4: P2.1 median-of-5 + ncu.
- Tier 5: P2.2 nsys kern_sum CSV for both runners.
- Tier 1-4 mandatory for P2.4 if pursued.

**Decision gate P2:**
- ✅ Verdict written and committed → P2 closes regardless of which
  way the answer falls. There is NO "ship perf change" requirement
  in P2; the deliverable is the answer itself.
- ⚠️ gemm_bench's shape table doesn't include a 7168×7168 INT8
  exact match → use the closest approximation (e.g., 8192×8192) and
  document the geometry mismatch in BENCH-CEILING-V100.md §1.

**ETA:** 4–6 hr (mostly measurement + writing).

---

### Phase P3 — Per-(M, shape) dispatcher + DSv4 MoE catalog sweep

**Goal:** populate the dispatch table for the 6 DSv4-flash MoE
shapes × 8 M values (= 48 cells) and encode the rule in
`dispatch.h`. This is the deployment-readiness path (success-criteria
path 2 part).

**Pre-conditions:**
- P0.3 (asymmetric N≠K CLI) AND P0.4 (`dispatch.h` skeleton) complete.

**Steps:**

1. **P3.1 — DSv4-flash shape catalog**:
   - Document the 6 canonical shapes in
     `tools/tc-grid/docs/V12-DESIGN.md §5` (new section, extending
     SPRINT-019's §4):
     ```
     (1) 7168 × 18944   FFN-up   (DSv4 expert; N≠K)
     (2) 18944 × 7168   FFN-down (DSv4 expert; N≠K, transposed)
     (3) 2048 × 7168    attn-out (DSv4; N≠K)
     (4) 4096 × 4096    attn-QKV (square, smaller)
     (5) 7168 × 7168    sprint-019 baseline (square)
     (6) 8192 × 8192    square control
     ```
   - Cross-check against DSv4-flash model card / `lmdeploy/turbomind/
     deploy/config.py` to confirm the catalog is the right 6.
     If profiling has surfaced others (intent §5 Q3), extend.

2. **P3.2 — Multi-shape sweep**:
   - Run **v12_ms3 + v12s (+ v10/v11 controls)** across the 6 shapes
     × 8 M values, median-of-5.
   - Output: `tools/tc-grid/docs/grid-sweep-SPRINT-020-PHASE-3.csv`
     with columns: shape (NxK), M, kernel, BM, BN, BK, W, ATOMS_M,
     ATOMS_N, KSPLIT, TF, rel, p99, maxabs.
   - For each (M, shape) cell, the per-cell champion is the kernel
     with the highest TF AND `rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0`.

3. **P3.3 — Per-(M, shape) rule encoding**:
   - For each of the 48 (M, shape) cells, populate the
     `DispatchEntry` in `dispatch.h::kPerShapeTable[]`.
   - If a champion at (M, shape) is `more than 5%` better than the
     M-only champion (i.e., shape matters), the shape-aware entry
     is mandatory. Else, the M-only entry suffices (saves rule
     surface).
   - **Threshold-adjacent verification**: M ∈ {63, 65, 255, 257,
     1023, 1025} × all 6 shapes; confirm `dispatch.h::pick()` returns
     a reasonable kernel at each. Output:
     `tools/tc-grid/docs/grid-sweep-SPRINT-020-PHASE-3-thresholds.csv`.

4. **P3.4 — Correctness re-validation**:
   - Full-rule re-run: invoke each (M, shape) cell via the
     `dispatch.h` path (not explicit `--version`); confirm output
     matches the explicit `--version <champion>` invocation
     bit-for-bit.
   - This catches any rule-table typo (e.g., a shape mapped to the
     wrong KSPLIT).

5. **P3.5 — Performance certification**:
   - Median-of-5 per (M, shape) cell; assert no regression > 2% vs
     the explicit invocation. Document in
     `tools/tc-grid/docs/grid-sweep-SPRINT-020-PHASE-3-certified.csv`.

**Tier coverage:**
- Tier 3: P3.4 (bit-compare via dispatch path).
- Tier 4: P3.2 + P3.5 (median-of-5 + per-cell).
- Tier 5 deferred unless a (M, shape) cell underperforms vs M-only by
  > 5% — then nsys kern_sum to investigate.

**Decision gate P3:**
- ✅ All 48 cells have rule entries; threshold-adjacent dispatch is
  correct; no (M, shape) cell regresses > 2% via the dispatch path
  → P3 closes.
- ⚠️ Some (M, shape) cell falls outside the v12 gate (rel > 1e-2 or
  p99 > 1.0) → that cell ships v10/v11 fallback in `dispatch.h`;
  document in P3.4 output.
- ❌ Some (M, shape) cell catastrophically fails (rel > 1, maxabs >
  100) → debug as a correctness bug; P4 blocked on this cell.

**ETA:** 6–10 hr (most of the time is measurement; 48 cells × 5 runs
× ~30s/run = ~120 min compute, plus ncu-spot-checks where outliers
appear).

---

### Phase P4 — DSv4 inference integration (lmdeploy turbomind backend)

**Goal:** wire `dispatch.h`'s rule into the actual DSv4-flash inference
path (success-criteria path 2 finale). This is the largest single
new-code phase in the sprint.

**Pre-conditions:**
- P3 complete with `dispatch.h` fully populated.

**Steps:**

1. **P4.1 — Integration interface design**:
   - The tc-grid kernels are `__global__` templated CUDA; lmdeploy
     turbomind expects callable C++ functions through its
     `Registry`-style framework. Two integration approaches:
     - **(a) Direct embed**: vendor `tools/tc-grid/kernels/v12_kernels.cuh`
       + `dispatch.h` into `lmdeploy/turbomind/triton_models/.../`,
       expose `dsv4_int8_gemm(stream, A, B, scales, C, M, N, K)`
       as a C entry point, and patch the DSv4 INT8 expert dispatch
       in `lmdeploy/pytorch/models/deepseek_v4.py` (or
       `lmdeploy/turbomind/turbomind.py`) to call it.
     - **(b) External shim**: build `tools/tc-grid/` as a shared
       library; load via `ctypes` from
       `lmdeploy/turbomind/turbomind.py`.
   - **Decision**: prefer (a). Less runtime coupling; gives DSv4
     a single static link.

2. **P4.2 — Stand up the integration surface**:
   - New file `lmdeploy/turbomind/ds_v4_int8_dispatch.h` declaring
     `void dsv4_int8_gemm(...)`.
   - New file `lmdeploy/turbomind/ds_v4_int8_dispatch.cu` that
     `#include`s the relocated v12 kernel headers and calls
     `dispatch.h::pick()`.
   - Patch `lmdeploy/turbomind/setup.py` / CMakeLists so the new
     translation unit compiles with `-arch=sm_70`.
   - Add a pybind11 binding `_turbomind.dsv4_int8_gemm(...)` that
     accepts `torch.Tensor` inputs (via `dlpack`, per the existing
     turbomind pattern in `lmdeploy/turbomind/turbomind.py`).

3. **P4.3 — DSv4 PyTorch hook**:
   - `lmdeploy/pytorch/models/deepseek_v4.py` (or the equivalent
     INT8 MoE-expert dispatch point) calls `_turbomind.
     dsv4_int8_gemm` when the model config requests v12-family kernels.
   - Feature-flag this at runtime: `model_config.use_v12_int8 = True`
     to opt in; default OFF in this sprint (Tier 6 validates the
     opt-in path before flipping the default).

4. **P4.4 — Integration smoke test**:
   - Standalone Python test (`tests/integration/test_dsv4_int8_v12.py`):
     - Construct a 1-layer DSv4-flash expert at the (7168×18944)
       FFN-up shape.
     - Feed a fixed seed batch through both the existing reference
       INT8 path AND the v12-family path.
     - Assert per-element `rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0`
       (v12 gate, per `feedback_pre_dequant_defeats_int8` — INT8
       gmem stays INT8 in both paths).
   - `compute-sanitizer --tool memcheck` on the integration entry
     point.

5. **P4.5 — Per-shape sanity certification**:
   - Repeat P4.4 for each of the 6 DSv4 shapes × M ∈ {64, 2048}.
   - Output: `tools/tc-grid/docs/grid-sweep-SPRINT-020-PHASE-4-
     integration.csv`.

**Tier coverage:**
- Tier 1: P4.4 sanitizer.
- Tier 2: P4.2 + P4.3.
- Tier 3: P4.4 + P4.5 bit-compare per shape.

**Decision gate P4:**
- ✅ All 6 shapes × {64, 2048} M values cleared the v12 gate via the
  integration path → P4 closes; P5 proceeds.
- ❌ Any cell fails the gate (especially: row-major vs col-major B
  mismatch, or scale-tile misalignment between tc-grid's data_gen
  and the DSv4 model's actual scale layout) → debug the
  data-layout assumption mismatch before P5.

**ETA:** 10–16 hr (pybind11 + DSv4 model wiring + data-layout
matching are the time sinks).

---

### Phase P5 — End-to-end DSv4-flash sample-generation correctness

**Goal:** prove the integrated v12 family produces correct
DSv4-flash outputs at the model level, not just kernel-output level
(success-criteria path 2 finale).

**Pre-conditions:**
- P4 complete with `model_config.use_v12_int8 = True` opt-in path.

**Steps:**

1. **P5.1 — Reference baseline**:
   - With `use_v12_int8 = False`, generate 500 sample completions
     from a fixed prompt set on DSv4-flash INT8. Record token
     sequences AND logit distributions for the first 32 tokens of
     each completion. Output: `tests/integration/dsv4_baseline.pkl`.

2. **P5.2 — v12 path**:
   - With `use_v12_int8 = True`, run the same 500 prompts. Record
     same outputs. Output: `tests/integration/dsv4_v12.pkl`.

3. **P5.3 — Token-distribution comparison**:
   - KL divergence over the first-32-token logit distributions:
     `KL(P_v12 || P_baseline) ≤ 0.05` per prompt (mean) AND ≤ 0.15
     at p99.
   - Token-sequence exact-match for low-temperature generation
     (temperature ≤ 0.1): ≥ 95% of completions match baseline.
   - For higher-temperature generation: token-sequence Hamming
     distance ≤ 5% per completion.
   - Output: `tools/tc-grid/docs/SPRINT-020-P5-correctness.csv`.

4. **P5.4 — Speed certification**:
   - Time the 500-sample generation under both configs; v12 path
     median-of-3 ≤ baseline median-of-3 + 5% (i.e., no slowdown).
     Target: v12 path 5–15% FASTER at decode-heavy generation
     (where v12s_ks8 dominates).

5. **P5.5 — Default-flag flip** (conditional):
   - If P5.3 + P5.4 pass, flip `model_config.use_v12_int8 = True` as
     the default for DSv4-flash. Else: keep the opt-in path; document
     the gap in REPORT-14.

**Tier coverage:**
- Tier 6 ONLY (model correctness; kernel correctness was P3/P4).

**Decision gate P5:**
- ✅ Token-distribution KL within thresholds AND no slowdown → flip
  default.
- ⚠️ KL within threshold AND modest slowdown (< 10%) → keep opt-in,
  document for SPRINT-021 perf tuning.
- ❌ KL outside threshold → debug as a numerical-accumulation issue
  (likely v12 gate vs DSv4 quality gate mismatch); document and
  flag for SPRINT-021.

**ETA:** 4–8 hr (mostly inference-time + analysis).

---

### Phase P6 — Close-out: REPORT-14 + memory updates + FOLLOWUPS

**Steps:**

1. **P6.1 — REPORT-14.md**:
   Mirror REPORT-13 structure:
   1. Headline + per-(M, shape) champion table + integration status.
   2. Commit history.
   3. P0 cleanup outcomes (sanitizer / N≠K / dispatch.h).
   4. P1 + P2 ceiling answer + recommended SPRINT-021 path.
   5. P3 MoE catalog grid + per-cell champion rule.
   6. P4 integration shape correctness.
   7. P5 model-level KL + slowdown numbers.
   8. CUTLASS Gemm70 ratio at the new dispatch path's hot cells.
   9. **Architectural-decision answer** (intent §5 Q1): explicit
      verdict on turbomind port / CUTLASS extension / deployment-
      only.
   10. What's left on the table (forward levers for SPRINT-021).

2. **P6.2 — Memory updates**:
   - If P2.2 confirmed mio_throttle is the v12 wall AND gemm_bench
     showed headroom: append a memory file
     `v100_int8_v12_ceiling_diagnosis.md` with the ncu evidence.
   - If P4 surfaced a data-layout mismatch worth remembering: new
     memory `v100_v12_to_lmdeploy_layout_contract.md`.
   - Update MEMORY.md index.

3. **P6.3 — SPRINT-020-FOLLOWUPS.md**:
   - Capture: vision-doc deferral (P0.6); any P4/P5 quality-gate
     surprises; any P2.4 swizzle prototype that didn't ship;
     SPRINT-021 recommendation (port / extension / ceiling-accept).

4. **P6.4 — SPRINT-020-DEFERRED.md**:
   - Carry forward: INT4 BN=256, v5 persistent-CTA, 96×128 tile,
     CUTLASS 3.x dep upgrade, sB swizzle revisit (if P2.4 not run),
     BM=192/256 spill rotation. Mark conditions for each.

5. **P6.5 — Working-tree audit**:
   - `git status` clean except for committed artifacts.
   - `tools/tc-grid/include/dispatch.h` is the canonical dispatch
     surface; legacy explicit `--version` paths in `launch_int8.cu`
     either delete or comment `// historical, kept for testing`.

**ETA:** 4–6 hr.

---

## 5. Files Summary

### New files

| Path | Purpose |
|---|---|
| `tools/tc-grid/include/dispatch.h` | Per-(M, shape) champion rule (P0.4, P3.3) |
| `tools/tc-grid/tests/test_v12s_full_racecheck.cu` | v12s atomic sanitizers (P0.2) |
| `tools/tc-grid/tests/test_dispatch_threshold.cu` | M-threshold dispatch test (P3.3) |
| `tools/tc-grid/docs/BENCH-CEILING-V100.md` | gemm_bench vs v12_ms3 (P1, P2) |
| `tools/tc-grid/docs/REPORT-14.md` | Sprint close (P6.1) |
| `tools/tc-grid/docs/turbomind-bench/gemm_bench-SPRINT-020-P1-base.txt` | Raw gemm_bench output (P1.4) |
| `tools/tc-grid/docs/turbomind-bench/comparison-SPRINT-020-P2.csv` | Head-to-head (P2.1) |
| `tools/tc-grid/docs/grid-sweep-SPRINT-020-PHASE-{0..5}.csv` | Per-phase sweeps |
| `tools/tc-grid/docs/ncu/SPRINT-020-P{0..5}-*.csv` | Per-phase ncu exports |
| `tools/tc-grid/docs/SPRINT-020-P5-correctness.csv` | DSv4 KL + Hamming (P5.3) |
| `external/lmdeploy-build/CMakeLists.txt` | Out-of-tree gemm_bench build (P1.1) |
| `cuda-patches/lmdeploy/gemm-bench.patch` | Re-enable gemm_bench upstream (P1.1) |
| `lmdeploy/turbomind/ds_v4_int8_dispatch.h` | Integration interface (P4.2) |
| `lmdeploy/turbomind/ds_v4_int8_dispatch.cu` | Integration implementation (P4.2) |
| `tests/integration/test_dsv4_int8_v12.py` | Standalone integration test (P4.4) |
| `tests/integration/dsv4_baseline.pkl` | Reference output for P5.1 |
| `tests/integration/dsv4_v12.pkl` | v12-path output for P5.2 |
| `docs/sprints/SPRINT-020-FOLLOWUPS.md` | Discovered follow-ups (P6.3) |
| `docs/sprints/SPRINT-020-DEFERRED.md` | Carried-forward deferred items (P6.4) |

### Modified files

| Path | Change |
|---|---|
| `tools/tc-grid/src/main.cu` | `--n-list`, `--k-list`, `--shape-list` (P0.3); consult `dispatch.h::pick()` for default kernel (P0.4) |
| `tools/tc-grid/src/data_gen.cu` | Asymmetric N≠K buffer alloc (P0.3) |
| `tools/tc-grid/src/launch_int8.cu` | Route through `dispatch.h::pick()` by default; explicit `--version` retained for testing (P0.4) |
| `tools/tc-grid/docs/V12-DESIGN.md` | §5 MoE catalog (P3.1) |
| `lmdeploy/turbomind/turbomind.py` | Pybind binding for `dsv4_int8_gemm` (P4.2) |
| `lmdeploy/turbomind/setup.py` or CMake | New TU compilation (P4.2) |
| `lmdeploy/pytorch/models/deepseek_v4.py` | Optional v12 dispatch hook (P4.3) |

### Pod-local artifacts (gitignored; CSV outputs committed)

- `external/lmdeploy-build/build/` — turbomind+gemm_bench build tree.

---

## 6. Definition of Done

### Per phase (every commit gate)

The 10-item SPRINT-019 §1.2 no-skip rule, with the **v12-family
recalibrated gate** (`rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0`) for v12-
derived kernels. **No phase-specific relaxations.** For Tier 6
(P4-P5), the gate extends to token-distribution KL ≤ 0.05 (mean) /
≤ 0.15 (p99) on the 500-sample reference set.

Commit message includes (where applicable):
- Headline TF at affected M(s), median-of-5.
- Target stall before/after %.
- v12/CUTLASS ratio at the new cell (if applicable).
- For P5: KL summary and slowdown ratio.

If any item fails: change is reverted from default dispatch; reason
logged in REPORT-14.

### Sprint close

1. **P0 follow-ups closed**: v12s sanitizer clean; N≠K CLI verified;
   `dispatch.h` skeleton + threshold M values verified; ncu template
   fix in sprint template.
2. **Ceiling answer published** in `BENCH-CEILING-V100.md` (P2.3);
   verdict committed.
3. **Per-(M, shape) dispatcher** in `dispatch.h` covering all 48 cells
   (6 shapes × 8 M values); no cell regresses > 2% via the
   dispatch path.
4. **DSv4 integration shape correctness** (P4): all 6 shapes × {64,
   2048} M values pass v12 gate via the integration path.
5. **DSv4 end-to-end correctness** (P5): 500-sample KL within
   thresholds; slowdown ≤ 5%.
6. **REPORT-14.md** published.
7. **Memory updates** committed.
8. **SPRINT-020-FOLLOWUPS.md** + `SPRINT-020-DEFERRED.md** captured.
9. **No new ptxas spill warnings** in any shipped kernel.
10. **Working tree clean**: `git status` shows only committed artifacts.

---

## 7. Risks

### R1 — gemm_bench build collapses under time pressure
gemm_bench is commented out upstream; nvbench is an extra dep;
transitive turbomind tree is large. The TURBOMIND-INSIGHTS §counterfactual
#3 "1–2 days" estimate may be optimistic.
**Mitigation**: Hard 16-hr time-box on P1 (3× the original estimate
per `feedback_effort_estimation_undocumented_hardware`). Escape hatch
in P1.2: drop nvbench, replace with a 30-line cudaEvent harness.
P2 still proceeds even with a stripped harness — the ceiling answer
is what matters, not the harness sophistication.
**Severity**: HIGH (gates the headline ceiling answer).

### R2 — gemm_bench's shape catalog doesn't include 7168×7168 INT8
turbomind's `test/models.h` config table is built for their model
catalog (Llama / Qwen variants); INT8 at exactly N=K=7168 may not
appear.
**Mitigation**: Use closest match (8192×8192 INT8 is likely in the
table); document geometry mismatch in BENCH-CEILING-V100.md §1.
TF / element scales linearly with K so ceiling-question answer
remains directionally valid.
**Severity**: LOW.

### R3 — v12s sanitizer surfaces a real race
The deferred sanitizer run could uncover a bug (uninitialized scratch
buffer, KSPLIT-boundary atomic ordering issue, …) that requires a
v12s fix.
**Mitigation**: P0 front-loads the work so a fix has time to land
before P3 sees v12s. If a fix is non-trivial, fall back to v10s_ks8
at M=64 in the dispatcher and ship without v12s. This loses 5%
M=64 perf vs SPRINT-019 close but doesn't block the sprint.
**Severity**: MEDIUM.

### R4 — Asymmetric N≠K reveals a v12_ms3 dispatch hole
v12_ms3 was only sprint-019 validated on square shapes. The
asymmetric MoE shapes may surface bank-conflict pathologies or
register-pressure issues that didn't appear at N=K.
**Mitigation**: P3.2's full grid sweep covers all 48 cells; failures
get v10/v11 fallback in `dispatch.h`. The dispatch surface is
designed to allow per-cell kernel choice.
**Severity**: MEDIUM.

### R5 — DSv4 INT8 data layout mismatch
`lmdeploy/turbomind` may pack INT8 weights in a different memory
layout than tc-grid's `data_gen.cu` assumes (row-major B vs
col-major B; scale tile interleave; group_size mismatch with
tc-grid's QK_INT8=32 vs turbomind's group_size=128).
**Mitigation**: P4.4 standalone integration test catches this BEFORE
P5 end-to-end. The first integration assertion is "tc-grid output ==
lmdeploy current INT8 path output" — any divergence here is a
layout bug. Carry a `feedback_pre_dequant_defeats_int8` discipline:
INT8 gmem stays INT8, dequant happens in SMEM.
**Severity**: HIGH (could consume P4 budget).

### R6 — Token-distribution KL exceeds threshold (P5)
v12 family recalibrated to `rel ≤ 1e-2` (looser than v11's `1e-3`);
KL on DSv4-flash logits may exceed 0.05 due to FP16-acc accumulation
noise.
**Mitigation**: P5.3 KL threshold (0.05 mean, 0.15 p99) is generous
relative to v12 gate. If exceeded, the rel-vs-KL mapping is the
new diagnostic; investigate which (M, shape) cell drove KL up. May
need per-cell tighter gate or v11 fallback at quality-sensitive shapes.
**Severity**: MEDIUM.

### R7 — lmdeploy build coupling
Adding a new translation unit to `lmdeploy/turbomind/` requires
rebuilding the pybind11 extension. If the build system is brittle
(per `CLAUDE.md` in lmdeploy: "Controlled via setup.py + CMake…
DISABLE_TURBOMIND…"), iteration time could balloon.
**Mitigation**: P4.2 budgets 16 hr; if iteration time exceeds 60s
per build, scope-reduce to ctypes-shim integration (path (b) in
P4.1). Slower at runtime but faster to iterate; can be tightened
in SPRINT-021.
**Severity**: MEDIUM.

### R8 — Reference DSv4 baseline is unstable
P5.1's 500-sample baseline depends on a fixed seed AND a stable
inference path. If the existing DSv4-flash INT8 inference path has
nondeterminism (e.g., atomic accumulations elsewhere), the baseline
itself drifts run-to-run.
**Mitigation**: P5.1 runs the baseline 2× (different seeds) and
asserts the baseline-vs-baseline KL is < 0.01 (much tighter than the
v12-vs-baseline threshold). If baseline self-KL exceeds 0.01, the
KL comparison is invalid — fall back to token-sequence Hamming
distance only.
**Severity**: LOW.

### R9 — Scope creep
The sprint has 7 phases + 2 tracks. Realistic effort budget below
shows 42–72 hr; sprint cadence is 30–50 hr per SPRINT-019.
**Mitigation**: Phase-level decision gates allow partial-ship.
Minimum viable close: P0 + P1 + P2 + P3 ship → sprint partial-
success (ceiling answered, dispatch encoded, but no
integration). P4 + P5 can defer to SPRINT-021 if budget runs out.
**Severity**: HIGH.

### R10 — gpu-01 contention
Same as SPRINT-019 R7. gpu-02-4090rtx unavailable.
**Mitigation**: `kubectl get pods -n llm -o wide` before each sweep.
**Severity**: LOW.

### R11 — DCGM-exporter re-enabled
Same as SPRINT-019 R6. ncu unreliable if DCGM-exporter reads
concurrently.
**Mitigation**: Pre-flight check in every ncu run (§2.5).
**Severity**: LOW.

### R12 — Measurement noise on 1–2% gates
Same as SPRINT-019 R9.
**Mitigation**: Median-of-5 per gate; escalate to median-of-9 if
variance exceeds 1%.
**Severity**: LOW.

### R13 — `dispatch.h` API drift
The `pick()` API is consumed by tc-grid AND by
`lmdeploy/turbomind/ds_v4_int8_dispatch.cu`. If P3 changes the API
after P4 has already wired the pybind surface, the integration
breaks.
**Mitigation**: P0.4 freezes the `DispatchKey` / `DispatchEntry`
signature BEFORE P3 starts populating the table. P3 only writes
data; API stays stable.
**Severity**: LOW.

### R14 — gemm_bench result is geometrically biased
Even with a close shape match, gemm_bench tests its own configs;
the result may favor turbomind's tile choices over ours. Comparing
TFLOPS apples-to-apples is harder than comparing for a fixed shape.
**Mitigation**: Compare TFLOPS PLUS ncu stall breakdowns (P2.2). If
turbomind's TFLOPS is higher but its mio_throttle is similar, the
gain is in tile selection (chase that). If its mio_throttle is
dramatically lower, the gain is in dequant / mainloop restructure
(SPRINT-021 lever).
**Severity**: LOW.

### R15 — gemm_bench's tile choice doesn't include the v12_ms3 tile
turbomind might not test `128x128x16_w4` (v12_ms3's champion); their
sm70_884_8.cu lists `(128, 128, 16, 2, 2, 1)` and `(64, 128, 32, 1,
4, 1)` — different W/atoms decomposition.
**Mitigation**: Treat gemm_bench's *best tile at the shape* as the
ceiling. We don't need apples-to-apples tiles; we need ceiling.
**Severity**: LOW.

### R16 — DSv4-flash model not actually deployed in pod
The DSv4-flash INT8 weights may not be in the pod's
`models/` directory. P4-P5 work assumes the inference path is
operable.
**Mitigation**: P0.0 pre-flight (added below): verify a baseline
DSv4-flash inference run succeeds before sprint kicks off. If
weights missing: download/rsync as P0 sub-step; budget for it.
**Severity**: MEDIUM.

### R17 — Memory writes flagged as "save in memory" before evidence in
hand
Per `feedback_dont_skip_plan_steps`, the sprint plan's evidence chain
should ship before any new memory writes (e.g., a "v12 ceiling
diagnosis" memory). Don't write speculation as memory.
**Mitigation**: P6.2 conditions every new memory write on P2.2 /
P4.4 evidence having landed in commits.
**Severity**: LOW (process discipline).

---

## 8. Security

Same surface area as SPRINT-019. Summary:
- No external network for kernel work. CUTLASS vendored.
- P1 adds an external dep (`nvbench` via `FetchContent`); document the
  pinned version in BENCH-CEILING-V100.md §1.
- No credentials. kubectl via pre-authenticated kubeconfig.
- No data exfil. Synthetic distributions for kernel work; P5 uses an
  internal prompt set (no PII).
- Sanitizer coverage: memcheck on every new kernel; racecheck +
  initcheck on atomic kernels (P0.2).
- Inline-asm constraints: no new PTX wrappers in this sprint (v12 is
  the production set); P4 binding code is C++ only.
- `lmdeploy/turbomind/ds_v4_int8_dispatch.{h,cu}` are NOT shipped
  upstream; private fork only per AGENTS.md.

---

## 9. Dependencies

### Hardware
- gpu-01 V100 32 GB sm_70, sole V100 in cluster.
- `tcg-dev` pod in `llm` namespace, source at `/src/tools/tc-grid`.
- CUDA 12.2.2, driver pinned to sprint-019 version.
- gpu-02-4090rtx unavailable (qwen3-moe-rotorquant).

### Tooling
- `ncu` (Nsight Compute), `nsys` (Nsight Systems).
- `compute-sanitizer` with memcheck, racecheck, initcheck.
- `cuobjdump` for SASS inspection.
- `ptxas --verbose` for register/spill report.
- `nvbench` (P1; pinned via FetchContent).
- `cmake` ≥ 3.20.

### Code
- `_deps/cutlass-src/` at v2.11.0 (sprint-018 pinned).
- `kernels/v12_kernels.cuh` champions (v12, v12_ms3, v12s).
- `kernels/mma_sm70.cuh` FP16-acc wrapper.
- `research/lmdeploy/` at a captured revision (P1.1 records the
  rev-parse HEAD).
- `lmdeploy/turbomind/` pybind11 surface.

### Workflow
- Laptop → pod sync via rsync.
- DCGM-exporter pause before every ncu run.
- DSv4-flash INT8 weights in pod's `models/` (R16 pre-flight).

---

## 10. Open Questions

### Q1 — Architectural direction (intent §5 Q1): PROPOSED PATH AND ALTERNATIVES

**Proposed: hybrid path 3 (this draft).**

P1+P2 stand up gemm_bench standalone for a definitive ceiling answer
in 12–22 hr. P3+P4+P5 ship the v12 family into DSv4-flash production
in 20–34 hr. Total 32–56 hr, within sprint cadence.

**Rationale for choosing this over the alternatives:**

- **Alternative A — Wholesale turbomind GEMM port** (intent path 1a
  "architectural break: turbomind"): a port means extracting
  ~30+ headers from `research/lmdeploy/src/turbomind/kernels/gemm/`
  into our tree, resolving their CUTLASS-style template tree, wiring
  it through tc-grid's launch path, and validating bit-correctness.
  Realistic estimate: 30–50 hr just for porting + validation, with
  HIGH risk of subtle correctness drift (per
  `feedback_effort_estimation_undocumented_hardware`, multiply by 3
  given opaque ISA boundaries and CUTLASS template depth → 50–80 hr).
  Doing this WITHOUT a ceiling check first means we don't know
  whether the port will actually produce a TF improvement — we might
  spend a sprint discovering that turbomind also caps at ~40 TF. The
  hybrid path runs the ceiling check FIRST (P1+P2) for 1/3 the cost.
  If P2's answer is "headroom exists," SPRINT-021 commits to the
  port with evidence. If "no headroom," we never spent the porting
  time at all.

- **Alternative B — CUTLASS 2.11 extension with custom INT8 dequant
  prologue** (intent path 1b): CUTLASS 2.11 doesn't expose the
  prologue stage cleanly; per REPORT-13 §5.4, "CUTLASS 3.x focuses
  on sm_80+; staying on 2.x is the V100 path." A 2.11 extension
  requires hand-writing a custom
  `cutlass::gemm::device::GemmUniversal<...>` partial specialization,
  threading through a dequant-and-mma fused stage. Estimated effort:
  20–40 hr; risk: as high as the port (CUTLASS 2.x doc surface is
  thin). Has the same "did we leave perf on the table?" question
  unanswered — equally needs the ceiling check first. Defer to
  SPRINT-021 conditional on P2 outcome.

- **Alternative C — Pure deployment integration** (intent path 2):
  ships P0+P3+P4+P5; skips P1+P2. This is the lowest-risk path and
  guaranteed to satisfy intent §5 success-criteria path 2. But it
  leaves the ceiling question unanswered, which means SPRINT-021
  starts in the same uncertainty SPRINT-020 inherited from
  SPRINT-019. The 12–22 hr P1+P2 investment is small relative to
  the value of clearing the architectural question.

- **Alternative D — Pure gemm_bench / pure CUTLASS extension** (no
  integration): ceiling-only sprint. Leaves the
  deployment-readiness debt accruing for another sprint. Given the
  P0 follow-ups (sanitizer, N≠K CLI, dispatcher rule) are already
  blocking other work, "no integration" is worse than the hybrid.

**Conditions that would change the proposed path:**
- If gpu-01 is rebuilt / replaced before sprint kickoff →
  re-evaluate; some H100 / A100 access changes the calculus toward
  CUTLASS 3.x.
- If DSv4-flash production deployment is unblocked by other work
  before sprint kickoff → drop P4+P5; pure ceiling sprint (P0+P1+P2)
  with the remaining time on sB swizzle (FOLLOWUPS item 5) and
  prep for SPRINT-021 port.
- If a user-side strong opinion lands (e.g., "deployment is
  P0-urgent, ceiling is N/A") → drop P1+P2; pure deployment sprint.

### Q2 — gemm_bench shape coverage

`research/lmdeploy/src/turbomind/kernels/gemm/test/models.h` defines
the `config` table that drives gemm_bench's `idx` axis. Does this
table include INT8 at the v12_ms3 shape (7168 × 7168)? If not, the
closest match (likely 8192 × 8192 or a model-derived shape) is used,
and the ceiling comparison is approximate. Resolve by reading
`models.h` early in P1 and documenting the chosen `idx` in
BENCH-CEILING-V100.md §1.

### Q3 — MoE shape catalog completeness (intent §5 Q3)

The 6 shapes in §3.1 / P3.1 are derived from REPORT-13 §9 candidates
+ SPRINT-019 P6.1 list. If DSv4-flash profiling has surfaced other
critical N×K combinations (e.g., a 14336× shape from a different
expert size), extend mid-sprint. This is a low-risk extension once
the N≠K CLI lands in P0.3.

### Q4 — Sanitizer follow-up scope (intent §5 Q4)

v12s only in this sprint. Other atomic kernels (v10s, v3s) carry
their own sprint-historical sanitizer status; if any are
production-active they should be re-checked, but doing so is out of
scope for SPRINT-020. Tracked as SPRINT-020-FOLLOWUPS deferred item.

### Q5 — Deployment-integration target (intent §5 Q5)

**Proposed: `lmdeploy/turbomind` backend** (P4 design). The pytorch
backend (`lmdeploy/pytorch/models/deepseek_v4.py`) is the alternative;
its kernel-dispatch surface is `lmdeploy/pytorch/kernels/` (Triton +
CUDA) and the integration would be a Python-side dispatch hook rather
than a pybind11 binding.

**Rationale**: the turbomind backend already has CUDA kernels in its
pipeline and a stable pybind11 surface (`lmdeploy/turbomind/turbomind.py`
~800 lines). Adding a new TU and binding is mechanical. The pytorch
backend would require either writing a Triton wrapper around v12
kernels (translation overhead) or a separate CUDA extension with
its own build machinery. For SPRINT-020's scope, turbomind backend
is the cheaper integration.

**Conditions that would switch to pytorch backend:**
- If the DSv4-flash deployment target is pytorch-backend-only at
  the user's site → switch.
- If `DISABLE_TURBOMIND` is set in pod env by default → switch.
- If turbomind upstream rejects sm_70 maintenance in a future
  version → switch.

Resolve in P0 pre-flight (R16); if turbomind backend doesn't run
DSv4-flash in our pod today, the integration target IS pytorch by
default.

### Q6 — Vision document deferral (intent §5 Q6)

Proposed: defer to SPRINT-021. The architectural decision SPRINT-020
answers (P2 verdict) is upstream of any roadmap commitments; running
`/vision` BEFORE P2 would lock in the wrong sequence. Run `/vision`
at the SPRINT-020 close, with P2's answer in hand.

### Q7 — Dispatcher version-number allocation

SPRINT-019 reserved: 50 (v12), 51 (v12_ms3), 60 (v12s), 52 reserved
unused (v12_bm192 closed negative). SPRINT-020's `dispatch.h` is
a higher-level surface ABOVE the version numbers — it picks (M, N,
K) → (version, BM, BN, BK, W, atoms, ksplit). No new version-number
allocations expected. If P2.4 ships an sB-swizzle v12_ms3 variant
as a stretch, allocate version=53.

### Q8 — REPORT-14 numbering

Assumed REPORT-14 is the close report (matches REPORT-9/10/11/12/13
sequence). No conflict expected.

---

## 11. Estimated total effort

| Phase | ETA | Notes |
|---|---:|---|
| P0 (foundation cleanup: sanitizer + N≠K CLI + dispatch.h + ncu fix) | 6–10 hr | Front-loaded; gates everything |
| P1 (turbomind gemm_bench standalone build) | 8–16 hr | 3× the TURBOMIND-INSIGHTS estimate; hard time-box |
| P2 (gemm_bench head-to-head + ceiling answer) | 4–6 hr | Measurement + writing |
| P3 (per-(M, shape) dispatcher + MoE catalog sweep) | 6–10 hr | 48 cells × median-of-5 |
| P4 (DSv4 inference integration via lmdeploy turbomind) | 10–16 hr | pybind + data-layout matching |
| P5 (end-to-end DSv4-flash sample correctness) | 4–8 hr | Inference + KL analysis |
| P6 (close: REPORT-14 + memory + FOLLOWUPS + DEFERRED) | 4–6 hr | |
| **Total** | **42–72 hr (~5–8 sessions)** | |

**Within-cadence minimum-viable close** (R9 mitigation): if the budget
runs at 50 hr, the minimum is P0 + P1 + P2 + P3 (~24–42 hr) — ceiling
question answered AND dispatcher rule encoded — with P4-P5 deferred
to SPRINT-021. This satisfies intent §5 success-criteria path 3
(hybrid bench-only + sanitizer + N≠K + asymmetric MoE).

---

## 12. Success / partial-success / failure summary

**Sprint succeeds if:**
- P0-P5 all close per their decision gates.
- BENCH-CEILING-V100.md publishes a definitive verdict.
- `dispatch.h` covers all 48 (M, shape) cells.
- DSv4-flash sample-generation KL within thresholds.
- REPORT-14 published.

**Sprint partially succeeds if:**
- P0 + P1 + P2 + P3 close (ceiling answered + dispatcher encoded),
  P4-P5 deferred. This is the minimum-viable close.
- OR: P1 fails to build gemm_bench within 16 hr but P0 + P3 + P4 +
  P5 close (deployment path only; ceiling answer is "structurally
  unknown; budget a dedicated SPRINT-021 sub-phase to own the
  turbomind build infrastructure before any port attempt").

**Sprint fails if:**
- P0 sanitizer surfaces an unfixable v12s race (correctness
  failure; production champion has to revert to v10s_ks8).
- OR: P4 integration reveals a fundamental data-layout mismatch
  between tc-grid kernels and DSv4 production (full
  refactor required; one-sprint scope inadequate).
- OR: any phase skipped without explicit user authorization (per
  `feedback_dont_skip_plan_steps`). Methodology failure regardless
  of TF / KL outcome.

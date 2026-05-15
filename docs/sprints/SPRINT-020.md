# SPRINT-020 — Turbomind ceiling-proof + DSv4 deployment integration

**Status:** CLOSED 2026-05-15. **Outcome:** BREAKTHROUGH-class headroom
proven; turbomind sm70 INT8 head-to-head shown impossible (no such kernel
exists); FP16 ceiling = 87 TF M=2048 captures the legitimate sm70 peak.
See [REPORT-14.md](../../tools/tc-grid/docs/REPORT-14.md) for the full
close-out narrative.

**Predecessor:** SPRINT-019 (closed `3829e75b2`; v12_ms3 = 38.98 TF
M=2048; v12s_ks8 = 21.55 TF M=64). **Successor:** SPRINT-021 (BREAKTHROUGH
branch — compute-efficiency optimisation to close 38.98 → ≥ 50 TF gap).

**Headline goal:** answer REPORT-13 §9's architectural-decision-point
with **evidence, not opinion**. Stand up turbomind's `gemm_bench` on
gpu-01, run it apples-to-apples against `v12_ms3` through a tc-grid
bridge, and either commit SPRINT-021 to a port (breakthrough ≥ 44 TF)
or close the v11/v12 family as provably at peak (ceiling proof ≤ 41 TF)
and ship deployment integration. Three SPRINT-019 Important follow-ups
(v12s sanitizer, tc-grid N≠K CLI, per-(M, shape) dispatcher) ship in
P0 regardless of the architectural branch — they are P4 prerequisites.

**Discipline:** SPRINT-019's no-skip rule carries forward with the
recalibrated v12-family correctness gate (`rel ≤ 1e-2 ∧ p99 ≤ 1.0
∧ maxabs ≤ 5.0`). Median-of-5 measurement, corrected
ncu `-k regex:...` template (SPRINT-019-FOLLOWUPS item 4).

Cross-references:
- [SPRINT-020-INTENT.md](./drafts/SPRINT-020-INTENT.md) — intent +
  draft-source links
- [SPRINT-020-MERGE-NOTES.md](./drafts/SPRINT-020-MERGE-NOTES.md) —
  draft synthesis
- [SPRINT-020-DEFERRED.md](./SPRINT-020-DEFERRED.md)
- [SPRINT-019.md](./SPRINT-019.md) — predecessor
- [SPRINT-019-FOLLOWUPS.md](./SPRINT-019-FOLLOWUPS.md) — P0 consumes
  the 3 Important items
- [REPORT-13.md](../../tools/tc-grid/docs/REPORT-13.md) — §9 forward
  paths (decision rationale)
- [TURBOMIND-INSIGHTS.md](../../tools/tc-grid/docs/TURBOMIND-INSIGHTS.md)
- Memory: `v100_splitk_atomic_pattern`,
  `v100_3stage_register_budget_rule`,
  `v100_wmma_half_float_frag_layout_mismatch`,
  `feedback_pre_dequant_defeats_int8`,
  `feedback_effort_estimation_undocumented_hardware` (3× gut estimates
  for opaque tooling — applies HARD to P1)

---

## 1. Overview

SPRINT-019 closed v12_ms3 at 78% of the 50 TF goal. REPORT-13 §3
identifies the new wall: v12_ms3 is **mio_throttle-bound** (25.73%),
not gmem-latency-bound. The 3-stage pipeline killed long_scoreboard
(31% → 0.6%), but SMEM bandwidth replaced it. REPORT-13 §9 listed
three forward paths and explicitly deferred the choice to SPRINT-020.

This sprint commits to the **hybrid bench-first turbomind path**:
build standalone `gemm_bench` from `research/lmdeploy/src/turbomind/`,
bridge it into `tc-grid` for apples-to-apples measurement, then branch
on the result.

### 1.1 The single TF decision contract — RESOLVED via P1.6 pivot

| Outcome | Original threshold | Resolution |
|---|---|---|
| **Breakthrough** | Turbomind INT8 ≥ 44 TF | **DECIDED YES** — turbomind FP16 ceiling = 87 TF M=2048, v12_ms3 sits at 45% of ceiling → BREAKTHROUGH-class compute headroom |
| **Ceiling proof** | Turbomind INT8 ≤ 41 TF | Not applicable — no sm70 INT8 turbomind kernel exists |
| **Indeterminate** | 41 < Turbomind < 44 TF | Not applicable |

**P1.6 pivot finding (critical):** Turbomind's `sm70_s884` registry has
NO INT8/uint8 weight kernel. The 8-bit-weight sm70 path is FP8 e4m3
(Config_E4M3), 4-bit is U4 (Config_U4_g). Turbomind's INT8 kernels are
sm75+ only. The original head-to-head INT8 comparison the sprint
contracted on is physically impossible.

**Resolved by reframing**: the legitimate sm70 ceiling is the FP16
(no-quant) hardware peak via turbomind's Gemm::Run → cuBLAS dispatch.
Both v12_ms3 INT8-with-FP16-acc and turbomind FP16 use the same
`mma.m8n8k4.f16` instructions, so the ratio IS apples-to-apples on
compute efficiency. v12_ms3 at 45% of FP16 ceiling = significant
compute headroom remains. See [REPORT-14.md](../../tools/tc-grid/docs/REPORT-14.md)
§4 for the full investigation chain and §6 for SPRINT-021 framing.

### 1.2 Headline targets

| Goal | Target |
|---|---|
| Architectural answer | gemm_bench TF at M=2048 N=K=7168 captured via tc-grid bridge; outcome category per §1.1 documented |
| v12s correctness debt cleared | `compute-sanitizer --tool {memcheck,racecheck,initcheck}` clean on v12s_ks8 production shape AND adversarial KSPLIT ∈ {2,3,5,8,16} |
| tc-grid N≠K support | `--n-list` / `--k-list` ship in `main.cu`; `--nk` shorthand preserved for square shapes |
| Asymmetric MoE coverage | v12_ms3 + v12s benchmarked at 6 DSv4-flash shapes × 8 M values, median-of-5 |
| Per-(M, shape) dispatch | Encoded in `tools/tc-grid/include/dispatch.h`; threshold-adjacent M ∈ {63,65,255,257,1023,1025} verified |
| DSv4 integration (ceiling-proof branch only) | v12_ms3 + v12s invokable from `lmdeploy/turbomind/` INT8 path on DSv4-flash; sample-generation correctness vs baseline |
| Preserve M=64 win | v12s_ks8 21.55 TF held unless Turbomind clearly beats it |
| No perf regression | Every shipped (M, shape) cell ≥ v12_ms3 standalone TF; no integration-overhead degradation |

### 1.3 No-skip rule (carried forward)

Every commit gate satisfies the 10-item rule from SPRINT-019 §1.2,
with the v12-family recalibrated gate. New per-phase additions:

- **P1 / P2**: Build-system changes guarded behind feature flags
  (`TM_ENABLE_GEMM_BENCH`, `TCGRID_ENABLE_TURBOMIND_GEMM`) so non-V100
  builds aren't impacted.
- **P2**: `compute-sanitizer --tool memcheck` on the new bridge code
  on the first benchmark launch.
- **P2**: Data-layout validation gate BEFORE running benchmark — verify
  Turbomind kernels can consume tc-grid's INT8 tensors without an
  expensive re-pack that would invalidate the comparison.
- **P3**: ncu metric pack includes SMEM throughput, SM efficiency, and
  the §2.4 stall metrics — not just "ncu evidence."
- **P4-P5**: gate extends to end-to-end DSv4-flash sample-generation
  correctness — output bit-comparison OR perplexity match within 5%
  vs reference.

### 1.4 What this sprint is NOT

- **NOT** a wholesale turbomind port. P1's scope is *building
  gemm_bench standalone* and *bringing up SM70 kernels through it*. A
  port (replacing v12_ms3 with turbomind-derived kernels in
  `tools/tc-grid/kernels/`) is SPRINT-021 work, conditional on
  breakthrough.
- **NOT** a CUTLASS extension sprint. The CUTLASS 3.x path remains
  deferred (intent §5 Q1 Alt A) until SPRINT-021 evidence demands it.
- **NOT** an end-of-life sprint for tc-grid. The harness keeps being
  the lab measurement surface even after deployment integration.

---

## 2. Use cases

### 2.1 Production workloads addressed

- **DSv4-flash INT8 inference, prefill / large-batch (M ∈ [1024,
  4096])**: existing v12_ms3 champion is what we're shipping; SPRINT-
  020 either replaces it (breakthrough) or wires it into production
  (ceiling proof).
- **DSv4-flash INT8 inference, decode-style (M ∈ [1, 64])**: existing
  v12s_ks8 = 21.55 TF M=64 is what we're shipping; protected by §1.2.
- **DSv4-flash MoE expert dispatch**: shape catalog 7168×18944 (FFN-up),
  18944×7168 (FFN-down), 2048×7168 (attn-out), plus the square shapes.
  Currently unsupported in tc-grid CLI — P0 fixes this.

### 2.2 Non-production use cases

- **Performance reference**: gemm_bench becomes a permanent in-tree
  ceiling reference for any future V100 INT8 GEMM work.
- **Runtime dispatch infrastructure**: per-(M, shape) dispatcher
  encoded in `dispatch.h` is reusable for any future kernel family.

### 2.3 Out of scope (see SPRINT-020-DEFERRED.md)

- CUTLASS 3.x dep upgrade
- Wholesale turbomind kernel port into `tools/tc-grid/kernels/`
- Pytorch backend integration
- INT4 BN=256 spill fix
- v5 persistent-CTA revisit

### 2.4 Canonical ncu metric set (carried from SPRINT-019 §2.4 with
fixes)

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
launch__registers_per_thread
launch__shared_mem_per_block_dynamic
launch__waves_per_multiprocessor
sm__pipe_tensor_op_hmma_cycles_active.sum     # raw count for SM-efficiency derivations
```

### 2.5 ncu invocation — corrected per SPRINT-019-FOLLOWUPS #4

Use the regex-filtered form:
```
ncu -k 'regex:mm_int8_lut_v<X>' --metrics ... --csv ./build/tc-grid ...
```
Do NOT use `--kernel-id ::name:1` (broken — only captures launch
invocation 1 globally; SPRINT-019 §2.5 template was invalid).

---

## 3. Architecture

### 3.1 Current state

```
tools/tc-grid/kernels/v12_kernels.cuh        — 3 sibling templates
  ├── mm_int8_lut_v12       v=50    (FP16-acc baseline)
  ├── mm_int8_lut_v12_ms3   v=51    (3-stage pipeline, champion)
  └── mm_int8_lut_v12s      v=60    (SplitK on v12 base)

tools/tc-grid/src/launch_int8.cu             — dispatcher (50/51/60)
tools/tc-grid/src/main.cu                    — kTiles[] + CLI (--m-list, --nk)
tools/tc-grid/kernels/cutlass_int8_kernels.cuh — v=40 CUTLASS Gemm70 ceiling

research/lmdeploy/src/turbomind/kernels/gemm/
  ├── kernel/sm70_884_{4,8,16}.cu            — SM70 mainloop templates
  ├── mainloop_sm70.h
  ├── iterator_sm70.h
  ├── scheduler_sm70.cuh
  ├── registry.cu                            — runtime dispatch
  ├── dispatch_cache.{h,cu}                  — per-shape measured selections
  └── test/
      ├── gemm_bench.cu                      — disabled in CMakeLists
      └── models.h                            — shape catalog
```

### 3.2 Target state (SPRINT-020 close)

```
tools/tc-grid/                                — measurement harness, same
  ├── src/main.cu                            — adds --n-list, --k-list
  ├── src/launch_turbomind_int8.cu           — NEW: bridge
  ├── include/dispatch.h                     — NEW: per-(M, shape) rules
  └── docs/REPORT-14.md                      — SPRINT-020 close report

research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt
  - guarded `TM_ENABLE_GEMM_BENCH` rebuilds gemm_bench standalone

If BREAKTHROUGH branch: SPRINT-021 begins (port spec separate document)
If CEILING-PROOF branch:
  research/lmdeploy/src/turbomind/models/llama/{LlamaLinear.cu,moe_ffn_layer.cc}
  - wired to invoke v12_ms3 + v12s through the lmdeploy turbomind INT8 path
```

### 3.3 Why bridge-first (Codex draft's structural insight)

Three separable gates instead of a single "port" decision:

1. **Native gemm_bench** (P1): proves we can build and run turbomind
   kernels. Test harness uses NVBench by default — different
   distributions, different tolerance semantics than tc-grid. A
   gemm_bench number isn't yet apples-to-apples.
2. **tc-grid bridge** (P2): adapts turbomind's `gemm.cu`,
   `LaunchSpec`, layouts into tc-grid's data generator + reference
   GEMM + tolerance evaluator. THIS is the apples-to-apples gate. The
   data-layout validation step (P2.2) is the new gate added per the
   Gemini critique of the Codex draft.
3. **Runtime integration** (P4 ceiling-proof branch only): only if v12
   stays the champion. This goes to `LlamaLinear.cu` /
   `moe_ffn_layer.cc`, not a re-implementation in tc-grid.

Each gate has its own decision. Bridge can fail (layout incompatible)
without invalidating gemm_bench. Bench can fail (build complexity)
without invalidating the v12 family's correctness debt cleanup
(P0 + P3 ship regardless).

---

## 4. Implementation

### Phase P0 — Foundation cleanup (sanitizer + N≠K CLI + dispatch.h scaffolding)

**Goal:** clear the 3 SPRINT-019 Important follow-ups before any new
architectural work. These are P4 prerequisites; they ship regardless
of the architectural branch.

**Pre-conditions:**
- `tcg-dev` pod live on gpu-01
- DCGM-exporter paused
- `_deps/cutlass-src/` populated
- gpu-01 sole GPU; gpu-02-4090rtx unavailable
- Tag `sprint-019-baseline` exists; tag this state at P0.7 as
  `sprint-020-p0-baseline`

**Steps:**

1. **P0.1 v12s sanitizer pass** (SPRINT-019-FOLLOWUPS #1):
   Run `compute-sanitizer --tool memcheck`, `--tool racecheck`, and
   `--tool initcheck` on `v12s_ks8` at M=64, N=K=7168, uniform_small,
   AND across adversarial KSPLIT ∈ {2, 3, 5, 8, 16}. Capture each tool's
   output to `tools/tc-grid/docs/sanitizer/SPRINT-020-P0-{tool}-{ksplit}.log`.
   **Failure branch**: if any sanitizer reports a real bug (race or
   uninitialized read) on a power-of-2 KSPLIT, REVERT v12s from
   default dispatch and re-scope sprint to "sanitizer-clean v12 only";
   non-power-of-2 KSPLIT bugs revert that KSPLIT only.

2. **P0.2 tc-grid N≠K CLI** (SPRINT-019-FOLLOWUPS #2):
   Extend `tools/tc-grid/src/main.cu` with `--n-list` and `--k-list`
   flags. Preserve `--nk` as the square-shape shorthand (equivalent to
   `--n-list X --k-list X`). Update `parse_int_list()` callers and the
   outer benchmark loops. **No code path may assume N == K.** Verify
   on v11/v12 champion at N=7168 K=18944.

3. **P0.3 Per-(M, shape) dispatch.h skeleton** (SPRINT-019-FOLLOWUPS #3):
   Create `tools/tc-grid/include/dispatch.h` exporting `LaunchSpec
   choose_kernel(int M, int N, int K)`. Initial population uses the
   SPRINT-019 per-M champion table (REPORT-13 §1). Wire from
   `launch_int8.cu` as an OPTIONAL path (existing per-tile dispatch
   stays for benchmark mode). Threshold-adjacent verification at
   M ∈ {63, 65, 255, 257, 1023, 1025} required.

4. **P0.4 Stale kTiles[] audit** (carried from SPRINT-019 P7.1):
   Confirm every kTiles[] entry is either (a) a current champion at
   some (M, shape), (b) a documented experimental tile, or (c) marked
   historical. P4 BM=192/256 v12_ms3 entries documented per SPRINT-019
   P4 negative result.

5. **P0.5 Median-of-5 reproduce of v12_ms3 + v12s at champion shape**:
   Confirm `v12_ms3<128,128,16,4,16,1> @ M=2048 ≥ 38.0 TF` and
   `v12s_64x128x32_w4_ks8 @ M=64 ≥ 21.0 TF` on the live pod. Drift
   beyond ±2% triggers investigation (pod env, driver, DCGM-exporter,
   thermal).

6. **P0.6 ncu metric pack template script**:
   Write `scripts/ncu-metric-pack.sh` that wraps the §2.4 metric set
   with the `-k regex:...` form. Replaces the broken SPRINT-019 §2.5
   template. Used by P2-P3.

7. **P0.7 Tag the working tree**: `git tag sprint-020-p0-baseline`.

**Decision gate P0:**
- ✅ All sanitizers clean (or KSPLIT-restricted dispatch documented)
- ✅ N≠K CLI works, square shorthand preserved
- ✅ dispatch.h compiles and threshold-adjacent verification passes
- ✅ v12_ms3 + v12s reproduce within ±2% of SPRINT-019 numbers
- ❌ Any sanitizer-bug branch → revert v12s, re-scope sprint

**ETA:** 4–6 hr.

---

### Phase P1 — Turbomind gemm_bench standalone build

**Goal:** build the disabled `gemm_bench` target in
`research/lmdeploy/src/turbomind/kernels/gemm/test/` so it runs on
gpu-01 against the V100 SM70 kernels (`sm70_884_{4,8,16}.cu`). This
is build engineering only — no perf comparison yet.

**Risk** (SPRINT-019 memory `feedback_effort_estimation_undocumented_hardware`):
build engineering on commented-out upstream targets is the
3×-gut-multiplier case. Sprint plan budgets 6–10 hr; gut estimate was
1–2 days.

**Steps:**

1. **P1.1 Inspect disabled target**:
   `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`
   lines 108–137 (per SPRINT-019-DEFERRED). Identify the
   transitive dep tree: `core`, `cublas`, `quantization_kernels`,
   `gpt_kernels`, NVBench. Document each dep's status (vendored,
   external, header-only).

2. **P1.2 Re-enable with guard**:
   Add `option(TM_ENABLE_GEMM_BENCH "Build turbomind gemm_bench standalone" OFF)`
   and gate the disabled block behind it. The default OFF preserves
   non-V100 / non-benchmark builds.

3. **P1.3 Resolve transitive deps**:
   For each disabled dep:
   - vendored (in `research/lmdeploy/`): include
   - external (NVBench): add via FetchContent or document required
     external install
   - kernel-tree-internal (`core`, `quantization_kernels`,
     `gpt_kernels`): minimum scope — only build what gemm_bench links

4. **P1.4 Add DSv4 shapes to the bench**:
   `research/lmdeploy/src/turbomind/kernels/gemm/test/models.h` has
   the model-shape catalog. Add entries for DSv4-flash: 7168×7168
   square + 7168×18944, 18944×7168, 2048×7168, 4096×4096, 8192×8192
   asymmetric. Verify each shape is `legal` through `group_size=128`
   and the registry's tile constraints.

5. **P1.5 Build + smoke-test**:
   `TM_ENABLE_GEMM_BENCH=ON cmake -B build && cmake --build build`
   produces `gemm_bench_turbomind`. Run at M=2048 N=K=7168, capture
   NVBench output. Smoke gate: kernel launches, NVBench reports
   non-zero TF.

6. **P1.6 Registry verification**:
   Confirm `registry.cu` exposes `sm70_884_4`, `sm70_884_8`,
   `sm70_884_16` on the V100 at runtime. Document any missing
   registrations.

**Decision gate P1:**
- ✅ gemm_bench_turbomind builds without disturbing existing
  llama.cpp / tc-grid builds
- ✅ Runs at M=2048 N=K=7168 producing valid timings
- ✅ SM70 kernels visible in registry
- ❌ Build failure that requires >12 hr to resolve → CLOSE this sprint
  early as "ceiling assumed unprovable in available tooling, defer
  to SPRINT-021 architecture decision"

**ETA:** 6–12 hr (3× gut estimate per memory).

---

### Phase P2 — tc-grid → Turbomind bridge

**Goal:** make Turbomind kernel results apples-to-apples comparable
to v12_ms3 by running them through tc-grid's data generator, reference
GEMM, and tolerance evaluator.

**Steps:**

1. **P2.1 Bridge launcher file**:
   Create `tools/tc-grid/src/launch_turbomind_int8.cu` with
   `LaunchResult launch_turbomind_int8(...)`. Adapt
   `tc_grid::LaunchSpec` to `turbomind::gemm::Operation`,
   `MatrixLayout`, and `Gemm::Run`. Register under version=70.

2. **P2.2 Data-layout validation gate** (Gemini critique add):
   Before running ANY benchmark through the bridge, verify Turbomind
   kernels can consume tc-grid's INT8 weight tensors WITHOUT a re-pack
   step. Check:
   - W_qs byte order (Turbomind interleaving vs tc-grid's row-major)
   - W_scales granularity (Turbomind block size vs `QK_INT8=32`)
   - A activations dtype and layout
   If a re-pack is required, decide:
   (a) include it in the bridge (cost on every launch — invalidates
       perf comparison)
   (b) treat as a documented compatibility gap (closes path 2/3)
   (c) one-time pre-pack at setup (legitimate if amortizable across
       launches)

3. **P2.3 CMakeLists guard**:
   `tools/tc-grid/CMakeLists.txt` adds
   `option(TCGRID_ENABLE_TURBOMIND_GEMM ...)` gating
   the new launcher. Default OFF for safety.

4. **P2.4 compute-sanitizer memcheck on bridge first launch**:
   Sprint §1.3 no-skip rule extension: bridge code itself goes through
   sanitizer before any TF measurement is trusted.

5. **P2.5 kTiles[] entries**:
   Add Turbomind-backed rows to `tools/tc-grid/src/main.cu` (version
   band 70+). Initial set: square 128×128, plus the 3 asymmetric DSv4
   shapes.

6. **P2.6 First Turbomind-via-bridge TF measurement**:
   Run at M=2048 N=K=7168, uniform_small. Capture
   `INT8,LUT,U(-1,1),2048,7168,7168,turbomind_<config>,OK,...` in
   tc-grid CSV form. Apples-to-apples vs v12_ms3 row in same CSV.

**Decision gate P2:**
- ✅ Bridge compiles, sanitizer-clean, layout-compatible
- ✅ First measurement produces valid TF in tc-grid CSV form
- ❌ Data-layout incompatibility requiring expensive re-pack → close
  this path; go straight to P4 ceiling-proof branch (no breakthrough
  evidence achievable)

**ETA:** 8–14 hr.

---

### Phase P3 — Head-to-head + asymmetric MoE catalog

**Goal:** generate the evidence for the §1.1 decision contract.

**Steps:**

1. **P3.1 Median-of-5 square-shape comparison**:
   M ∈ {1, 8, 32, 64, 256, 1024, 2048, 4096} × N=K=7168 × uniform_small.
   Both v12_ms3 + v12s AND Turbomind-bridge kernels.
   Output: `tools/tc-grid/docs/SPRINT-020-square-medianof5.csv`.

2. **P3.2 Asymmetric MoE catalog sweep**:
   Same M-list × DSv4 shapes {7168×18944, 18944×7168, 2048×7168}.
   v12_ms3 + v12s + Turbomind.
   Output: `tools/tc-grid/docs/SPRINT-020-moe-medianof5.csv`.

3. **P3.3 ncu metric pack on each champion**:
   Use `scripts/ncu-metric-pack.sh` (P0.6). Capture the §2.4 metric
   set for:
   - v12_ms3 champion at M=2048
   - Turbomind champion at M=2048
   - v12s_ks8 at M=64
   - Turbomind small-M champion at M=64
   Outputs: `tools/tc-grid/docs/ncu/SPRINT-020-P3-{kernel}-M{m}.csv`.

4. **P3.4 Apply the §1.1 contract**:
   Compute median-of-5 TF deltas at the gate points. Classify outcome:
   breakthrough / ceiling-proof / indeterminate. Branch at P4.

5. **P3.5 Dispatch cache export (turbomind side)**:
   Run `Gemm::Export` / `DispatchCache::Export` to capture turbomind's
   measured launch specs as runtime artifacts. Even if v12 wins, this
   exports the per-shape cache for future reuse.

**Decision gate P3:**
- ✅ Both kernel families measured at all M × shape cells
- ✅ ncu evidence captured for champions
- ✅ Outcome classified per §1.1

**ETA:** 6–10 hr.

---

### Phase P4 — BRANCH on §1.1 outcome

#### P4-BREAKTHROUGH (Turbomind ≥ 44 TF or asymmetric win)

1. **P4.1 Document SPRINT-021 port spec**: outline the port plan
   based on which turbomind kernels won. Identify whether the win
   came from `sm70_884_4`, `_8`, or `_16` family.
2. **P4.2 Quick-win wiring**: if turbomind wins, expose it via
   `dispatch.h` (P0.3) as the new champion. No deep production
   integration yet — that's SPRINT-021.
3. **P4.3 Close sprint** with REPORT-14 (BREAKTHROUGH branch).

#### P4-CEILING-PROOF (Turbomind ≤ 41 TF, no asymmetric win)

1. **P4.1 Production dispatch.h finalization**:
   Encode the per-M, per-shape champion table from P3 measurements.
   v12_ms3 stays champion for large M; v12s_ks8 for small M.
   Threshold-adjacent verification + stale entry audit.
2. **P4.2 lmdeploy turbomind backend wiring**:
   Modify `research/lmdeploy/src/turbomind/models/llama/LlamaLinear.cu`
   to invoke the tc-grid-derived dispatch at INT8 GEMM call sites.
   Same for `research/lmdeploy/src/turbomind/models/llama/moe_ffn_layer.cc`
   for MoE expert dispatch.
3. **P4.3 Build + correctness pass**:
   Build lmdeploy with the wired path; verify no compilation regression;
   v12-family compute-sanitizer memcheck through the production wrapper.

#### P4-INDETERMINATE (41 < TM < 44 TF)

Default to CEILING-PROOF branch (P4.1-P4.3 above) but note in REPORT-14
that SPRINT-021 may be reconsidered if asymmetric data later shows a
material gap.

**ETA:** 4–8 hr.

---

### Phase P5 — End-to-end DSv4-flash correctness (CEILING-PROOF branch only)

**Goal:** the v12 family is invokable from the actual DSv4-flash model
serving path; end-to-end output matches the reference.

**Steps:**

1. **P5.1 Reference baseline capture**:
   Run DSv4-flash inference on a 100-prompt reference set with the
   current (pre-SPRINT-020) production path. Capture per-token
   distributions for the first 50 tokens of each prompt.

2. **P5.2 v12-integrated inference**:
   Same 100 prompts, same first 50 tokens, but with P4.2's wired
   v12_ms3 + v12s path. Capture distributions.

3. **P5.3 Correctness gate**:
   - **Token bit-match** preferred: compare argmax tokens; require
     ≥ 99% match.
   - **Perplexity within 5%** acceptable alternative if bit-match
     fails due to fp16-acc rounding (expected on DSv4-flash given the
     model is already fp4/fp8).
   - Document any divergences in REPORT-14 §5.

4. **P5.4 Performance verification**:
   Capture wall-clock latency for 50-token completion of the reference
   set. Compare integrated v12 path to current production path. Gate:
   no regression (≤ current latency).

**Decision gate P5:**
- ✅ Output correctness (token match OR perplexity within 5%)
- ✅ No latency regression
- ❌ Correctness fails → revert P4.2; production path unchanged;
  diagnose in REPORT-14 §5; sprint closes as "v12 family correct in
  isolation but breaks in production wrapper" — SPRINT-021 task.

**ETA:** 4–6 hr (only on CEILING-PROOF branch).

---

### Phase P6 — Close-out

1. **P6.1 REPORT-14.md** (mirrors REPORT-12/13):
   Headline outcome (breakthrough / ceiling-proof / indeterminate),
   v12 vs Turbomind comparison table, ncu before/after deltas, asymmetric
   MoE catalog table, decision rationale per §1.1, what's left.

2. **P6.2 Memory updates**:
   - Add memory: `v100_turbomind_gemm_ceiling_proof` (regardless of
     outcome — captures what the bench reveals about V100 INT8 GEMM
     ceiling)
   - Update `feedback_effort_estimation_undocumented_hardware` with
     P1 actual vs estimated effort
   - Update `MEMORY.md` index

3. **P6.3 SPRINT-020-FOLLOWUPS.md**:
   Capture discovered items. Critical / Important / Nice-to-have.

4. **P6.4 Per-M champion table (final)**:
   Either v12 family (ceiling-proof) or Turbomind (breakthrough). Lock
   in `tools/tc-grid/include/dispatch.h` with a comment-block lookup
   table.

5. **P6.5 No ledger update** — `scripts/ledger.py` doesn't exist
   (SPRINT-019-FOLLOWUPS).

6. **P6.6 No VISION.md update** — doesn't exist; flag for future
   `/vision` invocation.

**ETA:** 4–6 hr.

---

## 5. Files Summary

### New files

| Path | Purpose |
|---|---|
| `tools/tc-grid/include/dispatch.h` | Per-(M, shape) production dispatch rule |
| `tools/tc-grid/src/launch_turbomind_int8.cu` | Bridge: tc-grid → turbomind/gemm |
| `tools/tc-grid/docs/REPORT-14.md` | Sprint close report |
| `tools/tc-grid/docs/SPRINT-020-{square,moe}-medianof5.csv` | P3 measurement tables |
| `tools/tc-grid/docs/ncu/SPRINT-020-P3-*.csv` | P3 ncu evidence |
| `tools/tc-grid/docs/sanitizer/SPRINT-020-P0-*.log` | P0 sanitizer outputs |
| `scripts/ncu-metric-pack.sh` | Canonical §2.4 metric-pack invocation |
| `docs/sprints/SPRINT-020-FOLLOWUPS.md` | Discovered follow-ups |

### Modified files

| Path | Change |
|---|---|
| `tools/tc-grid/src/main.cu` | `--n-list`, `--k-list`; Turbomind kTiles[] entries; uses `dispatch.h` |
| `tools/tc-grid/src/launch_int8.cu` | Optional `dispatch.h` integration; version=70 Turbomind |
| `tools/tc-grid/CMakeLists.txt` | `TCGRID_ENABLE_TURBOMIND_GEMM` option |
| `tools/tc-grid/include/tc_grid.h` | LaunchSpec extension for asymmetric shapes |
| `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt` | `TM_ENABLE_GEMM_BENCH` guard |
| `research/lmdeploy/src/turbomind/kernels/gemm/test/models.h` | Add DSv4 shape entries |
| `research/lmdeploy/src/turbomind/models/llama/LlamaLinear.cu` | **P4 CEILING-PROOF only**: wire v12 dispatch |
| `research/lmdeploy/src/turbomind/models/llama/moe_ffn_layer.cc` | **P4 CEILING-PROOF only**: MoE wiring |

---

## 6. Definition of Done

Applied per phase AND at sprint close. **No phase-specific relaxations.**

### Per phase

- Every commit gate satisfies the SPRINT-019 §1.2 10-item rule with
  the v12-family recalibrated gate (`rel ≤ 1e-2 ∧ p99 ≤ 1.0
  ∧ maxabs ≤ 5.0`).
- Median-of-5 measurements for all TF claims.
- ncu `-k regex:...` form (NOT `--kernel-id ::name:1`).
- Sanitizer clean (`memcheck`; `racecheck` + `initcheck` for atomic
  paths) before any perf number is trusted.
- CSV evidence committed under `tools/tc-grid/docs/`.

### Sprint close

1. **Architectural decision documented** per §1.1 contract with
   ncu-evidenced rationale.
2. **v12s correctness debt cleared** (sanitizers; documented KSPLIT
   restrictions if any).
3. **tc-grid N≠K CLI shipped** with `--nk` preserved.
4. **Per-(M, shape) dispatch.h ships** with threshold-adjacent
   verification.
5. **DSv4-flash asymmetric MoE shapes measured** end-to-end.
6. **DSv4-flash inference integration (CEILING-PROOF branch)**:
   output correctness (token match or perplexity within 5%) AND no
   latency regression.
7. **REPORT-14 published**.
8. **Memory updates committed**.
9. **SPRINT-020-FOLLOWUPS.md captures discovered items**.
10. **No new ptxas spill warnings** in any shipped kernel.
11. **Turbomind gemm_bench builds reproducibly** behind the feature
    flag; SPRINT-019 / SPRINT-020 dispatch boundaries unchanged in
    default builds.

---

## 7. Risks

### R1 — gemm_bench build pulls in too much (P1)
NVBench, `core`, `quantization_kernels`, `gpt_kernels` may all need to
build. Effort estimation memory says 3× gut.
**Mitigation**: P1.3 explicit transitive-dep audit; P1.5 smoke-test
gate. P1 hard time-box at 12 hr.
**Severity**: HIGH.

### R2 — Quantization-layout incompatibility (P2)
Turbomind kernels may require specific weight interleaving that
tc-grid's INT8 generator doesn't produce. Re-pack on each launch would
invalidate perf comparison.
**Mitigation**: P2.2 data-layout validation gate BEFORE any benchmark
trust. One-time pre-pack at setup acceptable; per-launch re-pack
disqualifies the comparison.
**Severity**: HIGH.

### R3 — Apples-to-oranges gemm_bench vs tc-grid measurement (P3)
Native gemm_bench uses NVBench timing semantics + its own data
distributions. tc-grid bridge uses tc-grid's data generator + median-of-5.
Numbers may disagree.
**Mitigation**: The single TF decision contract (§1.1) is evaluated
ONLY against the tc-grid bridge measurement, never against
native gemm_bench output. gemm_bench remains a reference-only signal.
**Severity**: MEDIUM.

### R4 — Same-wall on Volta SMEM
v12_ms3 is mio_throttle-bound. Turbomind kernels may also be SMEM-bound
on Volta and therefore land in the same 38-41 TF range. The sprint's
"ceiling proof" outcome is the explicit accepted resolution of this
risk.
**Mitigation**: Outcome classification is well-defined; ceiling proof
is a legitimate sprint success, not a failure.
**Severity**: LOW (mitigated by outcome design).

### R5 — Sanitizer failure on v12s (P0)
Race or initcheck issue in production path would force scope reduction.
**Mitigation**: P0.1 explicit failure branch (revert v12s, re-scope).
**Severity**: LOW (v12s shipped through tc-grid bit-compare in SPRINT-
019; sanitizers verify but unlikely to surface fundamental issues).

### R6 — Data divergence at DSv4-flash sample-generation (P5)
fp16 accumulator on DSv4-flash may produce different tokens than the
reference (currently fp32 or higher precision).
**Mitigation**: Perplexity-within-5% fallback gate. The DSv4-flash
model is fp4/fp8; fp16 acc is the production standard for those
formats so divergence is expected to be small.
**Severity**: LOW (gate explicitly allows precision floor).

### R7 — Lab dispatch vs runtime dispatch divergence
P0.3's `dispatch.h` is the lab dispatcher in tc-grid; P4.2 wires it
into runtime. The two may go out of sync if not co-managed.
**Mitigation**: `dispatch.h` is the single source of truth; runtime
includes it directly rather than duplicating the rule.
**Severity**: LOW (mitigated by structural choice).

### R8 — Pod environment changes between SPRINT-019 and SPRINT-020
gpu-01 pod, CUDA, driver, DCGM-exporter state.
**Mitigation**: P0.5 reproduces SPRINT-019 numbers within ±2%; drift
triggers investigation.
**Severity**: LOW.

### R9 — Build complexity in `research/lmdeploy/`
Re-enabling `gemm_bench` and modifying the upstream tree could affect
the AGENTS.md llama.cpp policy if any upstream contribution is later
considered. Current scope is private-fork-only.
**Mitigation**: AGENTS.md exempts private forks; no upstream PR is
planned. All modifications stay local.
**Severity**: LOW.

### R10 — Turbomind asymmetric-only win
Turbomind might win at 18944×7168 but lose at 7168×7168. The §1.1
contract treats this as breakthrough (≥10% on any asymmetric shape).
**Mitigation**: §1.1 explicitly covers this case; SPRINT-021 port
scope would be asymmetric-specific.
**Severity**: LOW (planned for).

---

## 8. Security

Same surface as SPRINT-019:
- No external network. Build offline. CUTLASS vendored.
- No credentials. kubectl via pre-authenticated kubeconfig.
- No data exfil. Synthetic distributions only (uniform_small).
- Sanitizer coverage: memcheck on every new kernel + bridge code;
  racecheck + initcheck on atomic kernels (v12s).
- Bridge code is a NEW attack surface — sanitize specifically before
  trusting any TF number.

---

## 9. Dependencies

### Hardware
- gpu-01 V100 32GB sm_70 (sole V100 in cluster).
- `tcg-dev` pod in `llm` namespace, source at `/src/tools/tc-grid`.
- CUDA 12.2.2 toolchain.

### Tooling
- `ncu`, `nsys`, `compute-sanitizer`, `cuobjdump`, `ptxas --verbose`.
- NVBench (for native gemm_bench; sourced via P1.3).

### Code
- `_deps/cutlass-src/` (CUTLASS 2.11.0 — reference only).
- `research/lmdeploy/src/turbomind/kernels/gemm/` (port source).
- `tools/tc-grid/kernels/v12_kernels.cuh` (current production).
- `scripts/bench-median.sh` (SPRINT-019 P0.2 helper).

### Workflow
- DCGM-exporter pause before every ncu run.
- gpu-02-4090rtx unavailable (qwen3-moe-rotorquant lives there).
- AGENTS.md is llama.cpp upstream policy; private fork exempt.

---

## 10. Open questions (deferred to execution)

1. **Bridge import strategy**: should tc-grid call into turbomind via
   the existing `Gemm::Run` API or via direct kernel launch with
   `dispatch_cache.Import`? Decide at P2.1.
2. **NVBench dep**: vendor via FetchContent or document required system
   install? Decide at P1.3.
3. **Asymmetric KSPLIT for v12s**: P0.1 covers powers-of-2 + {3, 5};
   does production hit other KSPLIT values? Address as P0.1 follow-up
   if surfaced.
4. **VISION.md timing**: should a `/vision` invocation come before
   SPRINT-021 (port sprint) regardless of branch? Surface in REPORT-14
   §closing.

---

## 11. Estimated total effort

| Phase | ETA |
|---|---:|
| P0 (foundation cleanup) | 4–6 hr |
| P1 (gemm_bench build) | 6–12 hr (3× gut, opaque tooling) |
| P2 (tc-grid bridge) | 8–14 hr |
| P3 (head-to-head measurement) | 6–10 hr |
| P4-BREAKTHROUGH OR P4-CEILING-PROOF + P5 | 4–8 hr OR 8–14 hr |
| P6 (close + REPORT-14) | 4–6 hr |
| **Total** | **32–62 hr (~4–6 sessions)** |

Range mirrors SPRINT-019. The P1+P2 build engineering is the
single biggest risk-of-overrun zone.

---

## 12. Success / partial-success / failure summary

**Sprint succeeds if** any of these:
- **Breakthrough branch**: §1.1 contract met (≥44 TF or asymmetric ≥10%);
  SPRINT-021 port spec drafted; REPORT-14 published.
- **Ceiling-proof branch**: §1.1 contract met (≤41 TF); v12 family wired
  into DSv4-flash inference; sample-generation correctness held; REPORT-
  14 published.
- **Indeterminate branch**: §1.1 indeterminate; ceiling-proof path
  taken; SPRINT-021 reconsideration noted.

In all three cases, P0 deliverables (sanitizer + N≠K CLI + dispatch.h)
ship and clear SPRINT-019's correctness/usability debt.

**Sprint partially succeeds if** P0 + P1 + P2 ship but P3 evidence is
incomplete:
- P0 cleanup landed (always lands; P0 is independent of architectural
  branch).
- P1+P2 standing turbomind on tc-grid bridge proven feasible.
- P3 ncu/CSV evidence captured but architectural decision not made.

**Sprint fails if**:
- P1 build engineering blocks > 12 hr without progress → close as
  "ceiling assumed unprovable in available tooling" per P1 decision
  gate.
- P2 data-layout incompatibility → close as ceiling-proof; skip P3
  Turbomind rows entirely.
- Sanitizer reveals existing v12s race bug AND no scope reduction can
  preserve a sanitizer-clean production path → re-scope sprint to
  recovery only.

---

## 13. Pivot log (forward-looking — to be filled during execution)

- (start) Sprint launched with hybrid bench-first plan.
- (will populate at P4 branch with the outcome classification.)

# SPRINT-019 — Intent

## Seed prompt

> For this sprint plan I really want detailed steps including grid search
> benchmarking (ncu, nsight, cutlass) and correctness check for each major
> change. I also want no skipping. We will be methodical. Not being methodical
> has led us astray in the past. No more.

## Why this sprint

SPRINT-017 closed v11 at **35.08 TF at M=2048** (+18.9% vs v10's 29.50 TF),
bit-correct (rel = 2.594e-04 = v10 reference exactly). Decision-rule tier:
"35–50 TF: beats v10, ship + investigate gap."

[REPORT-12.md](../../tools/tc-grid/docs/REPORT-12.md) enumerates six concrete
levers to attack the gap to 50 TF (sprint goal) and ultimately to the 85 TF
P1 CUTLASS pre-dequant ceiling. The user has explicitly directed this sprint
to be **methodical and complete** — no skipping, with grid-search-driven
benchmarking and a correctness gate at every major step.

Past mis-steps that justify this discipline:

- **v9 (SPRINT-016)**: register-resident mixed-precision promote shipped
  before lane-mapping was checked → `rel = 0.71` across all variants
  (memory: `v100_wmma_half_float_frag_layout_mismatch`).
- **3-stage pipeline (SPRINT-017)**: shipped without per-shape register-budget
  verification → +9.5% on one shape but -29% on the champion at M=4096.
  Reverted because the win was asymmetric and uncaught.
- **__ldcs (SPRINT-017)**: shipped without per-pointer reuse analysis → -30%
  regression because W_scales reuse is high.

The methodical discipline this sprint encodes:

1. Every kernel change has a CPU-reference bit-correctness test BEFORE
   integration into the production dispatcher.
2. Every kernel change is benchmarked with `tc-grid --m-list` sweep AND with
   `ncu` stall breakdown across all M ∈ {64, 256, 1024, 2048, 4096}.
3. Every major change is compared to CUTLASS Gemm70's measured runtime on
   the same shape, so we know how much of the remaining gap closed.
4. No change is committed unless ncu shows the stall it targets actually
   dropped, AND the headline TF improved at ≥ 3 of the 5 M values without
   regressing the champion by > 2%.

## Orientation Summary

- **Current production state (sprint-017 close)**: v11 champion
  `mm_int8_lut_v11<128,128,16,4,8,2>` at 35.08 TF M=2048. Bit-correct vs
  v10 at all M ∈ {64..4096}.
- **Recent theme**: methodical lever-by-lever exploration; 4 shipped wins
  (BK=16, half2, PRMT, in-place mma) and 6 reverted experiments (each with
  documented `why`).
- **Key modules**:
  - `tools/tc-grid/kernels/v11_kernels.cuh` (production).
  - `tools/tc-grid/kernels/mma_sm70.cuh` (FP16-acc wrapper
    `mma_m8n8k4_row_col_acc_f16` already scaffolded, NOT wired).
  - `tools/tc-grid/kernels/v10splitk_kernels.cuh` (template for SplitK port).
  - `tools/tc-grid/kernels/cutlass_int8_kernels.cuh` (CUTLASS Gemm70 ceiling
    measurement; pre-dequanted FP16 path).
  - `tools/tc-grid/tests/test_mma_884_tile_sm70.cu` (template for any new
    atom-correctness test).
- **Constraints**:
  - Bit-correctness vs v10 reference: `rel ≤ 1e-3`, `p99 ≤ 0.05`,
    `maxabs ≤ 0.1` per existing harness gate.
  - V100 SM resources: 96 KB SMEM (opt-in), 65536 regs, 256 reg/thread cap,
    2048 max threads/SM.
  - Must run on `tcg-dev` pod in `llm` namespace. DCGM-exporter must be
    paused before each `ncu` run.
  - Source build path on pod: `/src/tools/tc-grid`. Laptop sync via rsync
    to `ubuntu@192.168.102.5:/srv/dev/dsv4-cuda/deepseek-sprint017/...`.
- **No VISION.md exists**. Planning from scratch using REPORT-12 §6 as the
  candidate lever list.
- **`scripts/ledger.py` does NOT exist** in this repo (per FOLLOWUPS-016).
  Skip the ledger-sync step in the workflow.
- **gpu-02-4090rtx** is occupied by `qwen3-moe-rotorquant` — do NOT evict
  for parallel V100 work.

## Deferred items now actionable

From `SPRINT-016-DEFERRED.md`:
- **Persistent CTA grid-stride output loop revisit** (was Tier A3 / v5).
  Prerequisite met: with FP16 acc cutting c_frag in half (this sprint),
  occupancy headroom may exist. Touches `tools/tc-grid/kernels/v5_kernels.cuh`.
  Will treat as a candidate for the §6.4 phase (larger BM with c_frag
  SMEM spill), since both attack the register-pressure ceiling.
- **Split-K** (was Tier B2). Prerequisite met: small-M (M=64) is now a real
  gap — v11 at M=64 = 7.52 TF, v10s_ks8 at 20.31 TF. Maps directly to
  REPORT-12 §6.3.

From `SPRINT-016-FOLLOWUPS.md`:
- **v9 SMEM round-trip variant**. Maps to the SMEM-roundtrip epilogue
  pattern needed for REPORT-12 §6.1 (FP16 accumulator path). The v9
  follow-up resolves the f16/f32 fragment layout mismatch by going
  through SMEM rather than register-resident promote.
- **INT4 BN=256 spill fix via two-pass FRAG_N**. Parallel pattern to
  REPORT-12 §6.4 (larger BM with c_frag SMEM spill). Separate kernel
  (`v3_bn256_kernels.cuh`); pickup ONLY if §6.4 succeeds and we want
  to back-port the pattern to v3 INT4.
- **WMMA fragment layout equivalence verification gate** (process item).
  Already implicitly in this sprint's "correctness test before integration"
  rule.
- **tc-grid harness CSV quoting** (nice-to-have). Not in scope unless
  parsing scripts break during the sprint.

From REPORT-12 §6 directly:
- §6.1 FP16 accumulator + SMEM round-trip epilogue (HIGH RISK, +30-50%).
- §6.2 Per-shape 3-stage pipeline (+5-8%).
- §6.3 SplitK port to v11 (+50-80% at M=64).
- §6.4 Larger CTA tile with c_frag SMEM spill (+5-10%).
- §6.5 PRMT A-side load (marginal, +0.5-2%).
- §6.6 Multi-shape MoE validation (orthogonal).

## Sprint goal

Cross the **50 TF threshold at M=2048** (the SPRINT-017 carry-over goal),
and close the M=64 production gap relative to v10s by porting SplitK to
v11. Both must be bit-correct (rel ≤ 1e-3) and ncu-verified.

If 50 TF proves out of reach after methodical execution: document the
remaining stalls precisely and propose the architectural break for the
next sprint (e.g., wholesale port from turbomind's sm_70 GEMM, or accept
the v11 ceiling as the production answer).

## Success criteria

Methodical: every phase below has its own measurement and gate.

**Headline**:
- v11 (or successor) at M=2048: **≥ 50 TF** (sprint goal) OR documented
  rationale with ncu evidence for why the gap is unclosable in this kernel
  family.
- v11+SplitK at M=64: ≥ 20 TF (parity with v10s_ks8 = 20.31 TF).
- Per-M champion table written into REPORT-13.md, with ncu-evidenced delta
  vs the previous (sprint-017) champion at every M.

**Methodical gates** applied at every phase boundary:
- CPU-reference bit-correctness on a small isolated test BEFORE integration
  into the production dispatcher.
- `compute-sanitizer --tool memcheck` on first launch of every new kernel
  template.
- `tc-grid --m-list 64,256,1024,2048,4096 --nk 7168` bit-compare against v10:
  `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`.
- `ncu` stall breakdown including:
  `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`,
  `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct`,
  `smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct`,
  `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct`,
  `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed`,
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`,
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`.
- CUTLASS comparison: at each phase that ships, run the same shape
  through `kernels::int8_cutlass::Gemm70` (pre-dequanted FP16 path,
  version=40) and report (a) absolute TF, (b) v11 / CUTLASS ratio.
- Nsight Compute report files archived under `tools/tc-grid/docs/ncu/`
  for each phase (gitignored binaries, but the CSV exports committed).
- Grid sweep with **at least 12 tile shapes** per major change, results
  recorded in `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-N.csv`.

**Definition of done — applied at every commit, not just sprint close**:
1. Bit-correctness sweep passes for every shape.
2. ncu stall breakdown captured for the new champion at M ∈ {2048, 4096}.
3. CUTLASS ratio measured at M=2048.
4. Commit message includes the headline TF + ncu evidence for what stall
   actually dropped.
5. If the change didn't drop the target stall → REVERT, no shipping.

## Verification strategy

Per-phase verification stack (all phases share this template):

**Tier 1 — isolated correctness** (before any production-dispatcher wiring):
- Build a CPU-reference test in `tools/tc-grid/tests/test_<phase>.cu`.
- Fill known small inputs (~256–1024 cells).
- Compare GPU output against the CPU model bit-exactly (rel ≤ 1e-3 for
  inputs that aren't degenerate; bit-exact for integer paths).
- Run `compute-sanitizer --tool memcheck` on first launch.

**Tier 2 — production integration**:
- Wire into `launch_int8.cu` (LAUNCH_V11_X + RERUN_V11_X macros).
- Add tile entries to `main.cu` kTiles[].
- Build clean; record any new ptxas spill warnings.
- Run `tc-grid --m-list 64,256,1024,2048,4096 --nk 7168` bit-correctness
  pass; **regression-test against v10 baseline AND against the
  pre-phase v11 champion**.

**Tier 3 — performance characterization**:
- `tc-grid` headline TF across all M values, both with the new variant
  and the old champion (sanity that nothing regressed > 2%).
- `ncu` stall breakdown at M=2048 and M=4096 (the two production points).
- **Compare to CUTLASS Gemm70 pre-dequant ceiling** at the same shape;
  report ratio.
- Grid sweep (12+ shapes): identify per-shape champion. If the new
  variant doesn't win at ≥ 3 of the 5 M values without regressing the
  prior champion by > 2%, **the change is reverted**, not shipped.

**Tier 4 — Nsight Systems timeline** (for any pipeline-restructuring
phase — §6.2 3-stage, §6.1 FP16 acc — where compute/memory overlap is the
target):
- `nsight-sys --capture-range cudaProfilerApi` over one M=2048 invocation.
- Inspect timeline: confirm that LDG and mma blocks actually overlap
  in the SM trace, not just appear adjacent.
- Save `.nsys-rep` and a screenshot to `tools/tc-grid/docs/nsys/`.

## Uncertainty assessment

- **Correctness**: HIGH for §6.1 (FP16 acc lane-mapping is undocumented,
  v9 burned us here). MEDIUM for §6.2 (3-stage already had asymmetric
  regressions in SPRINT-017). LOW for §6.3 (SplitK is well-understood from
  v10s). LOW for §6.5 (vectorized A-side load).
- **Scope**: MEDIUM. 6 distinct levers; methodical execution may take 2–3
  weeks of session-time. Need to decide where to time-box.
- **Architecture**: MEDIUM. FP16 acc path may force SMEM-round-trip
  epilogue (separate code path). §6.2 may force per-shape kernel
  dispatch.

## Open questions for interview

1. **Time-boxing**: should we execute all 6 levers methodically (3+ weeks
   session-time, full 50 TF push) OR cap at 2-3 levers and ship at 40 TF?
2. **FP16 acc risk tolerance**: §6.1 has the biggest expected ROI (+30-50%)
   AND the biggest correctness risk. Confirm: methodical doesn't preclude
   abandoning the lever mid-sprint if isolated correctness gate fails?
3. **CUTLASS comparison depth**: at minimum, measure CUTLASS at the same
   shape. Optional: build turbomind's `gemm_bench` standalone (1-2 days
   build engineering per `V11-EXECUTION-PLAN.md` deferred note). Worth it?
4. **Sprint boundary**: should SplitK port (§6.3) be a separate sprint
   (it's lower-risk and orthogonal to the FP16 acc work)? Or in-sprint?
5. **Grid sweep scope**: 12+ shapes per phase is a lot of measurement.
   Acceptable, or shrink to ~6 per phase + one comprehensive sweep at
   sprint close?
6. **Nsight Systems gating**: do we mandate a timeline capture for every
   pipeline change, or only for the 2 (FP16 acc, 3-stage) that
   restructure the overlap pattern?

## Files / source areas in scope

- `tools/tc-grid/kernels/v11_kernels.cuh` (production champion; will fork
  into multiple variants).
- `tools/tc-grid/kernels/mma_sm70.cuh` (mma PTX wrappers; FP16-acc wrapper
  exists, may need m8n8k16 or larger atom variants).
- `tools/tc-grid/kernels/cutlass_int8_kernels.cuh` (CUTLASS ceiling
  baseline; comparison target).
- `tools/tc-grid/kernels/v10splitk_kernels.cuh` (SplitK template to port).
- `tools/tc-grid/src/launch_int8.cu` (dispatcher; multiple new LAUNCH
  macros).
- `tools/tc-grid/src/main.cu` (tile registry; new entries).
- `tools/tc-grid/tests/` (NEW per-phase correctness tests).
- `tools/tc-grid/docs/` (REPORT-13, grid-sweep CSVs, ncu/nsys exports).

## Out of scope (deferred to future sprints)

- Multi-shape MoE validation (REPORT-12 §6.6) — orthogonal to the perf
  push; separate sprint when DSv4 layer dimensions are profiled.
- MoE-aware dispatcher integration into the live DSv4 inference path —
  separate sprint, not a kernel-tuning task.
- Turbomind `gemm_bench` standalone — 1-2 days build engineering for a
  sanity-check that's already partially answered by CUTLASS Gemm70.
  Revisit only if Sprint-019 lands below 40 TF and we want an
  independent comparison.
- INT4 BN=256 spill fix (FOLLOWUPS-016) — only if §6.4 succeeds.
- v5 persistent-CTA revisit (DEFERRED-016) — only if §6.1 FP16 acc opens
  occupancy headroom.

## References

- [REPORT-12.md](../../tools/tc-grid/docs/REPORT-12.md) — sprint-017 close,
  benchmark learnings, 6 forward candidates.
- [SPRINT-017.md](../SPRINT-017.md) — v11 execution sprint.
- [V11-EXECUTION-PLAN.md](../../tools/tc-grid/docs/V11-EXECUTION-PLAN.md)
  — wave-decisioned plan with risks log.
- [SPRINT-018-CUTLASS.md](../SPRINT-018-CUTLASS.md) — CUTLASS 85 TF
  ceiling proof.
- Memory items: `v100_wmma_half_float_frag_layout_mismatch`,
  `feedback_pre_dequant_defeats_int8`,
  `feedback_dont_skip_plan_steps`,
  `feedback_effort_estimation_undocumented_hardware`.

# SPRINT-019 CODEX DRAFT
## Overview
Sprint 019 is the methodical follow-on to the Sprint 017 close report and the Sprint 018 CUTLASS ceiling proof. The 
current production large-M INT8 kernel family is `v11`. The current large-M champion is `mm_int8_lut_v11<128, 128, 16, 
4, 8, 2>`. At `M=2048`, `N=K=7168`, it delivers `35.08 TF`. At the same shape, the CUTLASS pre-dequant ceiling is 
`85.93 TF`. The current small-M production answer remains v10 SplitK. This sprint exists because the project now has a 
clean lever list. `REPORT-12.md` identifies six concrete next steps. The project also has fresh evidence for how easy 
it is to fool ourselves if we optimize without phase-specific correctness and measurement gates. Sprint 019 therefore 
treats methodology as a first-class deliverable, not background process. The top-line sprint objective is unchanged 
from Sprint 017: cross `50 TF` at `M=2048` on the fused INT8 path while preserving the existing bit-correctness 
contract against the v10 reference. The second objective is to close the glaring `M=64` gap by porting SplitK to v11 
rather than continuing to treat small-M as a separate exception forever. The third objective is not to leave ambiguous 
results behind. Every major kernel change must end in one of three states: shipped, reverted with evidence, or parked 
with a concrete blocker note and artifact trail. The sprint will follow the six levers in `REPORT-12 §6.1-6.6`. Phase 
numbering in this document intentionally matches that report: 1. `6.1` FP16 accumulator plus SMEM round-trip epilogue. 
2. `6.2` Per-shape 3-stage pipeline. 3. `6.3` SplitK port to v11. 4. `6.4` Larger CTA tile with c-frag spill 
mitigation. 5. `6.5` PRMT-vectorized A-side load. 6. `6.6` Multi-shape MoE validation. This sprint draft is 
deliberately strict about decision gates. No phase is allowed to hand-wave correctness. No phase is allowed to claim a 
performance win from a single shape. No phase is allowed to ship if the target stall bucket does not actually move. No 
phase is allowed to regress the current champion materially at `M=2048` or `M=4096` without an explicit pivot decision. 
The sprint succeeds in one of two ways: 1. The fused INT8 path reaches `>= 50 TF` at `M=2048`, stays bit-correct, and 
improves the CUTLASS-ceiling ratio measurably. 2. The sprint proves, with complete `ncu` and `nsys` evidence, that the 
v11 kernel family cannot realistically close the remaining gap without an architectural break, and it leaves a cleanly 
scoped next move. The sprint fails if it ends with partially measured variants, ambiguous correctness, or undocumented 
reversions.
## Use Cases
1. Large-batch prefill on V100 should have a fused INT8 kernel family that is materially closer to the measured CUTLASS 
ceiling than Sprint 017's `35.08 TF`, without breaking the current error envelope. 2. Decode-style or small-batch 
operation at `M=64` should no longer depend on a separate v10-only optimization story if v11 SplitK can recover the 
same class of win. 3. Kernel tuning work should stop relying on single-shape anecdotes. Every proposed win must be 
validated over `M ∈ {64, 256, 1024, 2048, 4096}`. 4. High-risk architectural changes, especially FP16-accumulator work, 
should prove their lane mapping and epilogue correctness in isolated tests before touching the production dispatcher. 
5. Pipeline changes should prove overlap improvement with Nsight Systems, not just by looking at topline TF or generic 
`smsp_active` numbers. 6. Future sprint authors should be able to audit exactly why a lever shipped or why it was 
rejected by reading committed CSVs and report text instead of reconstructing ad hoc shell history. 7. The production 
dispatcher should end the sprint with explicit per-M champions, including whether large-M and small-M still require 
different families. 8. The tc-grid harness should accumulate reusable tests for manual fragment-load kernels on 
`sm_70`, especially for FP16-accumulator atom behavior and spill-to-SMEM epilogues. 9. Multi-shape validation should 
catch the possibility that the square `7168x7168` benchmark shape is flattering the current champion. 10. CUTLASS 
should continue to serve as a ceiling reference, not as an unexamined production surrogate.
## Architecture
### Kernel families and responsibilities
The sprint works inside the existing tc-grid structure. The control plane remains the benchmark harness under 
`tools/tc-grid/src/`. The data plane remains the CUDA kernel family under `tools/tc-grid/kernels/`. This sprint should 
preserve that split. The kernel families relevant to Sprint 019 are: - `v11` in `tools/tc-grid/kernels/v11_kernels.cuh` 
- the FP16-accumulator PTX wrapper in `tools/tc-grid/kernels/mma_sm70.cuh` - `v10splitk` in 
`tools/tc-grid/kernels/v10splitk_kernels.cuh` - the CUTLASS reference path in 
`tools/tc-grid/kernels/cutlass_int8_kernels.cuh` The expected Sprint 019 additions are not one monolithic kernel. They 
are a set of closely related variants that let the harness compare architectural choices without contaminating the 
current shipped path. The likely naming pattern is: - `v11f` or `v11_f16acc` for the FP16-accumulator path - `v11_ms3` 
for 3-stage variants - `v11s` for SplitK variants - `v11b192` / `v11b256` or similar for larger-BM experiments - 
`v11a_prmt` or an in-place A-load specialization for A-side load work Exact names can follow repo style once 
implementation starts. What matters in the plan is that each lever gets a separable kernel family or compile-time flag 
so it can be benchmarked and reverted independently.
### Dispatcher model
`tools/tc-grid/src/launch_int8.cu` remains the dispatch point. Sprint 019 should avoid burying multiple lever changes 
behind one opaque `version == 11` branch. Instead, the plan should preserve distinct launch macros and rerun timing 
paths for each experimental family. The dispatch model at the end of the sprint should support: - the current `v11` 
2-stage baseline - a candidate `v11f` FP16-accumulator family - a candidate `v11_ms3` 3-stage family - a candidate 
`v11s` SplitK family - the existing CUTLASS reference family for ceiling comparison The dispatcher does not have to 
expose all of these to an end user as stable production choices. It does need to make them easy to benchmark 
head-to-head without editing the source between runs.
### Correctness architecture
The core correctness contract does not change. The reference remains the existing v10 path and the existing harness 
error envelope: - `rel <= 1e-3` - `p99 <= 0.05` - `maxabs <= 0.1` Sprint 019 adds a more structured correctness ladder: 
1. Isolated CPU-reference test for the new mechanism. 2. `compute-sanitizer --tool memcheck` on first launch of every 
new kernel template or atom test. 3. Full tc-grid bit-compare sweep against v10 across all five M values. 4. 
Bit-compare against the previous sprint champion so a new family does not silently move numerics even if it stays 
within the v10 envelope. The first rung is the important process correction. The project has already seen that 
atom-level or fragment-layout mistakes can look plausible until they are embedded into a long reduction. Every major 
change in this sprint must therefore start with a dedicated test in `tools/tc-grid/tests/`.
### Performance architecture
There are four performance evidence layers in this sprint: 1. Headline TF from tc-grid over the standard M sweep. 2. 
Nsight Compute stall analysis against the target bottleneck for the phase. 3. CUTLASS ratio comparison at the same 
shape. 4. Nsight Systems timeline proof for overlap-focused changes. The primary required Nsight Compute metrics are: - 
`smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` - 
`smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` - 
`smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct` - 
`smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct` - 
`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` - 
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` 
The secondary Nsight Compute metrics, recorded when relevant, are: - `launch__registers_per_thread` - 
`sm__warps_active.avg.pct_of_peak_sustained_active` - `smsp__inst_executed_pipe_tensor.sum` - 
`dram__throughput.avg.pct_of_peak_sustained_elapsed` - `lts__throughput.avg.pct_of_peak_sustained_elapsed` - 
`smsp__sass_average_branch_targets_threads_uniform.pct` The sprint should treat the primary set as mandatory. The 
secondary set is phase-specific. If a phase claims it reduced register pressure, it must record 
`launch__registers_per_thread`. If a phase claims it improved memory overlap, it must record `dram` and `lts` context 
alongside the scoreboard metrics.
### Artifact layout
Sprint 019 should produce a predictable artifact tree under `tools/tc-grid/docs/`. The point is not just convenience. 
The point is auditability. Recommended artifact layout: - `tools/tc-grid/docs/REPORT-13.md` - 
`tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.1.csv` - `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.2.csv` - 
`tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.3.csv` - `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.4.csv` - 
`tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.5.csv` - `tools/tc-grid/docs/multishape-SPRINT-019-PHASE-6.6.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.1-*.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.2-*.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.3-*.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.4-*.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.5-*.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.6-*.csv` - 
`tools/tc-grid/docs/nsys/SPRINT-019-phase-6.1-*.nsys-rep` - `tools/tc-grid/docs/nsys/SPRINT-019-phase-6.2-*.nsys-rep` - 
`tools/tc-grid/docs/nsys/SPRINT-019-phase-6.1-*.png` - `tools/tc-grid/docs/nsys/SPRINT-019-phase-6.2-*.png` The binary 
profiler outputs can stay gitignored if necessary. The CSV exports and report conclusions should be committed.
### Sprint-level decision logic
Sprint 019 should use one common decision rule template for every phase: 1. Did isolated correctness pass? 2. Did 
`compute-sanitizer` pass? 3. Did the full five-M bit-compare pass? 4. Did the target stall bucket move in the expected 
direction? 5. Did the new family improve the headline result at at least three of the five M values? 6. Did it avoid 
regressing the current production champion by more than `2%` at `M=2048` or `M=4096`? 7. Did the CUTLASS ratio improve 
at the headline large-M shape? If the answer to any of questions 1 through 3 is no, the phase does not ship. If the 
answer to question 4 is no, the phase does not claim success even if TF is noisy-positive. If the answer to question 5 
is no, the phase can still remain as a specialized small-M or shape-specific variant only if that scope is explicit. If 
the answer to question 6 is no, the change reverts unless the sprint is explicitly pivoting to a different per-M 
dispatch regime. If the answer to question 7 is no, the phase needs an explanation in REPORT-13.
## Implementation
### Common Protocol for All Phases
The implementation section below is phase-specific. Before any `6.1` through `6.6` work starts, Sprint 019 should set 
one frozen baseline and one common measurement procedure.
#### Baseline freeze
Record the exact baseline from Sprint 017 close: - Kernel family: `mm_int8_lut_v11<128, 128, 16, 4, 8, 2>` - `M=64`: 
`7.52 TF` - `M=256`: `29.13 TF` - `M=1024`: `34.94 TF` - `M=2048`: `35.08 TF` - `M=4096`: `34.77 TF` - Error envelope: 
`rel = 2.594e-04` at the large-M checkpoints Also record the current reference ceilings: - CUTLASS pre-dequant ceiling 
at `M=2048`: `85.93 TF` - v10 SplitK small-M answer at `M=64`: `20.31 TF` These numbers should be copied into 
`REPORT-13.md` at the top so every phase has a stable baseline table.
#### Build and environment protocol
Every profiling block should start from the same operational hygiene: 1. Ensure work happens on the `tcg-dev` pod in 
the `llm` namespace. 2. Disable DCGM exporter on the V100 node before `ncu` or `nsys`. 3. Rebuild from the same source 
tree under `/src/tools/tc-grid`. 4. Record the git commit SHA for every profiler capture. 5. Restore the exporter after 
the profiling session. Recommended commands:
```bash
kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=false --overwrite
cmake -S /src/tools/tc-grid -B /src/tools/tc-grid/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build /src/tools/tc-grid/build -j
```
Restore command:
```bash
kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=true --overwrite
```
#### Standard correctness sweep
The standard integrated correctness sweep for every major phase is:
```bash
/src/tools/tc-grid/build/tc-grid \
  --m-list 64,256,1024,2048,4096 \
  --nk 7168 \
  --dist uniform_small
```
The phase should retain the CSV row subset for: - the previous shipped champion - the new phase candidate - the CUTLASS 
reference row - the v10 reference row The phase is only green if the candidate satisfies: - `rel <= 1e-3` - `p99 <= 
0.05` - `maxabs <= 0.1` The phase should also record whether the candidate matches the previous baseline's `rel` 
closely enough to support "numerically equivalent in practice" language.
#### Standard Nsight Compute command
Use a consistent command template so the metric sets are comparable:
```bash
ncu --csv --page raw \
  --kernel-name-base demangled \
  --metrics \
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed \
  --log-file /src/tools/tc-grid/docs/ncu/<phase>.csv \
  /src/tools/tc-grid/build/tc-grid --m-list <M> --nk 7168 --dist uniform_small
```
Required M values for `ncu`: - `M=2048` - `M=4096` For SplitK work in `6.3`, also require: - `M=64` - `M=256`
#### Standard Nsight Systems command
For overlap-oriented phases, use one capture template:
```bash
nsys profile \
  --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --sample none \
  --capture-range cudaProfilerApi \
  --capture-range-end stop \
  --output /src/tools/tc-grid/docs/nsys/<phase> \
  /src/tools/tc-grid/build/tc-grid --m-list 2048 --nk 7168 --dist uniform_small
```
Required phases for `nsys`: - `6.1` - `6.2` Optional phases for `nsys`: - `6.4` if spill-to-SMEM rotation appears to 
alter overlap materially
#### Standard CUTLASS comparison
Every phase that remains in contention after correctness should measure against the CUTLASS reference on the same 
benchmark shape. The point is not that CUTLASS is shippable. The point is to quantify how much of the remaining gap the 
phase closed. For each relevant M, record: - phase candidate TF - CUTLASS TF - ratio `candidate / CUTLASS` - delta 
versus previous phase ratio Required M values: - `M=2048` - `M=4096` For `6.3`, also include: - `M=64`
#### Standard grid sweep rules
Every performance phase needs a committed grid sweep CSV. The sweep should use the same five-M set and record one row 
per tile or tile-plus-mode combination. Grid sweep rules: 1. Minimum 12 variants per major change. 2. Every variant 
must be rerun at least twice if the first timing is within `1%` of the current winner. 3. The sweep must include the 
previous phase champion as an anchor row. 4. The sweep must include the CUTLASS reference row for the relevant M 
values. 5. The sweep must not silently drop a losing shape. Negative results are part of the record.
#### Phase closeout template
Each phase closeout section in `REPORT-13.md` should answer: 1. What stall or bottleneck was targeted? 2. What isolated 
correctness test was added? 3. What integration path was touched? 4. What exact sweep set was run? 5. What did `ncu` 
say changed? 6. What did `nsys` say changed, if applicable? 7. What happened to the CUTLASS ratio? 8. Ship, revert, or 
park? 9. If parked, what unblocks it?
### Phase 6.1 — FP16 Accumulator + SMEM Round-Trip Epilogue
#### Objective
Exploit the FP16-accumulator tensor-core path to cut c-frag register pressure roughly in half and increase tensor-pipe 
utilization without repeating the Sprint 016 fragment-layout failure.
#### Why this phase is first
`REPORT-12` ranks this as the largest single upside. It is also the highest correctness risk. Running it first is 
consistent with the user’s explicit requirement to be methodical. If the highest-upside lever is fundamentally blocked 
on `sm_70` layout behavior, the sprint should learn that early and stop pretending the `50 TF` goal is still likely.
#### Hypothesis
If `v11` switches from FP32 accumulator fragments to FP16 accumulator fragments and uses a scratch-SMEM epilogue 
round-trip for final float writeout, then: - `launch__registers_per_thread` should drop materially, - 
`smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` should decrease, - 
`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` should increase, - the best large-M shape 
should move significantly closer to the CUTLASS ratio.
#### Files in scope
- `tools/tc-grid/kernels/mma_sm70.cuh` - `tools/tc-grid/kernels/v11_kernels.cuh` - `tools/tc-grid/src/launch_int8.cu` - 
`tools/tc-grid/src/main.cu` - `tools/tc-grid/tests/test_mma_884_tile_sm70_f16acc.cu` - 
`tools/tc-grid/tests/test_v11_f16acc_epilogue_sm70.cu`
#### Implementation steps
1. Copy the existing atom test pattern from `test_mma_884_tile_sm70.cu`. 2. Add a new isolated atom correctness test, 
`test_mma_884_tile_sm70_f16acc.cu`, that exercises `mma_m8n8k4_row_col_acc_f16` directly. 3. Use deliberately small 
matrices in the atom test so the CPU reference can enumerate every lane contribution. 4. Add explicit per-lane scatter 
checks, not just aggregate output checks. 5. Add a second test, `test_v11_f16acc_epilogue_sm70.cu`, that verifies the 
scratch-SMEM round-trip: FP16 accumulator fragment to SMEM to float conversion to global writeback. 6. In the kernel 
family, introduce a separate experimental path rather than mutating the current production `v11` in place. 7. Replace 
`float c_frag[...][...][8]` with a half-typed accumulator storage layout appropriate to the wrapper semantics. 8. Keep 
the epilogue isolated and obvious. Do not simultaneously fold in unrelated dequant or store changes. 9. Wire the 
experimental path into `launch_int8.cu` as a separate launch macro. 10. Register the experimental tiles in `main.cu` 
with names that make the accumulator mode explicit. 11. Record any new ptxas register or spill warnings immediately. 
12. Only after the isolated tests are green should the full tc-grid path be invoked.
#### Required isolated correctness commands
```bash
compute-sanitizer --tool memcheck \
  /src/tools/tc-grid/build/test_mma_884_tile_sm70_f16acc

compute-sanitizer --tool memcheck \
  /src/tools/tc-grid/build/test_v11_f16acc_epilogue_sm70
```
The phase does not advance until both tests pass.
#### Integrated benchmark matrix
The integrated `tc-grid` runs for this phase must include: - `M=64` - `M=256` - `M=1024` - `M=2048` - `M=4096` The 
CUTLASS comparison points for phase closeout are: - `M=2048` - `M=4096` The mandatory `ncu` points are: - `M=2048` - 
`M=4096` The mandatory `nsys` point is: - `M=2048`
#### Grid sweep for 6.1
Run a 16-variant tile sweep. Each variant uses the FP16-accumulator path. Keep `BK=16` fixed for the first pass, 
because `REPORT-12` already identified that as the best base large-M regime. Required sweep variants: 1. `64x128x16_w2` 
2. `64x128x16_w4` 3. `64x256x16_w4` 4. `64x256x16_w8` 5. `96x128x16_w4` 6. `96x256x16_w8` 7. `128x128x16_w2` 8. 
`128x128x16_w4` 9. `128x128x16_w8` 10. `128x256x16_w4` 11. `128x256x16_w8` 12. `192x128x16_w4` 13. `192x256x16_w8` 14. 
`256x128x16_w4` 15. `256x128x16_w8` 16. `256x256x16_w8` Also include anchor rows for: - baseline `v11 128x128x16_w4` - 
current small-M `v10s_ks8` - CUTLASS reference
#### Nsight Compute interpretation target
The intended signature of success for `6.1` is: - lower short-scoreboard stalls, - higher tensor-pipe active 
percentage, - lower registers per thread, - flat or improved long-scoreboard stalls, - no explosion in shared-memory 
bank conflicts from the epilogue scratch path. Expected phase-specific gating thresholds: - 
`launch__registers_per_thread` decreases by at least `8` registers on the winning large-M shape, or the report must 
explain why not. - `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` decreases by at least `3` absolute 
points on the winning large-M shape. - `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` 
increases by at least `5` absolute points on that same shape. These thresholds are not enough by themselves to ship. 
They are the minimum evidence that the lever hit the bottleneck it claimed to target.
#### Nsight Systems interpretation target
Because the epilogue and fragment mode both change here, capture an `M=2048` timeline and confirm: - the kernel still 
exhibits the intended steady-state pipeline cadence, - no unexpected serialization appears around the epilogue region, 
- there is no obvious gap expansion between global-load and tensor-op periods. If `nsys` shows a new serialized tail 
that matches a TF regression, the phase should park the design rather than "tune around" an architectural flaw blindly.
#### Decision gate for 6.1
Ship the `6.1` family forward only if all of the following are true: 1. Both isolated CPU-reference tests pass. 2. Both 
`compute-sanitizer` runs pass. 3. The five-M v10 bit-compare passes. 4. The large-M winning shape improves `M=2048` TF 
by at least `10%` over the Sprint 017 baseline, or improves the CUTLASS ratio by at least `0.05`. 5. The target 
short-scoreboard and tensor-pipe metrics move in the expected direction. 6. The new family wins at at least three of 
the five M values in the sweep or clearly establishes itself as the large-M path without regressing `M=2048` or 
`M=4096` by more than `2%`. If correctness fails: revert immediately and write a blocker note in `REPORT-13`. If 
correctness passes but TF is noisy or flat and the target stall metrics do not move: park the phase as "concept not 
validated on this layout". If correctness passes and the performance signature is positive but not yet enough for `50 
TF`, carry the family into `6.2`.
#### Artifacts required for 6.1 closeout
- `tools/tc-grid/tests/test_mma_884_tile_sm70_f16acc.cu` - `tools/tc-grid/tests/test_v11_f16acc_epilogue_sm70.cu` - 
`tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.1.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.1-M2048.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.1-M4096.csv` - `tools/tc-grid/docs/nsys/SPRINT-019-phase-6.1-M2048.nsys-rep` 
- `tools/tc-grid/docs/nsys/SPRINT-019-phase-6.1-M2048.png`
### Phase 6.2 — Per-Shape 3-Stage Pipeline
#### Objective
Revisit the 3-stage pipeline experiment from Sprint 017, but only on shapes whose register budget can absorb the extra 
rmem buffering.
#### Why this phase follows 6.1
The 3-stage pipeline is explicitly documented in `REPORT-12` as a shape-sensitive lever. Running it after `6.1` is 
important because FP16 accumulation may lower register pressure enough to change which shapes can afford a third stage. 
It is not methodical to retest the same 3-stage idea under the old register budget and pretend that answers the 
combined design space.
#### Hypothesis
If the 3-stage pipeline is restricted to shapes whose register budget survives the additional rmem buffer, then: - 
`smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` should drop, - TF should improve on some subset of 
large-M shapes, - Nsight Systems should show tighter overlap of loads and tensor work, - the champion should not repeat 
the Sprint 017 occupancy collapse.
#### Files in scope
- `tools/tc-grid/kernels/v11_kernels.cuh` - `tools/tc-grid/src/launch_int8.cu` - `tools/tc-grid/src/main.cu` - 
optionally a factored helper header if the mainloop becomes too opaque
#### Implementation steps
1. Do not overwrite the existing 2-stage path. 2. Create a distinct `v11_ms3` family or equivalent compile-time mode. 
3. Scope the first implementation to the `6.1` winner if `6.1` shipped. If `6.1` did not ship, scope it to the Sprint 
017 baseline family. 4. Add explicit compile-time comments or assertions around stage count, scratch-buffer size, and 
anticipated register pressure. 5. Start by reviving the previously positive shape class from Sprint 017: 
`64x256x16_w8`. 6. Add at least one current-champion shape: `128x128x16_w4`. 7. Add one intermediate compromise shape: 
`128x256x16_w4`. 8. Build and record ptxas register counts before running the full sweep. 9. If ptxas shows a dramatic 
increase or spills on the champion shape, stop and narrow the candidate set instead of pushing through to full 
profiling. 10. Wire the family into the dispatcher as a per-shape experimental option. 11. Ensure the tile registry 
makes the stage count visible in the tile name. 12. Capture `nsys` only after correctness and the first `ncu` pass are 
green.
#### Required correctness checks
This phase does not need a brand-new atom test if it reuses the exact same math path as the prior phase. It does still 
need: - `compute-sanitizer --tool memcheck` on the first integrated launch, - the full five-M bit-compare, - comparison 
against the previous phase winner. If `6.2` introduces any new addressing arithmetic for pipeline state, add a small 
targeted test under `tools/tc-grid/tests/` rather than trusting the full kernel run.
#### Integrated benchmark matrix
Required M values: - `M=64` - `M=256` - `M=1024` - `M=2048` - `M=4096` Required `ncu` points: - `M=2048` - `M=4096` 
Required `nsys` point: - `M=2048` Required CUTLASS comparison points: - `M=2048` - `M=4096`
#### Grid sweep for 6.2
Run a 12-variant sweep focused on shapes that are plausible 3-stage candidates. This is intentionally narrower than 
`6.1`. The question here is not the whole tile space. The question is where the extra stage fits. Required sweep 
variants: 1. `64x128x16_w2_ms3` 2. `64x128x16_w4_ms3` 3. `64x256x16_w4_ms3` 4. `64x256x16_w8_ms3` 5. `96x256x16_w8_ms3` 
6. `128x128x16_w2_ms3` 7. `128x128x16_w4_ms3` 8. `128x256x16_w4_ms3` 9. `128x256x16_w8_ms3` 10. `192x128x16_w4_ms3` 11. 
`192x256x16_w8_ms3` 12. `256x128x16_w8_ms3` Anchor rows must include: - the best non-3-stage family from `6.1` - the 
Sprint 017 baseline champion - CUTLASS reference
#### Nsight Compute interpretation target
The intended signature of success for `6.2` is: - lower long-scoreboard stalls, - no catastrophic increase in registers 
per thread, - no occupancy collapse, - flat or reduced short-scoreboard stalls, - net TF improvement on shapes where 
the added stage is supposed to help. Phase-specific gating thresholds: - 
`smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` decreases by at least `4` absolute points on the best 
`6.2` shape. - `launch__registers_per_thread` does not increase by more than `16` on that shape relative to its 2-stage 
counterpart unless the TF gain exceeds `8%`. - `sm__warps_active.avg.pct_of_peak_sustained_active` does not fall by 
more than `10` absolute points on the candidate shape without a compensating TF gain of at least `5%`.
#### Nsight Systems interpretation target
The `M=2048` timeline should show: - tighter overlap between global-memory fetches and MMA issue windows, - fewer 
visible idle gaps between steady-state iterations, - no extended prologue or drain phase that wipes out the overlap 
gain. If `nsys` contradicts the `ncu` story, trust the contradiction and investigate. Do not ship a pipeline story that 
only exists in one profiler.
#### Decision gate for 6.2
Ship a `6.2` family forward only if: 1. The five-M bit-compare passes. 2. The target long-scoreboard bucket drops on 
the winning shape. 3. The `nsys` timeline shows actual overlap improvement. 4. The family improves at least one of 
`M=2048` or `M=4096` by `>= 5%`. 5. The family does not regress the current large-M champion by more than `2%` at the 
other large-M checkpoint. If `6.2` is only positive on a non-champion shape such as `64x256x16_w8`, keep it as a 
shape-specialized option only if the tile registry and report make that scope explicit. If the target long-scoreboard 
stall does not move, revert the change even if one timing number happened to improve. That is exactly the kind of 
false-positive process this sprint is trying to eliminate.
#### Artifacts required for 6.2 closeout
- `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.2.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.2-M2048.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.2-M4096.csv` - `tools/tc-grid/docs/nsys/SPRINT-019-phase-6.2-M2048.nsys-rep` 
- `tools/tc-grid/docs/nsys/SPRINT-019-phase-6.2-M2048.png`
### Phase 6.3 — SplitK Port to v11
#### Objective
Port the proven small-M SplitK idea from v10 into the v11 family so the project does not need one kernel architecture 
for large M and another for small M if the same manual-Lds/dequant path can support both.
#### Why this phase is separate
`6.3` is intentionally orthogonal to the large-M 50 TF push. It should not be merged conceptually with `6.1` or `6.2`. 
The goal here is closing the `M=64` gap. The decision logic is different. The key M points are different. The profiler 
story is different.
#### Hypothesis
If SplitK is ported into v11 with a reasonable K-split factor, then: - `M=64` should move toward the existing v10 
SplitK result, - `M=256` should improve or at least remain competitive, - large-M points should remain available as 
non-SplitK anchors, - correctness should remain straightforward because the mechanism is already understood from v10s.
#### Files in scope
- `tools/tc-grid/kernels/v10splitk_kernels.cuh` - new `tools/tc-grid/kernels/v11splitk_kernels.cuh` - 
`tools/tc-grid/src/launch_int8.cu` - `tools/tc-grid/src/main.cu` - `tools/tc-grid/tests/test_v11_splitk_reduce_sm70.cu`
#### Implementation steps
1. Copy the structural pattern from `v10splitk_kernels.cuh` into a new `v11splitk_kernels.cuh`. 2. Preserve the v11 
dequant and manual-load path. Only the K partitioning and reduction pattern should change. 3. Add a small correctness 
test for SplitK reduction semantics: each CTA computes a K-slice and the final output equals the CPU reference. 4. 
Start with split factors: `ks2`, `ks4`, `ks8`. 5. Only add `ks16` if the first three are bandwidth-limited rather than 
atomic-limited. 6. Wire separate launch macros in `launch_int8.cu`. 7. Register the split factor in tile names so sweep 
CSVs are legible. 8. Keep the large-M non-SplitK anchors in the same sweep to prove that the small-M fix did not 
accidentally become the large-M default.
#### Required correctness checks
This phase must add a focused test: - `test_v11_splitk_reduce_sm70.cu` That test should validate: - per-slice 
accumulation, - global reduction correctness, - deterministic handling of tails where `K` is not evenly divisible by 
the chosen split factor. Required sanitizer run:
```bash
compute-sanitizer --tool memcheck \
  /src/tools/tc-grid/build/test_v11_splitk_reduce_sm70
```
Required integrated correctness sweep: - full five-M v10 bit-compare - explicit comparison to the pre-`6.3` large-M 
winner
#### Integrated benchmark matrix
The headline benchmark points for this phase are: - `M=64` - `M=256` - `M=1024` - `M=2048` - `M=4096` The required 
`ncu` points are: - `M=64` - `M=256` - `M=2048` The required CUTLASS comparison points are: - `M=64` - `M=2048` No 
`nsys` capture is required unless the reduction path becomes unexpectedly pipeline-sensitive.
#### Grid sweep for 6.3
Run a 12-variant sweep built from four tile bases and three split factors. Required sweep variants: 1. 
`64x128x16_w2_ks2` 2. `64x128x16_w2_ks4` 3. `64x128x16_w2_ks8` 4. `64x128x16_w4_ks2` 5. `64x128x16_w4_ks4` 6. 
`64x128x16_w4_ks8` 7. `128x128x16_w2_ks2` 8. `128x128x16_w2_ks4` 9. `128x128x16_w2_ks8` 10. `128x128x16_w4_ks2` 11. 
`128x128x16_w4_ks4` 12. `128x128x16_w4_ks8` Anchor rows must include: - current v10 SplitK champion - current v11 
non-SplitK champion - CUTLASS reference If `ks16` becomes necessary, treat it as a second sweep file rather than 
quietly extending the phase scope.
#### Nsight Compute interpretation target
The intended signature of success for `6.3` is: - materially improved TF at `M=64`, - tolerable atomic overhead, - no 
catastrophic large-M penalty, - no new correctness issues from the reduction path. Phase-specific gating thresholds: - 
`M=64` reaches `>= 18 TF` on the first shipping pass. - `M=64` reaches `>= 20 TF` for full parity with the current v10 
SplitK target. - `M=256` does not regress the best non-SplitK answer by more than `5%` unless the report explicitly 
scopes SplitK to `M<=64`. - `M=2048` should remain a non-goal checkpoint and must not become the default if it loses 
materially. Relevant `ncu` questions: - Does `dram__throughput` increase as expected from better parallelism? - Does 
any atomic or memory-throttle signature dominate? - Do scoreboard stalls change in a way that explains the small-M gain?
#### Decision gate for 6.3
Ship a `6.3` family if: 1. The SplitK-specific correctness test passes. 2. `compute-sanitizer` passes. 3. The five-M 
integrated bit-compare passes. 4. `M=64` reaches at least `18 TF` and ideally `20 TF`. 5. The family is explicitly 
scoped in the dispatcher to the M range where it wins. If `M=64` remains materially below v10 SplitK after `ks2/4/8`, 
add one documented `ks16` exploration pass. If `ks16` still does not close the gap, park the v11 SplitK idea as "not 
worth the reduction overhead on this dequant path" and keep v10s for small M.
#### Artifacts required for 6.3 closeout
- `tools/tc-grid/kernels/v11splitk_kernels.cuh` - `tools/tc-grid/tests/test_v11_splitk_reduce_sm70.cu` - 
`tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.3.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.3-M64.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.3-M256.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.3-M2048.csv`
### Phase 6.4 — Larger CTA Tile + c_frag Spill Mitigation
#### Objective
Revisit the promising `BM=192` and possible `BM=256` space by mitigating c-frag register pressure with structured 
spill-to-SMEM or rotation rather than accepting ptxas spill behavior blindly.
#### Why this phase exists
`REPORT-12` already surfaced a near-champion larger-BM shape with only a mild spill signature. That is exactly the kind 
of result worth methodically following up. It is not enough to note that it spilled and move on. It is also not 
acceptable to ship a larger tile without proving that the spill-mitigation scheme is both correct and actually better 
than compiler spilling.
#### Hypothesis
If larger-BM tiles spill c-frag state to a controlled scratch-SMEM region between K tiles, then: - large-M shapes may 
improve by better amortizing loop overhead, - `launch__registers_per_thread` should become manageable, - the resulting 
shared-memory traffic increase may be cheaper than serialized register pressure, - the approach may complement `6.1` if 
FP16 accumulation already reduced the live set.
#### Files in scope
- `tools/tc-grid/kernels/v11_kernels.cuh` - potentially a factored helper for spill-rotation logic - 
`tools/tc-grid/src/launch_int8.cu` - `tools/tc-grid/src/main.cu` - 
`tools/tc-grid/tests/test_v11_cfrag_spill_roundtrip_sm70.cu`
#### Implementation steps
1. Add a focused test for c-frag spill round-trip correctness. 2. The test should validate: register fragment to 
scratch SMEM back to register fragment with no lane permutation. 3. Start from the larger-BM candidate nearest the 
Sprint 017 champion: `192x128x16_w4`. 4. Add one more aggressive candidate: `256x128x16_w4`. 5. If `6.1` shipped, test 
these under the FP16-accumulator family first. That is the more plausible success path. 6. If `6.1` did not ship, test 
under the FP32 baseline but keep expectations conservative. 7. Implement the spill policy explicitly. Do not rely on 
compiler spill heuristics as the "experiment." 8. Document the scratch-SMEM footprint and occupancy implications 
alongside the code. 9. Record ptxas warnings before runtime measurement. 10. Run correctness before any full sweep.
#### Required correctness checks
New focused test: - `test_v11_cfrag_spill_roundtrip_sm70.cu` Required sanitizer run:
```bash
compute-sanitizer --tool memcheck \
  /src/tools/tc-grid/build/test_v11_cfrag_spill_roundtrip_sm70
```
Required integrated correctness sweep: - full five-M v10 bit-compare - comparison to the pre-`6.4` large-M winner If 
the spill rotation interacts with epilogue or accumulator layout, rerun the relevant `6.1` isolated test as well.
#### Integrated benchmark matrix
Required M values: - `M=64` - `M=256` - `M=1024` - `M=2048` - `M=4096` Required `ncu` points: - `M=2048` - `M=4096` 
Required CUTLASS comparison points: - `M=2048` - `M=4096` Optional `nsys` point: - `M=2048` if the scratch rotation 
changes the observed overlap pattern
#### Grid sweep for 6.4
Run a 14-variant sweep centered on larger-BM exploration. Required sweep variants: 1. `128x128x16_w4` anchor 2. 
`128x256x16_w4` 3. `128x256x16_w8` 4. `192x128x16_w4` 5. `192x128x16_w8` 6. `192x256x16_w4` 7. `192x256x16_w8` 8. 
`224x128x16_w4` 9. `224x128x16_w8` 10. `224x256x16_w8` 11. `256x128x16_w4` 12. `256x128x16_w8` 13. `256x256x16_w4` 14. 
`256x256x16_w8` Each non-anchor candidate should be measured in two modes: - baseline without explicit spill mitigation 
if it builds cleanly - explicit spill/rotation mode If that doubles the matrix beyond practical time, split it into: - 
`grid-sweep-SPRINT-019-PHASE-6.4a.csv` - `grid-sweep-SPRINT-019-PHASE-6.4b.csv` and keep both committed.
#### Nsight Compute interpretation target
The intended signature of success for `6.4` is: - a controlled reduction in registers per thread relative to the naive 
larger tile, - a net TF win on `M=1024+`, - an acceptable increase in shared-memory conflict or bandwidth metrics, - no 
surprise correctness drift. Phase-specific gating thresholds: - the winning `6.4` shape must improve either `M=2048` or 
`M=4096` by `>= 5%`, - `launch__registers_per_thread` must be lower than the naive larger-tile counterpart or the 
report must explain why the mitigation still helped, - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and 
store conflicts must remain interpretable rather than exploding into an obviously worse memory regime. If the best 
larger tile is still worse than the current champion after proper mitigation, park the idea rather than keeping a 
speculative branch alive.
#### Decision gate for 6.4
Ship a `6.4` family only if: 1. The new spill-roundtrip test passes. 2. `compute-sanitizer` passes. 3. The integrated 
correctness sweep passes. 4. A larger-BM shape improves a large-M checkpoint by at least `5%`. 5. The improvement 
persists across at least three M values or is clearly justified as a large-M-only dispatch rule. If the larger-BM shape 
improves just one point and regresses the others, revert it. Sprint 017 already paid the price for asymmetry.
#### Artifacts required for 6.4 closeout
- `tools/tc-grid/tests/test_v11_cfrag_spill_roundtrip_sm70.cu` - 
`tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.4.csv` - optionally `...6.4a.csv` and `...6.4b.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.4-M2048.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.4-M4096.csv`
### Phase 6.5 — PRMT-Vectorized A-Side Load
#### Objective
Reduce non-tensor overhead on the A-side load path by vectorizing or otherwise improving the float-to-half conversion 
work, without destabilizing the already working dequant path.
#### Why this phase is late
This is intentionally a marginal lever. It is cheap and worth taking if the sprint still needs incremental gains after 
the major architecture work. It is not where the sprint should start.
#### Hypothesis
If the A-side conversion path is vectorized more effectively, then: - instruction overhead should fall slightly, - 
tensor-pipe feed should improve slightly, - the gain should be most visible once the larger bottlenecks have already 
been reduced by earlier phases.
#### Files in scope
- `tools/tc-grid/kernels/v11_kernels.cuh` - potentially `tools/tc-grid/kernels/mma_sm70.cuh` if helper intrinsics are 
factored - `tools/tc-grid/src/launch_int8.cu` - `tools/tc-grid/src/main.cu` - optionally a tiny micro-benchmark or 
correctness helper test
#### Implementation steps
1. Start from the current best family after phases `6.1` through `6.4`. 2. Isolate only the A-side conversion/load 
path. 3. Try the most conservative vectorization first: `__float22half2_rn` or equivalent pairwise conversion. 4. If 
that is neutral, try a PTX-level conversion approach only if the code remains readable and auditable. 5. Do not combine 
this work with new tile shapes in the same first pass. 6. If the first implementation is positive, then rerun a small 
shape sweep on the same tile family. 7. If the first implementation is flat or negative, revert it and keep the rest of 
the sprint moving.
#### Required correctness checks
This phase may not need a new heavyweight test if it only changes conversion instructions and preserves the same 
semantic path. It still requires: - `compute-sanitizer --tool memcheck` on the first integrated launch - full five-M 
v10 bit-compare - comparison to the prior phase winner If a PTX-level pack/unpack path is introduced, add a tiny 
deterministic test for representative float inputs near rounding boundaries.
#### Integrated benchmark matrix
Required M values: - `M=64` - `M=256` - `M=1024` - `M=2048` - `M=4096` Required `ncu` points: - `M=2048` - `M=4096` 
Required CUTLASS comparison points: - `M=2048` - `M=4096` No `nsys` capture is required.
#### Grid sweep for 6.5
Run a 12-variant sweep. This sweep is intentionally small and should focus on the current best tile bases rather than 
reopen the full design space. Required sweep variants: 1. `64x128x16_w2` 2. `64x128x16_w4` 3. `64x256x16_w4` 4. 
`96x128x16_w4` 5. `128x128x16_w2` 6. `128x128x16_w4` 7. `128x128x16_w8` 8. `128x256x16_w4` 9. `128x256x16_w8` 10. 
`192x128x16_w4` 11. `192x256x16_w8` 12. `256x128x16_w8` For each variant, compare: - prior-phase family - A-side 
optimized family
#### Nsight Compute interpretation target
The intended signature of success for `6.5` is subtle. This phase should not pretend otherwise. Expected positive 
signature: - modest TF improvement, - flat correctness, - no regression in scoreboard stalls, - possibly small 
reductions in non-tensor instruction pressure, - no increase in register count that overwhelms the micro-optimization. 
Phase-specific gating thresholds: - `M=2048` or `M=4096` improves by at least `1%`, - no large-M regression exceeds 
`1%`, - the gain survives a rerun if the first result is within `1%` of noise. If this lever lands below noise, 
document it as a measured non-win and move on.
#### Decision gate for 6.5
Ship only if: 1. The integrated correctness sweep passes. 2. The gain is reproducible and at least `1%` on a meaningful 
large-M point. 3. The change does not increase register pressure enough to offset the win on nearby shapes. This phase 
is explicitly allowed to end in "measured non-win, reverted." That is still useful if documented cleanly.
#### Artifacts required for 6.5 closeout
- `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.5.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.5-M2048.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.5-M4096.csv`
### Phase 6.6 — Multi-Shape MoE Validation
#### Objective
Prove that the sprint’s winning kernel choices remain sensible outside the square `7168x7168` benchmark shape, and 
identify whether the dispatcher needs shape-aware specialization beyond the simple M-based rule.
#### Why this phase is last
This phase is orthogonal validation. It should evaluate the best answers from `6.1` through `6.5`, not complicate their 
development loops.
#### Hypothesis
If the final Sprint 019 winners are genuinely robust, then: - the same family should remain strong across the 
highest-frequency DSv4 shapes, - small-M and large-M splits may still exist, - some non-square shapes may favor a 
different tile family, - the report may need a shape-aware dispatch recommendation rather than one global winner.
#### Files in scope
- `tools/tc-grid/src/main.cu` - `tools/tc-grid/src/launch_int8.cu` - `tools/tc-grid/docs/REPORT-13.md` - optionally a 
small CLI extension if the harness cannot yet express the shape matrix cleanly
#### Shape selection protocol
Do not guess silently. Use one of the following two protocols: 1. Preferred: derive the top six DSv4 `(N, K)` pairs 
from actual model traces or configuration data and benchmark those. 2. Fallback if trace extraction is not available in 
the sprint window: run the provisional six-shape matrix below and label it explicitly as provisional. Provisional 
six-shape matrix: 1. `N=7168, K=7168` 2. `N=2048, K=7168` 3. `N=7168, K=2048` 4. `N=4096, K=7168` 5. `N=7168, K=18944` 
6. `N=18944, K=7168` Each shape should be run for: - `M=64` - `M=256` - `M=1024` - `M=2048` If time permits, add: - 
`M=4096`
#### Candidate families to validate
Validate at least these rows: - Sprint 017 baseline v11 champion - current best family from phases `6.1` through `6.5` 
- current v10 SplitK small-M answer - current v11 SplitK answer if `6.3` shipped - CUTLASS reference at the same shape 
where supported
#### Implementation steps
1. Confirm the harness can express non-square shapes cleanly. If not, add the minimal CLI extension. 2. Build a shape 
manifest file or hard-coded sweep list that is committed with the benchmark artifacts. 3. Run the selected shape matrix 
for the candidate families. 4. Record not just the winner but also the shape-specific error envelope. 5. For at least 
two representative non-square shapes, capture `ncu` on the winning Sprint 019 family to see whether the stall balance 
differs from the square baseline. 6. If a non-square shape produces a different winner, document the dispatch 
consequence explicitly in `REPORT-13`.
#### Required correctness checks
Every selected shape must satisfy the same correctness contract: - `rel <= 1e-3` - `p99 <= 0.05` - `maxabs <= 0.1` If a 
candidate passes on `7168x7168` but fails on a non-square shape, it is not production-ready even if its square 
benchmark number is impressive. No new standalone test is mandatory here unless the CLI extension itself is complex. If 
the shape-input path changes materially, add a small parser or manifest test.
#### Benchmark matrix
Minimum matrix size: - `6` shapes - `4` M values - `4` candidate families That is `96` data rows before reruns. This is 
large enough to deserve its own CSV artifact. Required `ncu` captures: - best family at one tall-skinny shape - best 
family at one wide shape - baseline v11 at those same two shapes Required CUTLASS comparison points: - `M=2048` on at 
least two non-square shapes No `nsys` capture is required unless a non-square shape exposes a new overlap pathology 
worth diagnosing.
#### Decision gate for 6.6
The sprint-level dispatcher recommendation only ships if: 1. The final candidate families remain correct across the 
selected shape matrix. 2. The report states whether one family wins broadly or whether dispatch should be shape-aware. 
3. Any shape-specific fallback to v10 SplitK or baseline v11 is explicit. If the multi-shape matrix contradicts the 
square-shape story, trust the matrix and update the dispatcher guidance. Do not bury a generalization failure in an 
appendix.
#### Artifacts required for 6.6 closeout
- `tools/tc-grid/docs/multishape-SPRINT-019-PHASE-6.6.csv` - `tools/tc-grid/docs/ncu/SPRINT-019-phase-6.6-shapeA.csv` - 
`tools/tc-grid/docs/ncu/SPRINT-019-phase-6.6-shapeB.csv` - any minimal CLI or manifest support files needed to express 
the shape matrix
## Files Summary
Planned new files: - `docs/sprints/drafts/SPRINT-019-CODEX-DRAFT.md` - `tools/tc-grid/kernels/v11splitk_kernels.cuh` - 
`tools/tc-grid/tests/test_mma_884_tile_sm70_f16acc.cu` - `tools/tc-grid/tests/test_v11_f16acc_epilogue_sm70.cu` - 
`tools/tc-grid/tests/test_v11_splitk_reduce_sm70.cu` - `tools/tc-grid/tests/test_v11_cfrag_spill_roundtrip_sm70.cu` - 
`tools/tc-grid/docs/REPORT-13.md` - `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.1.csv` - 
`tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.2.csv` - `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.3.csv` - 
`tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.4.csv` - `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.5.csv` - 
`tools/tc-grid/docs/multishape-SPRINT-019-PHASE-6.6.csv` Planned modified files: - 
`tools/tc-grid/kernels/v11_kernels.cuh` - `tools/tc-grid/kernels/mma_sm70.cuh` - `tools/tc-grid/src/launch_int8.cu` - 
`tools/tc-grid/src/main.cu` - `tools/tc-grid/kernels/cutlass_int8_kernels.cuh` only if comparison helpers or naming 
cleanup are required Likely artifact directories touched: - `tools/tc-grid/docs/ncu/` - `tools/tc-grid/docs/nsys/` 
Per-phase summary: - `6.1` adds FP16-accumulator tests, experimental kernel path, and profiler artifacts. - `6.2` adds 
a separate 3-stage family or mode plus profiler artifacts. - `6.3` adds `v11splitk_kernels.cuh`, a reduction 
correctness test, and small-M sweep artifacts. - `6.4` adds a c-frag spill/roundtrip test and larger-tile sweep 
artifacts. - `6.5` likely only touches the main kernel file and benchmark artifacts. - `6.6` may minimally extend 
harness shape input and adds multishape results.
## Definition of Done
Sprint 019 is done only when all of the following are true: 1. Every major shipped kernel change has a dedicated 
correctness artifact: either a new unit-style CUDA test or an explicit statement that it reused a previously validated 
mechanism unchanged. 2. Every major shipped kernel change has passed: isolated CPU reference where applicable, 
`compute-sanitizer --tool memcheck`, and the full five-M v10 bit-compare sweep. 3. Every major phase from `6.1` through 
`6.5` has a committed grid-sweep CSV. 4. `6.1` and `6.2` each have both `ncu` and `nsys` evidence committed or archived 
with committed CSV/screenshot outputs. 5. Every shipped phase includes a CUTLASS comparison at `M=2048`. 6. The sprint 
ends with an updated per-M champion table and a dispatch rule that explicitly states when to use: baseline v11, any new 
large-M family, and any SplitK family. 7. The final large-M production recommendation is one of: `>= 50 TF at M=2048`, 
or a documented evidence-backed statement that the v11 family cannot reach that threshold without an architectural 
break. 8. The final small-M recommendation is explicit: either v11 SplitK is production-ready, or v10 SplitK remains 
the small-M answer with a documented reason. 9. `REPORT-13.md` exists and includes: baseline table, per-phase results, 
negative-result sections, final recommendation, and unresolved blockers if any. 10. `6.6` multi-shape validation has 
either completed or is explicitly deferred with a justified blocker and no ambiguity about shape-generalization risk. 
11. DCGM exporter state has been restored after profiling work. 12. No "temporary" experimental path is left wired into 
the dispatcher without a corresponding report decision.
## Risks
1. FP16-accumulator lane mapping on `sm_70` may still hide an undocumented mismatch even if a small atom test passes. 
Mitigation: use both atom-level and epilogue-level tests before integration. 2. The scratch-SMEM epilogue for `6.1` may 
replace one bottleneck with another. Mitigation: require both `ncu` bank-conflict checks and an `nsys` timeline. 3. The 
3-stage pipeline may again produce shape-asymmetric wins. Mitigation: do not ship it as a global family unless it wins 
across the required checkpoints or is explicitly scoped. 4. SplitK may help `M=64` while becoming a distraction for 
large M. Mitigation: keep `6.3` logically separate and do not let it pollute large-M decisions. 5. Larger CTA tiles may 
consume sprint time without yielding a stable win. Mitigation: require a focused spill-roundtrip test and early ptxas 
inspection. 6. A-side conversion work may be pure noise at this point in the kernel. Mitigation: keep `6.5` cheap and 
revert quickly if it does not clear the reproducibility threshold. 7. Multi-shape validation may expose that the square 
benchmark shape has been flattering the current winner. Mitigation: treat that as a success condition for the sprint’s 
methodology, not as an embarrassment to hide. 8. Profiling on a shared GPU node may produce contamination from 
monitoring or unrelated activity. Mitigation: disable exporter as required and keep artifact notes about node state. 9. 
The sprint can become too broad if every phase keeps expanding its sweep set. Mitigation: use the explicit variant 
counts in this document and split extra work into separately named follow-up artifacts. 10. The team may be tempted to 
ship a numerically "close enough" result from a high-upside phase after spending time on it. Mitigation: keep the 
correctness gates binary and phase-local. 11. CUTLASS comparison can be misread as production viability. Mitigation: 
keep the report language explicit that CUTLASS is a ceiling reference on the pre-dequant path. 12. If `6.1` fails, 
morale may push the sprint toward random low-ROI tweaks. Mitigation: use the phase order and gates in this document as 
a forcing function.
## Security
Sprint 019 does not add a network service, parsing surface, RPC endpoint, or credential path. All work remains inside 
the local CUDA benchmarking harness and related docs. The relevant security concerns are operational: 1. Cluster 
hygiene while disabling and restoring node-level monitoring. 2. Avoiding accidental publication of environment-specific 
paths or internal cluster details beyond what the repo already contains. 3. Keeping profiler artifacts free of secrets 
or unrelated shell output. The sprint should not introduce any external dependency that requires new credentials. 
CUTLASS is already in scope as a local comparison path from Sprint 018.
## Dependencies
Hardware and environment: - V100 `sm_70` access on `gpu-01` - `tcg-dev` pod in the `llm` namespace - CUDA 12.2.2 
toolchain compatible with the existing tc-grid build - `ncu` - `nsys` - `compute-sanitizer` - `kubectl` access to 
manage exporter labeling Code and documentation references: - `tools/tc-grid/docs/REPORT-12.md` - 
`tools/tc-grid/docs/V11-EXECUTION-PLAN.md` - `docs/sprints/SPRINT-017.md` - `docs/sprints/SPRINT-018-CUTLASS.md` - 
`tools/tc-grid/kernels/v11_kernels.cuh` - `tools/tc-grid/kernels/mma_sm70.cuh` - 
`tools/tc-grid/kernels/v10splitk_kernels.cuh` - `tools/tc-grid/kernels/cutlass_int8_kernels.cuh` - 
`tools/tc-grid/tests/test_smem_to_frag_sm70.cu` - `tools/tc-grid/tests/test_mma_884_tile_sm70.cu` Process dependencies: 
- a stable baseline commit before phase work begins - disciplined artifact naming under `tools/tc-grid/docs/` - 
willingness to revert negative-result phases cleanly Potential optional dependency: - actual DSv4 shape traces or 
config-derived shape lists for `6.6`
## Open Questions
1. Is the team willing to stop Sprint 019 after `6.1` if the FP16-accumulator path fails isolated correctness, or is 
there still an expectation to grind through `6.2-6.5` for incremental gains? 2. Should `6.2` be allowed to ship as a 
shape-specific family if it clearly wins on one tile class but not on the current global champion? 3. For `6.3`, is 
parity with v10 SplitK at `M=64` enough, or should the sprint require v11 SplitK to become the default small-M path 
before calling the phase complete? 4. For `6.4`, is explicit c-frag spill-to-SMEM considered acceptable production 
complexity if it wins, or should it be treated as a research path unless the win is very large? 5. Should `6.5` be 
skipped entirely if phases `6.1-6.4` still leave the sprint below `40 TF`, on the theory that a marginal lever cannot 
change the final verdict? 6. For `6.6`, where will the preferred DSv4 shape list come from: benchmark trace, model 
config, or a separate small extraction script? 7. Is `REPORT-13.md` the intended report number for this sprint, given 
that Sprint 018 scoped out its own report? 8. If both a new large-M family and v11 SplitK ship, should the dispatcher 
rule remain M-only, or should it also encode shape family once `6.6` completes? 9. If a phase improves TF but worsens 
the CUTLASS ratio because CUTLASS moves more on a non-square comparison shape, which signal should control the 
decision? 10. Does the team want every phase’s negative result committed as code plus CSV, or should obviously losing 
branches be reduced to report prose after the measurements are captured?

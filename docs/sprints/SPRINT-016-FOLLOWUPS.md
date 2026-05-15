# SPRINT-016 Follow-ups

## v9 SMEM round-trip variant (correctness-preserving mixed precision)

- **What**: Implement chunked FP16/FP32 mixed precision using SMEM round-trip per promote. Sequence: (1) `wmma::store_matrix_sync(sBuf, c_h[fm][fn], 16, mem_row_major)`; (2) `wmma::load_matrix_sync(c_f_tmp, sBuf, 16, mem_row_major)` as float fragment; (3) `c_f[fm][fn].x[e] += c_f_tmp.x[e]` (same-layout add).
- **Why**: Original v9 design (register-resident promote) discovered to be incorrect due to WMMA fragment element-ordering mismatch between half and float accumulator types. Half/float c_frags use different lane→element mappings on sm_70, so element-wise `c_f.x[e] += __half2float(c_h.x[e])` corrupts the math (rel=0.71 measured). The SMEM round-trip is correctness-preserving but adds ~900 KB of SMEM traffic per CTA per matmul.
- **Severity**: Important. Mixed-precision is the natural next register-pressure attack; if it works it's worth 3-5% TFLOPS. If it doesn't, it tells us register-pressure has no easy lever and we must do XOR-swizzle or split-N (BN=256 fix).
- **Suggested sprint**: SPRINT-017.
- **Files**: `tools/tc-grid/kernels/v9_kernels.cuh` (rewrite the `promote()` lambda), or new `v9b_kernels.cuh`.

## INT4 BN=256 spill fix via two-pass FRAG_N

- **What**: Split FRAG_N=4 into two outer-product passes of FRAG_N=2 each. Inner K-loop runs twice, re-loading A from SMEM each pass. c_frag count drops from 32 to 16, fits 255-reg cap.
- **Why**: ptxas shows INT4 v3 128x256_w4_fm8_fn4 spills 1664 bytes. With 256 regs for c_frag alone the accumulator exceeds the sm_70 per-thread cap. Spill latency stacks on top of the already-dominant Short Scoreboard stall (Report 9). Trade-off is +SMEM bandwidth on A; expected net positive if spill latency > 2x A-load latency.
- **Severity**: Important. Unlocks BN=256 from the grid, potentially 5-10% TFLOPS uplift at large M for INT4.
- **Suggested sprint**: SPRINT-017.
- **Files**: would create `tools/tc-grid/kernels/v3_bn256_kernels.cuh`; modify `tools/tc-grid/src/launch_int4.cu`.

## Sprint-plan precondition: verify WMMA fragment layout equivalence

- **What**: Add "verify wmma::fragment element ordering equivalence between half and float accumulators on target arch" as a pre-implementation gate for any kernel design that mixes c_frag types.
- **Why**: v9 was planned and partially implemented before this constraint was caught at runtime. A correctness check in CUDA documentation review would have caught it at design time.
- **Severity**: Important — affects future planning quality. Update sprint-plan skill checklist.
- **Suggested sprint**: Tooling/process improvement, not a code sprint.
- **Files**: process/template change in `~/.claude/skills/sprint-plan/` or local sprint-template.md.

## tc-grid harness CSV quoting

- **What**: Quote the `dist` field in tc-grid's CSV output (e.g., `"U(-1,1)"`) so the embedded comma doesn't break field-by-comma splits. Currently parsing scripts need to know to recombine fields 2+3.
- **Why**: I dropped my first analysis pass entirely because `parts = line.split(',')` shifted column indices by 1 due to the unquoted dist. Easy to introduce silent bugs in any downstream consumer.
- **Severity**: Nice-to-have.
- **Suggested sprint**: Future ergonomics pass.
- **Files**: `tools/tc-grid/src/main.cu` (CSV output format).

## scripts/ledger.py for sprint state

- **What**: `/sprint-plan` skill references `scripts/ledger.py` for sprint state tracking. Does not exist in this repo. Either implement or remove the references.
- **Why**: Sprint workflow's intended automated state machine is missing — sprints are completed implicitly without a tracked ledger.
- **Severity**: Nice-to-have.
- **Suggested sprint**: When project conventions become more critical.
- **Files**: `scripts/ledger.py` (new), and `~/.claude/skills/sprint-plan/SKILL.md` cleanup.

## Multi-shape MoE validation (deferred from SPRINT-016 P1)

- **What**: Extend `main.cu` shapes beyond (M ∈ {64,256,1024,2048}, N=K=7168). Identify and add 2-3 DSv4-relevant MoE call shapes (likely from v25a/v25b decode paths).
- **Why**: All current measurements assume N=K=7168 from DSv4-Flash. If MoE call shapes diverge, our winners may not generalize.
- **Severity**: Important if/when v9 or BN=256 produces a winner — generalization should be a release gate.
- **Suggested sprint**: SPRINT-017 or SPRINT-018, paired with whichever optimization actually lands.
- **Files**: `tools/tc-grid/src/main.cu` shape entries.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| v9 SMEM round-trip variant | Important | SPRINT-017 | v9_kernels.cuh, launch_int8.cu |
| INT4 BN=256 spill fix (two-pass FN) | Important | SPRINT-017 | v3_bn256_kernels.cuh (new), launch_int4.cu |
| Sprint-plan WMMA layout precondition | Important | Process | sprint-plan SKILL.md |
| tc-grid CSV quoting | Nice-to-have | Future | main.cu |
| scripts/ledger.py | Nice-to-have | Future | scripts/ledger.py (new) |
| Multi-shape MoE validation | Important | SPRINT-017/018 | main.cu |

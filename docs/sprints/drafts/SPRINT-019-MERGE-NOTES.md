# SPRINT-019 — Merge Notes

## Drafts compared

- **CLAUDE-DRAFT** (49 KB) — structured P0–P6 phases with explicit decision
  gates, dependency-ordered, dependency-justified, ~22–34 hr ETA. Strongest
  on hardware-level specificity (FP16-acc lane probing, register-budget
  table, SplitK fp32 scratch).
- **CODEX-DRAFT** (58 KB) — process-spec-heavy, four-rung correctness
  ladder, standardized profiling commands, multi-shape MoE validation
  in-sprint. Strongest on artifact discipline and verification framework.
- **GEMINI-DRAFT** (19 KB) — tight per-phase summary with decision gates.
  Less detailed than the other two but hits all the right notes.

## Each draft's strengths (preserved in final)

### Claude
- **P0 baseline reproduction phase** as a hard gate (no Gemini/Codex equiv).
- **§3.3 phase-dependency rationale** explaining why FP16 must come first
  (it unlocks register budget for 3-stage and large-BM).
- **P1.1 empirical lane-probing routine** for FP16-acc, with explicit "Do
  NOT assume the FP32 formulas transfer."
- **Per-shape register-budget table** (P2.3) for 3-stage candidacy.
- **`racecheck` + `initcheck` mandate** for SplitK (P3.1).
- **Concrete spill-rotation strategy** for large BM (P4.2).
- **SASS verification step** for PRMT A-side (P5.2).
- **Effort estimate with 3× undocumented-hardware multiplier**.

### Codex
- **Multi-shape MoE validation as Phase 6.6** (Claude defers; user
  ratified in-sprint via interview).
- **Standardized profiling protocol** — exact `ncu` and `nsys` command
  templates so phases produce comparable data.
- **Correctness ladder mental model** — Isolated → Sanitizer → Sweep →
  Baseline.
- **Artifact layout** (per-phase CSVs, PNGs) as first-class.

### Gemini
- **Statistical hostility framing** (sampling bias as the enemy).
- **Risk table format** comparing Claude/Codex/Superior.

## Valid critiques accepted

### From Codex critique (against Claude draft)
- **Accept**: Claude P2.1 says "No new isolated correctness test required"
  for 3-stage. Violates the no-skip rule. FIX: every kernel-template change
  gets a CPU-reference correctness test in `tests/`, no exceptions.
- **Accept**: Claude P5.1 says "No new isolated test needed" for PRMT
  A-side. Same violation. FIX: same.
- **Accept**: Claude P5.4 says "Grid sweep is optional at this phase"
  and downgrades to "≥ 6 shapes". Same violation. FIX: every phase
  sweeps ≥ 12 tile shapes, no exceptions.
- **Accept**: Claude P5 decision gate "TF improved at any M" is softer
  than the global rule. FIX: every phase uses the global ship rule (≥ 3
  of 5 M values, no regression > 2%).
- **Accept**: Claude P3.2 has an "or grid.z==KSPLIT-1 only" branch for
  the SplitK fp32→fp16 cast that's unsafe (no grid-wide global ordering).
  FIX: mandate a separate epilogue kernel (or `cooperative_groups::grid_group`
  on sm_70, but a separate kernel is simpler).
- **Accept**: No repeated-run / variance handling for 1-2% gates. FIX:
  user chose median-of-5 in interview; bake into every commit gate.
- **Accept**: No M-values below 64. FIX: user chose extended M-list
  {1, 8, 32, 64, 256, 1024, 2048, 4096} for all phases.
- **Accept**: No threshold-adjacent M values (M=65, M=257) for dispatcher
  verification. FIX: add to P6 sprint-close verification.
- **Accept**: Missing risks: measurement noise, harness drift, dispatcher
  mis-selection, grid-search overfit, partial correctness masked by aggregate
  rel, SMEM-occupancy footprint, CUTLASS apples-to-oranges at small-M,
  SplitK global-sync assumption, SASS codegen instability, shape-count
  pruning bias. FIX: add to R-list in final.

### From Gemini critique (against both)
- **Accept**: SMEM footprint as an occupancy gate (not just register
  count). FIX: add `launch__shared_mem_per_block_static` to the canonical
  ncu metric set; require an occupancy sanity check at every phase.
- **Accept**: Atomic-collision scaling risk for high KSPLIT. FIX: P3.4
  KSPLIT sweep includes a contention measurement
  (`l1tex__t_sector_pipe_lsu_mem_global_op_atom.sum`).
- **Accept**: Tooling standardization gap in Claude. FIX: copy Codex's
  exact ncu/nsys command templates into the canonical profiling protocol.
- **Accept**: Multi-shape blind spot. FIX: incorporated as P6 via user
  interview.

## Critiques rejected (with reasoning)

- **Reject**: Codex critique suggests "abandon Claude's dependency
  ordering and allow §6.3 (SplitK) to run in parallel with §6.1."
  Reasoning for keeping the dependency order: SplitK should inherit the
  FP16-acc base if §6.1 succeeds, so it's cleaner to land §6.1 first,
  then port SplitK onto the v12 base. If §6.1 fails its correctness
  gate, §6.3 simply forks from v11 instead. The dependency edge is a
  "prefer" not a "require," but the sequential flow is correct.
- **Reject**: Gemini critique implies Claude's Wave-3 ordering is
  arbitrary. It's not — see Claude §3.3 dependency rationale.
- **Reject**: Codex's process-spec-heavy prose style for the whole
  document. The final document needs to be readable as a sprint plan
  AND as an executable checklist. Claude's structure is closer; we
  borrow Codex's profiling-protocol and ladder sections, not its
  overall style.

## Interview refinements applied (USER OVERRIDES)

1. **Scope**: all 6 levers methodically (intent default ratified).
2. **FP16 risk**: **NO time-box** — invest whatever it takes on §6.1.
   This overrides Claude's 8-hr time-box. Implication: §6.1 could
   consume the entire sprint. The user accepts that risk. FIX in final:
   remove the 8-hr abandonment rule; replace with "abandon ONLY if a
   correctness gate proves the lane mapping is unrecoverable, not on
   effort alone."
3. **MoE validation**: in-sprint as P6 (user chose Codex's option).
4. **Measurement rigor**: **median of 5 runs** + **extended M-list
   {1, 8, 32, 64, 256, 1024, 2048, 4096} for EVERY PHASE** (user chose
   the stricter option). Implication: measurement time per phase
   roughly doubles vs the median-of-3 + standard-M-list alternative.
   The user accepts that overhead in service of methodical rigor.

## Structural choices for the final document

1. **Phase numbering**: Adopt Claude's P0–P7 numbering (P0 reproduce,
   P1–P5 = §6.1–§6.5, P6 = §6.6 multi-shape, P7 = close-out).
2. **Naming convention**: Adopt Claude's `v12` family for the FP16-acc
   work, keeping `v11` as fallback (matches the dependency rationale).
3. **Canonical verification structure**: Adopt Codex's four-rung
   ladder (Isolated → Sanitizer → Sweep → Baseline) as the per-phase
   template.
4. **Profiling protocol**: Adopt Codex's standardized `ncu` and `nsys`
   command templates as §2.5 of the final doc.
5. **Decision rule**: One global rule (Intent §"Why this sprint", item
   4) applied to every phase without exception. Phase-specific
   relaxations from Claude's draft are removed.
6. **No-skip enforcement**: Each phase has a CPU-reference correctness
   test in `tools/tc-grid/tests/test_<phase>.cu`. No "this change is
   purely a mainloop rearrangement" exceptions.
7. **Grid sweep size**: ≥ 12 tile shapes per phase, no exceptions.
8. **Variance handling**: Every commit-gate measurement is median-of-5.
9. **M-list**: {1, 8, 32, 64, 256, 1024, 2048, 4096} for every phase.

## What goes in SPRINT-019-DEFERRED.md

Items proposed in drafts or critiques but explicitly out of scope:
- Turbomind `gemm_bench` standalone build (1-2 days; defer to SPRINT-020
  unless this sprint misses 50 TF).
- INT4 BN=256 spill fix (FOLLOWUPS-016; conditional on §6.4 success).
- v5 persistent-CTA revisit (DEFERRED-016; conditional on §6.1 success
  opening occupancy).
- MoE-aware dispatcher integration into live DSv4 inference (separate
  systems sprint).
- tc-grid harness CSV quoting (FOLLOWUPS-016; nice-to-have).
- `scripts/ledger.py` (FOLLOWUPS-016; missing, intentionally skipped).
- Cache-policy `Stream` revisit (sprint-017 negative result; revisit
  only with restricted-to-W_qs scope and per-shape gating).
- 96x128 tile shape exploration (turbomind has it; gridsearch will
  surface it if competitive).

# SPRINT-020 Intent

## Seed prompt

Plan SPRINT-020 to follow SPRINT-019, which shipped at 78% of the 50 TF
M=2048 goal (v12_ms3 champion = 38.98 TF, +11.1% over v11 sprint-017
baseline; v12s = 21.55 TF at M=64 closing the small-M production gap).
The sprint's headline goal was not met; REPORT-13 §6 documents why
(kernel went from gmem-bound to SMEM-bandwidth-bound after the 3-stage
pipeline closed 98% of long_scoreboard). REPORT-13 §9 lists three
forward paths — SPRINT-020 must choose one (or a hybrid).

This sprint is the **architectural decision point** for the V100 INT8
GEMM trajectory: either commit to closing the remaining 50 TF gap via
a major break (turbomind port or CUTLASS extension), or accept the
v12_ms3 ceiling and shift to the next-tier work (deployment integration
+ asymmetric MoE validation).

## Orientation summary

- **Project state**: SPRINT-019 just closed at commit `3829e75b2`. v12_ms3
  + v12s ship as production champions. Branch `sprint-016-tensor-unlock`.
  The v12 kernel family (v12, v12_ms3, v12s) is the codebase baseline now.

- **Recent direction (sprint-by-sprint)**: SPRINT-016 → v3 INT4/FP8/FP4
  ceiling work. SPRINT-017 → v10 manual loader + v11 manual Lds → 35 TF
  baseline. SPRINT-018-CUTLASS → 85 TF ceiling proof via pre-dequant
  (NOT production). SPRINT-019 → v12 FP16-acc + v12_ms3 3-stage + v12s
  SplitK → 39 TF champion. **Theme: push v11 family until ceiling,
  CUTLASS is the comparison reference.**

- **Key modules**: `tools/tc-grid/kernels/v12_kernels.cuh` (3 sibling
  templates), `mma_sm70.cuh` (PTX wrappers), `launch_int8.cu`
  (dispatcher), `cutlass_int8_kernels.cuh` (P1 CUTLASS reference),
  `_deps/cutlass-src/` (CUTLASS 2.11.0 pinned). Outside tc-grid:
  `research/lmdeploy/src/turbomind/kernels/gemm/` (turbomind 884 GEMM
  source).

- **Constraints to respect**: V100 sm_70 sole GPU; CUDA 12.2 in pod;
  DCGM-exporter must be paused before every ncu run; `gpu-02-4090rtx`
  unavailable (qwen3-moe-rotorquant lives there); no laptop GPU; AGENTS.md
  is upstream llama.cpp's no-AI-PR policy and applies to *upstream*
  submissions only (private forks exempt).

- **Vision document status**: **No `docs/sprints/VISION.md` exists.**
  SPRINT-016-018 closed without one; SPRINT-019 didn't create one either.
  Planning from prior CHECKPOINT.md context. If the user wants a vision
  doc, `/vision` is the workflow — flagging but not blocking.

- **SPRINT-019 deferred → now actionable** (per the SPRINT-019-DEFERRED
  conditionals after sprint close):
  - ✅ Turbomind gemm_bench standalone (sprint missed 50 TF → condition
    met)
  - ✅ Turbomind GEMM wholesale port (v12 ceiling confirmed at 39 TF →
    condition met)
  - ⚠️ INT4 BN=256 spill fix (was conditional on P4 success; P4 was
    negative — condition NOT met, stays deferred)
  - ⚠️ v5 persistent-CTA revisit (conditional on P1 success AND
    waves/SM > 2.0; P1 succeeded but waves/SM stayed at 5.6 — no
    occupancy headroom, condition NOT clearly met)
  - ✅ Cache-policy Stream (W_qs only) — main levers settled, eligible
  - ⏭️ CUTLASS 3.x extension — still future, dep upgrade needed
  - ⏭️ 96×128 tile exploration — still future
  - ⏭️ MoE-aware dispatcher in DSv4 — was SPRINT-021 target

- **SPRINT-019 follow-ups — Important items** (per
  SPRINT-019-FOLLOWUPS):
  1. v12s `compute-sanitizer` racecheck + initcheck (deferred during
     sprint execution; sprint §1.2 #2 gate)
  2. tc-grid CLI for asymmetric N≠K (blocks DSv4 MoE multi-shape
     validation; P6 only ran square shapes)
  3. Per-(M, shape) dispatch rule encoded in code (blocks DSv4
     inference integration)

- **Architectural decision required** (REPORT-13 §9): SPRINT-020's
  shape depends on which forward path is chosen — see Open Questions §5.

## Relevant codebase areas

- `tools/tc-grid/kernels/v12_kernels.cuh` — current production
  kernels (v12, v12_ms3, v12s). 503 lines, 3 sibling templates sharing
  the FP16-acc + 1×8-strip-mapping path.
- `tools/tc-grid/kernels/cutlass_int8_kernels.cuh` — CUTLASS 2.11
  reference path (pre-dequant, 85 TF ceiling).
- `tools/tc-grid/src/launch_int8.cu` — dispatcher; versions 50/51/60
  for v12/v12_ms3/v12s; CUTLASS at 40.
- `tools/tc-grid/src/main.cu` — kTiles[] registration; CLI parser
  (--m-list / --nk / --dist). Needs --n-list / --k-list extension
  for asymmetric N≠K validation.
- `research/lmdeploy/src/turbomind/kernels/gemm/` — turbomind's
  full sm_70 GEMM source tree. Pre-existing in repo; not built.
  Relevant for either gemm_bench standalone build OR wholesale port.
- `_deps/cutlass-src/include/cutlass/` — CUTLASS 2.11.0 templates;
  staying on 2.x is the V100 path.

## Constraints

- **Hardware**: V100 sm_70 only. No A100/H100 access. CUTLASS 3.x's
  MixedInputGemm pattern needs sm_80+.
- **Production correctness target**: DSv4-flash is FP4/FP8; v12-family
  gate is `rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0` (recalibrated for
  fp16 accumulator; see V12-DESIGN.md §4.2).
- **Sprint cadence**: budget guidance is 4-6 sessions = ~30-50 hr per
  sprint (sprint-019 metric).
- **No upstream PRs**: AGENTS.md is for upstream llama.cpp; this fork
  is private. AI-collab commits are fine.
- **Sanitizer coverage is a sprint-019 follow-up debt**: v12s atomic
  kernels need racecheck + initcheck cleared. SPRINT-020 should clear
  this before any new atomic-kernel work.

## Success criteria

Sprint succeeds if **at least one** of these is true at close:

1. **Architectural-break path**: Build either (a) turbomind gemm_bench
   standalone OR (b) a turbomind GEMM port and demonstrate a > 5 TF
   improvement at M=2048 over v12_ms3's 38.98 baseline, OR document
   a clean "ceiling proof" that the v12 family is at its peak (e.g.,
   turbomind hits ≤ 41 TF too at the same shape).

2. **Deployment-integration path**: Ship the 3 SPRINT-019 Important
   follow-ups (v12s sanitizer, tc-grid N≠K CLI, per-(M, shape)
   dispatcher), plus validate v12_ms3 / v12s against DSv4-flash
   asymmetric MoE shapes (7168×18944, 18944×7168, 2048×7168), AND
   wire the per-M champion table into the DSv4 inference path (out
   of `tools/tc-grid/`, into the actual model serving code).

3. **Hybrid path**: A mix of (1) and (2) — e.g., gemm_bench bench-only
   (skip wholesale port) + sanitizer cleanup + tc-grid N≠K + asymmetric
   MoE validation. Define "succeeds" as: gemm_bench delivers a
   definitive ceiling comparison AND asymmetric MoE numbers are
   captured.

## Verification strategy

- Per-phase no-skip rule from SPRINT-019 carries forward, with the
  v12-family recalibrated gate (rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0).
- Median-of-5 measurement protocol from `scripts/bench-median.sh`.
- ncu canonical metric set from SPRINT-019 §2.4. Use the CORRECTED
  `-k regex:...` template (SPRINT-019-FOLLOWUPS item 4) — sprint
  §2.5's `--kernel-id ::name:1` is broken.
- For sanitizer work: `compute-sanitizer --tool {memcheck,racecheck,
  initcheck}` on the atomic kernel + adversarial KSPLIT ∈ {2,3,5,8,16}.
- For integration work: end-to-end DSv4-flash sample-generation
  correctness, not just kernel-level rel/p99.

## Uncertainty assessment

- **Correctness**: Medium — adding a major external dep (turbomind
  build or CUTLASS extension) has its own correctness gates. Existing
  v12 kernels are well-validated.
- **Scope**: HIGH — the architectural decision (Q5 below) determines
  whether SPRINT-020 is a 30-hr or 60-hr sprint. Without the user's
  call on direction, the merge phase cannot produce a single coherent
  plan.
- **Architecture**: HIGH — either path involves significant new
  infrastructure (turbomind build, CUTLASS extension, or DSv4
  integration).

## Open questions

1. **Major direction** (HIGH-impact, must answer first): Pick one of
   the three REPORT-13 §9 paths — turbomind port, CUTLASS extension,
   or deployment integration? Or a hybrid?
2. **gemm_bench as reference vs port**: Even if not doing the wholesale
   port, should gemm_bench standalone be in scope as a one-off ceiling
   comparison (1-2 days build engineering per sprint-019 deferred)?
3. **Asymmetric MoE shapes**: Are the 6 DSv4 shapes from sprint-019
   §6.6 still the right catalog (7168×18944, 18944×7168, 2048×7168,
   4096×4096, 7168×7168, 8192×8192)? Or has DSv4 profiling surfaced
   more critical N×K combinations?
4. **Sanitizer follow-up scope**: Race/initcheck on v12s only, or
   expand to all atomic paths in the codebase (v10s, v3s, etc.)?
5. **Deployment integration target**: lmdeploy turbomind backend, or
   the pytorch reference path? Different infrastructure, different
   risk profile.
6. **No-VISION-doc decision**: Should sprint-020 include a `/vision`
   pass first, or proceed without a long-horizon document?

## Items explicitly excluded from this intent (still deferred)

- INT4 BN=256 spill fix (condition not met)
- v5 persistent-CTA revisit (condition not clearly met)
- 96×128 tile exploration
- CUTLASS 3.x dep upgrade (heavy; only if user wants to commit)
- Granular M ∈ [1, 8) dispatch (deployment-side concern)
- tc-grid CSV `dist` field quoting (nice-to-have)

## Vision context

No vision document exists. SPRINT-020 will be planned without
long-horizon sequencing. Flag for the user during interview: should
`/vision` run first to establish a roadmap before committing to one of
the three §9 paths?

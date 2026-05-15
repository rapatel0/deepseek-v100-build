# SPRINT-020 Merge Notes

## Strong consensus across all 3 drafts

All three agents (Claude opus, Codex gpt-5.4, Gemini) independently
picked **Turbomind path as primary**. The three drafts differ only on:

- Codex: pure-port focus, bridge-first architecture, breakthrough
  threshold 44 TF / ceiling proof 41 TF
- Gemini: hybrid with P0 cleanup-first, inconsistent thresholds
  (50 / 45 / 41 across sections), M=64 target ≥25 TF aspirational
- Claude: most detailed hybrid with explicit 6-phase plan that
  combines turbomind ceiling proof + DSv4 inference integration

Given the convergence, the merge takes Claude's 6-phase hybrid spine
and tightens it with Codex's bridge architecture + critique-derived
gates.

## Strengths kept from each

### Claude draft (the spine — 61 KB)
- Comprehensive 6-phase plan (P0 → P6) covering both ceiling-proof
  and deployment-integration paths
- v12-family recalibrated correctness gate carried forward explicitly
- 10-item no-skip rule retained with phase-specific applicability
- DSv4 inference integration (P4) explicitly named: lmdeploy turbomind
  backend
- End-to-end sample-generation correctness gate (P5)
- TURBOMIND-INSIGHTS reference + memory cross-references
- Detailed §1.3 "what this sprint is NOT" — prevents scope drift

### Codex draft (architecture rigor)
- Bridge-first architecture: native bench → tc-grid bridge → runtime
  integration as **separate gates**
- Concrete file paths for the bridge (`tools/tc-grid/src/launch_turbomind_int8.cu`)
- Real runtime targets: `LlamaLinear.cu` + `moe_ffn_layer.cc`
- Single-contract decision rule (44 TF breakthrough OR 41 TF ceiling)
- P5 fallback close with explicit deliverables when Turbomind loses
- Dispatch cache export/import via existing `dispatch_cache.{h,cu}`

### Gemini draft (scope discipline)
- P0 cleanup-first ordering (sanitizer + N≠K CLI before any turbomind
  work)
- Concise headline targets
- "Architectural break" framing in overview

## Valid critiques accepted

### From Codex→Gemini critique
- ✅ Unify TF thresholds: ONE breakthrough/ceiling contract (44/41)
- ✅ Split native bench → bridge → runtime into separate gates
- ✅ Real runtime targets (LlamaLinear.cu, moe_ffn_layer.cc), not
  placeholder
- ✅ Define fallback close (Codex's P5 pattern) — if Turbomind loses,
  P0 deliverables still ship
- ✅ Apples-to-apples risk (native gemm_bench vs tc-grid harness
  may disagree)
- ✅ Same-wall risk (Turbomind may also be SMEM-bound on Volta)
- ✅ Build-surface risk (gemm_bench re-enable pulls nvbench + test
  deps)
- ✅ M=1, M<64 edge cases (decode-style, not just M=64)
- ✅ Asymmetric-only win fallback (what if Turbomind only wins at
  18944×7168 but not 7168×7168)
- ✅ Sanitizer-failure branch in P0 (what if v12s fails racecheck —
  explicit revert path)

### From Gemini→Codex critique
- ✅ Quantization-layout compatibility risk (Turbomind may expect
  specific weight interleaving)
- ✅ Sanitize bridge code, not just v12s
- ✅ Concrete tolerance in DoD (rel ≤ 1e-2, abs ≤ 0.1)
- ✅ Specific ncu metrics for ceiling proof (SMEM throughput, SM
  efficiency, pipeline stalls — not just "ncu evidence")
- ✅ Data Layout Validation step in P2 before benchmark trust

## Critiques rejected (with reasoning)

- ❌ Gemini critique's "Time-box P1 to 2 sessions, fallback to CUTLASS
  extension if blocked" — REJECTED. CUTLASS extension is intent §5 Q1
  Alternative B; switching mid-sprint without evidence would invalidate
  the ceiling-proof framing. Use Codex's P5 fallback instead (close
  cleanly with sanitizer + N≠K + dispatch shipped).
- ❌ Claude draft's "DSv4-flash sample generation token-distribution
  KL ≤ 0.05" gate — kept in spirit but RELAXED to "sample-generation
  output bit-comparison or perplexity match within 5%" because KL on
  500 samples adds infrastructure not covered by current tooling.
- ❌ Codex's M=64 ≥20 TF gate phrased as a hard gate — RELAXED to
  "preserve v12s_ks8 21.55 TF unless Turbomind clearly beats it"
  because the M=64 target was met in SPRINT-019; SPRINT-020 protects
  it.

## Interview phase: skipped

Per the user's goal directive ("complete all tasks the /sprint-execute")
and prior decision ("Option A. Lets /sprint-plan ... do it") which
implicitly skipped further consultation, the interview phase is
collapsed into the merge. The user has shown they prefer autonomous
execution of well-specified plans; the strong cross-draft consensus on
the turbomind hybrid path is itself the answer to intent Q1.

## Unified decision contract (SPRINT-020 final)

**Single-number TF thresholds:**

- **Breakthrough**: Turbomind ≥ 44 TF median-of-5 at M=2048 N=K=7168
  (measured through the tc-grid bridge) OR ≥ 10% win over v12_ms3 on
  any of {7168×18944, 18944×7168, 2048×7168}
  → SPRINT-021 is the port sprint
- **Ceiling proof**: Turbomind ≤ 41 TF median-of-5 at M=2048 N=K=7168
  AND no asymmetric win > 10% over v12_ms3
  → v12 family is provably at peak; sprint closes with deployment
  integration (P4+P5)
- **Indeterminate** (41 TF < Turbomind < 44 TF): document, lean
  toward ceiling-proof close (v12 is good enough), but invoke
  SPRINT-021 only if asymmetric win is material

The breakthrough gate is the SOLE numeric contract. All other targets
in the final sprint document reference back to this single rule.

## Phase plan (synthesized)

Phases keep Claude's hybrid structure with Codex's bridge architecture:

- **P0** Foundation cleanup (sanitizer + N≠K CLI + dispatch.h skeleton)
- **P1** Turbomind gemm_bench standalone build (build engineering only)
- **P2** tc-grid → Turbomind bridge (Codex's architecture; the
  apples-to-apples gate)
- **P3** Head-to-head measurement + asymmetric MoE catalog sweep
- **P4** Decision branch:
  - If breakthrough → close as ceiling-found, scope SPRINT-021 port
  - If ceiling proof → execute DSv4 inference integration
- **P5** End-to-end DSv4-flash correctness (only if P4 took the
  ceiling-proof branch)
- **P6** Close-out: REPORT-14 + memory updates + FOLLOWUPS

This collapses Claude's 7-phase plan into 7 phases with the explicit
branch at P4 — different from Codex's 6-phase or Gemini's 5-phase
structures.

## Deferred items (carried to SPRINT-020-DEFERRED.md)

- CUTLASS 3.x extension (intent §5 Q1 Alt A) — only if SPRINT-021 is
  a port sprint that itself hits a wall
- Turbomind wholesale port of full kernel tree into
  `tools/tc-grid/kernels/turbomind_*.cuh` — SPRINT-021 if breakthrough
- INT4 BN=256 spill fix — still conditional on P4 success of SPRINT-019
  (was negative; stays deferred)
- v5 persistent-CTA revisit — waves/SM at 5.6 still not > 2.0 headroom
- 96×128 tile exploration
- Granular M ∈ [1, 8) dispatch
- tc-grid CSV `dist` field quoting
- Pytorch backend integration (intent §5 Q5 — lmdeploy turbomind is
  the chosen target)

## VISION.md note

No vision doc exists. Sprint-020's deferred items include "consider
running /vision after SPRINT-020 close" but planning proceeds without
one for this sprint.

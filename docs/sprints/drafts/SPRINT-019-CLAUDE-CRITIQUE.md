# SPRINT-019 — Claude Critique of CODEX and GEMINI Drafts

This critique evaluates `SPRINT-019-CODEX-DRAFT.md` and `SPRINT-019-GEMINI-DRAFT.md`
against the mandates in `SPRINT-019-INTENT.md`. The INTENT's load-bearing
demand is unambiguous: **methodical, no-skip execution, grid-search benchmarking
at every step, CPU-reference correctness gate before integration, ncu stall
breakdown, CUTLASS-ratio measurement, and the explicit decision rule
"target stall must drop AND ≥3 of 5 M-values win without >2% regression on the
champion, else revert."** I judge both drafts against that bar — not against
"is this a reasonable performance plan."

The two drafts represent two distinct philosophies:

- **Codex** is a *verification-process plan*. It encodes the methodology as a
  reusable protocol (artifact tree, decision-template, standardized tooling
  commands) and applies it uniformly across §6.1–§6.6.
- **Gemini** is a *technical-narrative plan*. It encodes the methodology as
  prose+pseudocode (a glossary, mathematical foundations, an "appendix"
  structure with code samples).

Both clear the bar in some places and miss it in others. Section 4 details a
side-by-side comparison; sections 2 and 3 dissect each in turn.

---

## 1. Headline assessment

| Dimension | Codex | Gemini |
|---|---|---|
| Methodical-discipline framing | Strong (correctness-ladder, standardized templates) | Weak (verification tier described, but not enforced) |
| Per-phase ncu metric specificity | Strong (canonical metric set + secondary) | Medium (3 metrics cited inline, less coverage) |
| Standardized tool commands | Strong (full ncu+nsys+CUTLASS templates) | Weak (only schematic bash blocks) |
| Grid-sweep specificity | Strong (16-variant lists, anchor rows mandated) | Critically weak (no shape lists, no anchor rule) |
| Per-phase decision gates | Strong (7-question template applied uniformly) | Weak (one gate per sub-phase, no revert rule) |
| Multi-shape MoE validation (§6.6) | Strong (full phase) | Critical gap (degenerate, 2 sub-steps) |
| Definition of Done | Medium (sprint-close checklist only) | Weak (loose appendix checklist) |
| Risk register | Strong (12 risks each with mitigation) | Critically weak (no risk section) |
| CUTLASS-ratio specificity | Strong (M=64/2048/4096 mandated per phase) | Weak (mentioned once in Tier 4, no protocol) |
| Effort estimation | Weak (no ETAs, no time-box) | Weak (no ETAs, no time-box) |
| INTENT alignment (no-skip) | Strong on protocol, weak on time-box | Weak on protocol, weak on time-box |
| INTENT alignment (correctness-first) | Strong | Medium (gates exist but are softer) |

Net: Codex is the closer fit to the INTENT's demand for methodical
no-skipping. Gemini's technical content is solid but its **plan-quality** is
substantially weaker — gates are inconsistent, risks are absent, and the
multi-shape and grid-sweep prescriptions don't satisfy the seed prompt.

---

## 2. Critique: SPRINT-019-CODEX-DRAFT.md

### 2.1 Strengths

#### S1. The correctness ladder (§3.3) is the right mental model
Codex introduces a four-rung correctness contract explicitly:
1. Isolated CPU-reference test
2. `compute-sanitizer --tool memcheck`
3. Full tc-grid bit-compare against v10
4. Bit-compare against previous sprint champion

This maps 1:1 to the INTENT's "every kernel change has a CPU-reference
bit-correctness test BEFORE integration." Step 4 (regression-against-previous-
champion, not just against v10 reference) is a subtle but important addition
that catches "still within v10 envelope but silently moved" bugs. Neither the
INTENT nor Gemini explicitly require step 4.

#### S2. Standardized ncu + nsys command templates (§3.4)
Codex commits to a single command form with a fixed metric set, output naming,
and `--page raw --kernel-name-base demangled` flags. This means a P1.4 ncu
capture is **numerically comparable** to a P5.4 ncu capture. Gemini's bash
block (B.3) shows `--kernel-name-filter "mm_int8_lut_v11"` for a phase that
should target v12 — a copy-paste hazard. Codex's template-first approach is
the discipline the INTENT asked for.

#### S3. Phase 6.6 as a first-class phase
Codex treats multi-shape MoE validation as a real phase with a 6-shape
provisional matrix (`7168x7168`, `2048x7168`, `7168x2048`, `4096x7168`,
`7168x18944`, `18944x7168`), four M values, four candidate families, ncu
captures at two non-square shapes, and a decision gate that **explicitly
allows the multi-shape matrix to override the square-shape result**. This is
the single highest-value addition in the draft. The INTENT marks §6.6 as
"orthogonal" and the user's CLAUDE-DRAFT defers it to SPRINT-020. Codex
rightly recognizes that not running §6.6 in-sprint risks shipping a
"square-shape champion" that fails on DSv4 MoE expert shapes.

#### S4. Per-phase decision template (§3.5) is binary and applied uniformly
The 7-question gate (correctness pass / sanitizer / 5-M sweep / target-stall
moved / wins at ≥3-of-5 M / no >2% regression / CUTLASS-ratio improved) is
applied identically to every phase. This is the "no-skip" forcing function:
every phase faces the same gauntlet, so an author can't quietly relax the
bar for a lever they're emotionally invested in.

#### S5. Negative-results-as-artifacts policy (§3.6)
Codex's grid-sweep rules require: "The sweep must not silently drop a losing
shape. Negative results are part of the record." This is correctness-discipline
applied to the *record-keeping* layer. The INTENT does not explicitly mandate
this; Codex's addition is genuinely value-adding for the audit trail the user
wants from REPORT-13.

#### S6. Per-phase artifact tree (§3.5)
The committed artifact paths
(`grid-sweep-SPRINT-019-PHASE-6.1.csv`, `ncu/SPRINT-019-phase-6.1-*.csv`,
`nsys/SPRINT-019-phase-6.1-*.nsys-rep`) are predeclared. This means a
phase cannot end "ambiguously" — either the named CSV exists with the
required rows or it doesn't. The INTENT's "Nsight Compute report files
archived under `tools/tc-grid/docs/ncu/`" is loose; Codex enforces it.

### 2.2 Weaknesses

#### W1. No ETAs, no time-box, no abandonment rule
This is Codex's single biggest gap relative to the INTENT. The INTENT
explicitly calls out:
- "Time-boxing" as Open Question Q1
- The `feedback_effort_estimation_undocumented_hardware` 3× multiplier in
  the references

Codex's Open Questions §11 asks "Is the team willing to stop Sprint 019 after
6.1 if FP16-accumulator path fails isolated correctness…" — but no decision
is made. There is no equivalent of the 8-hr lane-mapping time-box that the
CLAUDE-DRAFT specifies. Under Codex's plan, a stuck §6.1 could consume the
entire sprint window methodically.

This violates the spirit of "no-skip" in a subtle way: an unbounded
methodical investigation is also a way to fail the sprint. The INTENT's
"abandoning the lever mid-sprint if isolated correctness gate fails" question
(Q2) is left unanswered.

#### W2. §6.1 implementation steps are vague on the lane-mapping problem
Codex §6.1 Implementation step 2 says "Add a new isolated atom correctness
test… that exercises `mma_m8n8k4_row_col_acc_f16` directly." This is
correct as a starting point but **silently assumes the developer knows
the lane→element layout**. The whole reason §6.1 is high-risk is that the
FP16-acc fragment layout differs from FP32-acc and is undocumented in PTX
ISA (per memory `v100_wmma_half_float_frag_layout_mismatch`). The CLAUDE-DRAFT
P1.1 dedicates an explicit step ("Empirically derive the FP16-acc
`thread_offset_C` and `static_offset_C` formulas… Do NOT assume they do").
Codex omits this empirical-derivation discipline entirely. A naive reader
of Codex's §6.1 would write a test using FP32-acc formulas and the test
would pass by coincidence on small matrices, then fail at production scale —
exactly the SPRINT-016 v9 failure mode.

#### W3. Definition of Done is sprint-close-only
Codex's "Definition of Done" (§9) is 12 bullets that describe the *terminal*
state of the sprint. There is no equivalent of the CLAUDE-DRAFT's "Per phase
(every commit)" sub-section — i.e., the DoD applied to every shipped commit
in-flight. The INTENT explicitly says: "**Definition of done — applied at
every commit, not just sprint close**." Codex does not encode this. The
7-question decision gate (§3.5) functionally serves the same purpose, but
calling it "Decision Gate" not "DoD" loses some force; a developer skimming
Codex for the DoD checklist won't necessarily hit the per-phase gate.

#### W4. No sanitizer beyond memcheck
Codex mandates `compute-sanitizer --tool memcheck` everywhere. But the
SplitK phase (§6.3) introduces atomic operations into a shared scratch
buffer — and memcheck alone does not catch atomic ordering bugs or
races on the scratch. The CLAUDE-DRAFT correctly adds `--tool racecheck`
and `--tool initcheck` for §6.3. Codex's §6.3 only requires memcheck.
This is a real correctness hole: a SplitK reduction with a racy zero-init
of the scratch could pass memcheck and the bit-compare on small inputs,
then fail nondeterministically at production scale.

#### W5. Grid-sweep variant lists are good but anchor coverage is uneven
Codex specifies 16 variants for §6.1, 12 for §6.2, 12 for §6.3, 14 for §6.4,
12 for §6.5. All include "anchor rows" (Sprint-017 champion + CUTLASS).
However, only §6.2 explicitly includes the prior phase's winner as an anchor.
For §6.4 ("If 6.1 shipped, test these under the FP16-accumulator family
first"), the dependency is acknowledged but anchor enforcement is loose.
The "no-skip" demand implies that every phase's sweep must compare against
*both* the sprint-017 baseline AND the immediately-prior phase winner; Codex
isn't airtight on this.

#### W6. The `--dist uniform_small` choice is silently fixed
Every Codex command uses `--dist uniform_small`. This is a sensible default
inherited from sprint-017, but the FP16-acc phase introduces a new accumulator
mode where the *distribution* of accumulated values matters for the
correctness envelope (`p99`, `maxabs` are distribution-sensitive). Codex
doesn't include a "stress" distribution (e.g., uniform_wide, dense, or
heavy-tail) as an additional correctness gate. This is a missing edge case.

#### W7. CUTLASS-ratio framing has a subtle hole
Codex §3.4 lists CUTLASS comparison at `M=2048` and `M=4096` as standard,
with `M=64` added for §6.3. Open Question Q9 explicitly asks: "If a phase
improves TF but worsens the CUTLASS ratio because CUTLASS moves more on a
non-square comparison shape, which signal should control the decision?" —
and leaves it unresolved. This is a *real* decision-gate ambiguity that the
INTENT's "≥3 of 5 M values without regressing the champion" rule does not
directly address. Codex flags the problem but doesn't resolve it; a careful
reader could ship a §6.1 family that "improves TF" but fails the implicit
CUTLASS-ratio gate, with no rule for which trumps which.

### 2.3 Gaps in risk analysis

Codex's Risks (§10) has 12 items, well-mitigated individually, but:

- **R-missing-1: SMEM hard ceiling.** V100 SM has 96KB SMEM. §6.4's
  c_frag spill scheme and §6.1's SMEM round-trip epilogue both consume
  SMEM. Codex tracks `launch__registers_per_thread` but doesn't gate on
  `launch__shared_mem_per_block_static`. A larger-BM kernel that fits in
  registers but consumes 64KB+ SMEM drops occupancy to 1 CTA/SM. Neither
  Codex nor Gemini gate on SMEM. (The CLAUDE-DRAFT does include
  `launch__shared_mem_per_block_static` in its canonical metric set; Codex
  does not.)
- **R-missing-2: GPU clock warm-up.** First-kernel-invocation TF on V100
  is depressed by clock ramp. Codex does not mandate a warm-up kernel before
  timed `tc-grid` runs.
- **R-missing-3: gpu-01 noisy neighbor.** Codex mentions cluster hygiene
  generically but doesn't mandate a pre-run `kubectl get pods -n llm
  --field-selector spec.nodeName=gpu-01` check. (CLAUDE-DRAFT R7 does.)
- **R-missing-4: Build-time explosion.** With 12-16 variants per phase × 6
  phases, template instantiations could exceed 100. No gate on build time.
- **R-missing-5: Dispatcher version-ID collision.** Codex assigns new
  versions implicitly per family ("v11f", "v11_ms3", "v11s") without
  explicit dispatcher version numbers, while SPRINT-018 already used
  `version=40` for CUTLASS. No collision audit.

### 2.4 Missing edge cases

- **EC-1: Atomic-add tail handling in §6.3.** When `K` is not evenly
  divisible by `KSPLIT * BK`, the final K-slice has fewer iterations.
  Codex's `test_v11_splitk_reduce_sm70.cu` mentions "deterministic
  handling of tails where K is not evenly divisible," but the
  production benchmark `N=K=7168` IS divisible by all powers of 2 up
  to 2048, so the production path never exercises the tail. Codex
  should mandate a stress shape with a prime/odd K (e.g., `K=7177`)
  to exercise the tail.
- **EC-2: §6.1 SMEM-roundtrip scratch tile sizing.** The scratch tile
  size has to be at least one warp's worth of c_frag (32 lanes × 4
  halves = 128 halves = 256 bytes) but at most one CTA's worth.
  Codex doesn't specify the granularity. A tile that's too small
  serializes warps through the scratch; too large blows SMEM budget.
- **EC-3: `M < KSPLIT` in §6.3.** If the production dispatcher applies
  `v11_splitk_ks16` to a problem with M=8, the kernel's output-tile
  decomposition breaks. Codex's "scope explicitly to small-M" rule
  doesn't cover the M-vs-KSPLIT lower bound.
- **EC-4: Cold L2 vs warm L2 timing.** `tc-grid` reruns timing
  multiple times in a loop and reports the best — but the FIRST run
  exercises cold L2. Codex's grid-sweep rule says "rerun if within 1%
  of winner" but doesn't disambiguate cold-vs-warm.
- **EC-5: ptxas verbosity capture.** Codex says "Record any new ptxas
  spill warnings immediately" in §6.1 step 11. But it doesn't mandate
  a committed `ptxas-verbose-PHASE-N.txt` artifact. Spill warnings can
  be missed in scrollback.
- **EC-6: SMEM bank-conflict regression on the §6.1 epilogue.** The
  SMEM round-trip scatter pattern may itself introduce bank conflicts.
  Codex tracks `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`
  in the canonical metric set; good. But the gate threshold isn't
  specified.

### 2.5 Decision-gate quality

Codex's per-phase gates are **structurally good** (binary, evidence-based,
applied uniformly via the 7-question template) but have specificity gaps:

- §6.1 gate (item 4 in the per-phase template) demands "improves M=2048 TF
  by at least 10% over Sprint 017 baseline OR improves CUTLASS ratio by
  at least 0.05." This is a strong gate. Good.
- §6.1 ncu sub-gates ("≥8 registers dropped, ≥3 absolute pts short-scoreboard,
  ≥5 absolute pts tensor-pipe rise") are quantitative and specific. Good.
- §6.2 gate has a "≥5%" lift threshold and a "no regression by >2%" anti-rule.
  Good.
- §6.3 gate: "M=64 reaches ≥18 TF on first shipping pass, ≥20 TF for full
  parity." Good — sets a partial-success threshold.
- §6.4 gate: "improves either M=2048 or M=4096 by ≥5%" and "improvement
  persists across at least three M values." Good.
- §6.5 gate: "≥1% on a meaningful large-M point" + "no regression >1%".
  Good, with explicit "measured non-win" acceptance.

Weaknesses:
- No phase's gate references SMEM bank-conflict thresholds quantitatively.
- §6.6's gate says "trust the matrix and update the dispatcher guidance"
  but doesn't quantify what counts as a "shape-specific failure."
- Codex's "Did the CUTLASS ratio improve at the headline large-M shape?"
  (question 7 of the 7-question template) is qualitative; "needs an
  explanation in REPORT-13" is not the same as "blocks shipping."

### 2.6 ncu / CUTLASS / grid-sweep specificity

- **ncu**: Codex's primary 7-metric set is canonical and matches the INTENT's
  enumerated metrics exactly. Secondary 6 metrics (`registers_per_thread`,
  `warps_active`, `inst_executed_pipe_tensor`, `dram_throughput`,
  `lts_throughput`, `branch_targets_uniform`) are phase-specific. Good.
  *Gap*: no `launch__shared_mem_per_block_static`. *Gap*: no
  `smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct`
  (gmem long throttle) — relevant for §6.2.
- **CUTLASS**: comparison points at `M=2048, M=4096` (standard) and `M=64`
  for §6.3. Methodology: phase candidate TF, CUTLASS TF, ratio, delta vs
  prior phase. Good. *Gap*: no specification of the CUTLASS `version=40`
  pre-dequant path that the INTENT references.
- **Grid sweep**: 12-16 variants per phase, with anchor rows, with rerun
  rules ("rerun if within 1% of winner"), with negative-result retention.
  Strong. *Gap*: the variant lists are mostly square-aspect-ratio
  tile dimensions; §6.6 partially compensates.

### 2.7 Definition of Done completeness

Codex's DoD (§9) has 12 items at sprint close. It's auditable and complete
for the *terminal* state. **Gap (per W3 above): no per-commit DoD.** The
INTENT's "applied at every commit, not just sprint close" rule is not
explicitly enforced by Codex's prose. The 7-question decision gate
functionally serves this purpose but is not labeled as DoD; a reviewer
auditing a commit message for "did Codex's DoD apply?" gets ambiguity.

**Specific missing items relative to INTENT:**
- "Commit message includes the headline TF + ncu evidence for what stall
  actually dropped" — not in Codex's DoD.
- "If the change didn't drop the target stall → REVERT, no shipping" —
  present in §3.5 (question 4 of decision template) but not in DoD §9.

---

## 3. Critique: SPRINT-019-GEMINI-DRAFT.md

### 3.1 Strengths

#### S1. Glossary and hardware-constraints table
Gemini's "Glossary of Volta (SM70) Terms" and "Hardware Constraints for SM70"
table (255 regs/thread, 96 KB SMEM, 64 warps/SM, 1024 threads/CTA, 32
CTA/SM) are useful onboarding for anyone joining the sprint cold.
Neither Codex nor the INTENT have this. This is a small win.

#### S2. Mathematical foundations section
The "FP16 Accumulator Theoretical Peak" calculation (62.5 → 125 TF) and the
"Register Spilling Constraint" calculation (128x128 tile = 128 floats/lane
× 4 bytes = 512 bytes/thread = saturates 128-reg budget) ground the
performance targets in arithmetic. This is the right kind of pre-flight
analysis that the INTENT implicitly endorses.

#### S3. PRMT bias-trick explanation (Appendix A.2)
The walkthrough of the INT8→FP16 PRMT bias trick (`XOR 0x80, prmt with
0x64646464, subtract 1152`) is the clearest technical explanation in any of
the three drafts. It's not strictly *plan* content, but as a reference
artifact for future kernel work it's valuable.

#### S4. 3-stage pipeline pseudo-code (Appendix A.5)
Gemini provides actual pseudo-code for the 3-stage loop. While Codex
describes the loop in prose ("LDG → RMEM → STS → SMEM → MMA"), Gemini
shows the structure. This makes the §6.2 implementation more actionable.

#### S5. SplitK atomic contention model (Appendix A.3)
Gemini's `T_total = T_compute/Factor + T_atomic*Factor` model articulates
the U-shaped curve in KSPLIT and motivates the {2,4,8,16} sweep. Codex
just says "sweep {2,4,8} first, add 16 if bandwidth-limited" without the
underlying model.

### 3.2 Weaknesses

#### W1. **No risk section at all**
This is Gemini's most serious omission relative to the INTENT and to
Codex. The INTENT's "Uncertainty assessment" explicitly enumerates
correctness, scope, and architecture risks. Codex has 12 risks. **Gemini
has zero formal risk items**. The risks are alluded to in prose ("High
Risk / High Reward" for §6.1) but never enumerated, mitigated, or owned.

For a sprint that is explicitly built around prior incidents
(`v100_wmma_half_float_frag_layout_mismatch`, 3-stage pipeline regression,
__ldcs regression), failing to write a risk register is structurally
inconsistent with the no-skip mandate.

#### W2. **Decision gates are loose and inconsistent**
Gemini's "Sub-Phase Decision Gates" are short and qualitative:
- §6.1.1: "Correctness of the layout map verified by bit-exact match on
  an 8x8 tile." — what about 16x16? what about non-identity input?
- §6.1.3: "TFLOPS ≥ 40." — a single threshold, no specification of which
  M, no relationship to CUTLASS ratio, no "≥3 of 5 M values" rule.
- §6.2.2: "long_scoreboard stall must drop by >5% relative to Phase 6.1."
  — relative to phase 6.1 but on which shape? what if §6.1 didn't ship?
- §6.4.2: "Headline TF at M=4096 must improve relative to BM=128." — by
  how much? what counts as a regression at M=2048?

The INTENT's decision rule — "≥3 of 5 M values win without >2% regression on
the champion, else revert" — is never explicitly invoked. Gemini's gates
appear to be phase-local and don't enforce the cross-M consistency the
INTENT demands.

#### W3. **§6.6 is degenerate**
Gemini's §6.6 has *two* sub-steps: "Execute tc-grid on DSv4 shapes
(N=7168 K=7168, N=2048 K=18944, N=18944 K=2048)" and "Identify per-shape
champions. Decision Gate: Results recorded in REPORT-13.md."

This is a stub. Compare to Codex's full 6-shape × 4-M × 4-family matrix
(96 data rows), ncu captures at non-square shapes, and an explicit "trust
the matrix" override rule. Gemini's §6.6 fails the INTENT's "Multi-shape
MoE validation" line item entirely.

(Note: the CLAUDE-DRAFT explicitly defers §6.6 to SPRINT-020. Gemini
includes it in scope but reduces it to a token presence, which is
arguably worse than honest deferral.)

#### W4. **No ETAs, no time-box**
Same gap as Codex (W1), but more striking in Gemini because Gemini's
draft otherwise reads like a technical specification. No phase ETAs, no
"abandon §6.1 after N hours" rule, no consideration of the 3× multiplier
from `feedback_effort_estimation_undocumented_hardware`. A six-phase
plan with sub-phases and no time-box is a plan that **can't fail
methodically** — there's no clock to stop when correctness work runs
long.

#### W5. **§6.3 atomic-add direction has a correctness bug in the plan**
Sub-Phase 6.3.2 step 2: "Ensure dequantization happens *before* the atomic
add (since scales vary per CTA)." This is actually *backwards* relative to
the v10s SplitK design pattern that the phase is supposedly porting from.
In v10s, each K-slice CTA accumulates into a fp32 scratch *without
dequantization* (the dequantization is already baked into the per-tile mma
results because of how INT8 scales work per CTA), and the dequant happens
at the final epilogue stage. Reading this line literally, a developer would
write a kernel where each CTA dequants before atomic-add, which would
inflate the atomic operand range and harm L2 atomic throughput. This is
likely a phrasing error, but it shows a lack of rigor in the planning
content itself.

#### W6. **No standardized tool commands**
Appendix B has command snippets but they are inconsistent with the rest
of the plan:
- B.3 `ncu --kernel-name-filter "mm_int8_lut_v11"` — should be `v12` or
  the new family name; a copy-paste hazard.
- B.2 `--grid-sweep` flag — does this flag exist in `tc-grid`? Not
  obviously consistent with the harness CLI Codex shows
  (`--m-list 64,256,1024,2048,4096`).
- B.1 `--compare v10 --tolerance 1e-3` — is `--compare` a `tc-grid`
  flag? Not documented elsewhere.
- No `--page raw --csv` for ncu reproducibility.
- No standardized nsys command at all (just "Capture .nsys-rep for
  M=2048" in 6.2.3).

The INTENT lists exact ncu metric names; Gemini's metric usage (Appendix
B.3) shows a single metric per ncu invocation, not the canonical 7+
metric set.

#### W7. **Verification stack (§ Methodology) describes the protocol but doesn't enforce it**
Gemini lists "Tier 1 / Tier 2 / Tier 3 / Tier 4" verification steps as a
methodology section at the top of the doc, but the per-phase sub-phases
don't consistently exercise all four tiers. For example:
- §6.5.1 ("Replace dequant logic with `prmt.b32` bias trick") has one
  "Decision Gate: Correctness pass" — no ncu, no CUTLASS, no grid sweep.
- §6.6 has no Tier 1 or Tier 3 step.
- §6.4 has no Tier 1 step (no isolated correctness test for the spill
  rotation), violating the INTENT's "CPU-reference test BEFORE
  integration" rule.

The Methodology section exists but is not enforced phase-by-phase. This
is exactly the "process declared but not executed" failure mode the
INTENT is trying to prevent.

#### W8. **The "Champion Selection Matrix" (Appendix C.2) introduces ambiguity**
Appendix C.2 lists four ranked selection criteria:
1. Correctness (rel ≤ 1e-3)
2. Peak TFLOPS at M=2048
3. Warp occupancy
4. Bank conflict count

This contradicts the INTENT's primary decision rule ("≥3 of 5 M values"
+ "no >2% regression on champion"). A kernel that wins at M=2048 but
loses at M=4096 by 10% would be selected as "champion" by Gemini's
criterion 2 — but rejected by the INTENT's rule. This is a substantive
methodology error, not a phrasing one.

### 3.3 Gaps in risk analysis

The entire risk register is missing (W1 above). Specific risks that should
have been called out but weren't:

- **FP16-acc lane mapping undocumented** — alluded to in §3 Architecture
  ("introduces a fragment layout mismatch") but no formal risk item, no
  mitigation, no time-box.
- **3-stage register-budget collapse** — implicit in 6.2.1 but not as a
  risk.
- **SplitK atomic contention overhead** — partially captured by the
  Appendix A.3 model, but no risk item.
- **SMEM bank conflicts** introduced by §6.1's SMEM round-trip epilogue
  — entirely unaddressed.
- **gpu-01 noisy neighbor / DCGM exporter / cluster hygiene** — entirely
  absent.
- **Build-time explosion** — entirely absent.
- **Scope creep / time-box overrun** — entirely absent.

### 3.4 Missing edge cases

- **EC-G1: How is correctness verified if §6.1.1 fails?** Gemini's §6.1.1
  decision gate is "Correctness of the layout map verified by bit-exact
  match on an 8x8 tile." What if the layout map *can't be derived*? No
  branch in the plan.
- **EC-G2: Per-shape register-budget mapping in §6.2.1.** Gemini says "4
  CTAs/SM requires ≤ 128 registers" and "3rd stage adds ~16-32 registers."
  But which tile shapes get which budget? No table.
- **EC-G3: §6.3 KSPLIT correctness across non-divisible K.** Same as EC-1
  for Codex. Gemini doesn't mention tail handling at all.
- **EC-G4: §6.4 SMEM ceiling.** Gemini's hardware-constraints table (§6
  in this draft) lists 96 KB SMEM but §6.4 has no SMEM-budget gate.
- **EC-G5: Distribution dependence.** No mention of `uniform_small` vs
  other distributions; correctness envelope is implicitly
  distribution-fixed.
- **EC-G6: PTX `+r` constraint typos.** §3.1's PTX-Level Tensor Core
  Control mentions the MMA instruction but doesn't flag that hand-rolled
  inline-asm operand constraints are a known correctness footgun.

### 3.5 Decision-gate quality

Decision-gate quality is **weak**. Specific issues:

- **Per-sub-phase gates are isolated.** §6.1.1's gate is "8x8 bit-exact";
  §6.1.2's gate is "memcheck passes"; §6.1.3's gate is "TF ≥ 40." There
  is no rule for what happens when §6.1.1 and §6.1.2 pass but §6.1.3
  fails — does §6.1 revert? does it park? does it advance to §6.2 with
  v11 baseline?
- **No cross-M consistency rule.** The INTENT's "≥3 of 5 M values
  without >2% regression" is the load-bearing anti-cherry-pick rule.
  Gemini never explicitly invokes it.
- **Appendix C.1 phase-specific table** says, e.g., "6.1: HMMA
  Throughput ≥ 40 TF — Revert; check lane mapping." This is a single-
  metric, single-M gate. A kernel hitting 42 TF at M=2048 but 25 TF
  at M=4096 would pass.
- **No revert protocol** beyond "Revert; check lane mapping." What
  does "revert" mean — delete the kernel? leave it commented? park
  it as a separate variant? The CLAUDE-DRAFT specifies "revert v12
  from default dispatch (keep code as commented future work), log
  reason in REPORT-13." Gemini doesn't.
- **No multi-evidence requirement.** The INTENT requires ncu evidence
  AND TF improvement AND CUTLASS-ratio improvement. Gemini collapses
  this to one number per phase.

### 3.6 ncu / CUTLASS / grid-sweep specificity

- **ncu**: Gemini's Methodology Tier 3 mentions three metrics ("Long
  Scoreboard, Short Scoreboard, MIO Throttle") and `hmma_cycles_active
  .pct_of_peak`. Appendix B.3 shows one metric per ncu invocation.
  This is substantially less coverage than Codex's 11-metric set or
  the INTENT's enumerated 7-metric set. **Major gap relative to
  the INTENT.**
- **CUTLASS**: Mentioned only as Tier 4 ("Run kernels::int8_cutlass::
  Gemm70 (version=40)") and once in §6.6 commands. No per-phase
  CUTLASS-ratio protocol, no shape-specific points, no rule for
  when CUTLASS-ratio overrides headline TF. **Major gap.**
- **Grid sweep**: Mentioned in Methodology Tier 3 ("Headline TFLOPS
  across all M values") and §6.4.2 ("Run tc-grid --m 4096"). **No
  variant lists, no anchor-row requirement, no rerun protocol, no
  minimum count.** This is the biggest single specificity failure
  relative to the seed prompt's "grid search benchmarking."
- **Sweep CSV artifacts**: not predeclared. No `grid-sweep-SPRINT-019-
  PHASE-N.csv` convention.

### 3.7 Definition of Done completeness

Gemini's "Appendix D: Final Sprint Checklist" has 9 bullets at sprint
close. Items include:
- "All phases (6.1-6.6) implemented and verified."
- "Bit-correctness sweep passes for all DSv4 shapes."
- "Ncu reports archived for all champions."
- "M=64 parity with v10s achieved."
- "M=2048 goal of 50 TF hit (or ceiling documented)."

This is **structurally similar** to Codex's DoD but is loose: "implemented
and verified" doesn't say which artifacts, "Ncu reports archived" doesn't
say where or what metrics. Compared to Codex's 12-item DoD with
explicit artifact paths, Gemini's checklist is roughly half the
specificity.

Critically, like Codex, Gemini has **no per-commit DoD** — only a final
checklist. The INTENT's "applied at every commit, not just sprint close"
mandate is unmet.

---

## 4. Side-by-side: which draft serves the no-skip mandate better

| INTENT criterion | Codex | Gemini |
|---|---|---|
| CPU-reference test BEFORE integration | Yes, mandated in §3.3 + each phase | Methodology Tier 1 mentions it; phases inconsistently include it (§6.4 has no Tier 1 step) |
| compute-sanitizer first launch | Yes, every phase | Mentioned only in §6.1.2; absent elsewhere |
| racecheck for atomics | No (gap W4) | No |
| Full M-sweep at {64,256,1024,2048,4096} | Yes, §3.3 standard | Implicit (Appendix B.1) |
| Bit-compare against v10 reference (rel/p99/maxabs) | Yes, §3.3 with full envelope | Methodology Tier 1 cites rel only |
| ncu canonical metric set | Yes, primary 7 + secondary 6 | Three metrics + one peak metric (W6) |
| ncu at M=2048 AND M=4096 | Yes, every phase | M=2048 only in most phases |
| CUTLASS ratio at M=2048 | Yes, every phase | Only in Tier 4 (one mention) |
| Nsight Systems for §6.1, §6.2 | Yes, §3.4 + per-phase | §6.2.3 only |
| Grid sweep ≥ 12 shapes per phase | Yes, explicit variant lists | Not mandated |
| "Negative results part of the record" | Yes, §3.6 explicit | Not mandated |
| Decision rule: target stall must drop OR revert | Yes, §3.5 question 4 | No |
| Decision rule: ≥3 of 5 M wins OR revert | Yes, §3.5 question 5 | No |
| Decision rule: no >2% champion regression | Yes, §3.5 question 6 | Implicit in some sub-gates |
| DoD per commit (not just sprint close) | No (gap W3) | No |
| Time-box / 3× multiplier | No (gap W1) | No |
| Multi-shape MoE validation (§6.6) in-sprint | Yes, full phase | Yes, but stub (W3) |
| Risk register | Yes, 12 items | None (W1) |

**Headline:** Codex satisfies roughly 14 of 18 INTENT criteria fully; Gemini
satisfies roughly 5 of 18 fully. Codex is structurally the closer fit to the
no-skip / methodical / grid-sweep mandate. Gemini's technical content
(glossary, hardware table, math foundations, pseudo-code, atomic-contention
model) is genuinely valuable as reference material but is not the same thing
as a methodical sprint plan.

---

## 5. Concrete fixes to ship a hybrid

If we treat the two drafts as raw material rather than competitors, the
load-bearing fixes are:

1. **Adopt Codex as the procedural backbone.** Use Codex's §3.3 correctness
   ladder, §3.4 ncu/nsys command templates, §3.5 7-question decision gate,
   §3.6 grid-sweep rules, and §3.7 phase-closeout template as the plan's
   spine.
2. **Promote the 7-question decision gate to "Definition of Done — per
   commit."** Add a new top-level DoD section that splits per-commit and
   sprint-close, modeled on the CLAUDE-DRAFT §6 structure.
3. **Add Codex's missing time-box.** Adopt the CLAUDE-DRAFT's 8-hr lane-
   mapping abandonment rule + per-phase ETAs. Cite
   `feedback_effort_estimation_undocumented_hardware` explicitly.
4. **Inject empirical-lane-derivation discipline into §6.1.** Replace
   Codex §6.1 step 2 ("Add a new isolated atom correctness test that
   exercises mma_m8n8k4_row_col_acc_f16") with the CLAUDE-DRAFT P1.1
   structure: derive `thread_offset_C` empirically with identity inputs,
   do NOT assume FP32-acc formulas transfer.
5. **Add racecheck + initcheck to §6.3.** Codex's memcheck-only sanitizer
   set is insufficient for atomic-heavy SplitK.
6. **Add SMEM-budget gate to §6.4.** Track
   `launch__shared_mem_per_block_static` and gate at 96 KB ceiling with
   occupancy headroom.
7. **Borrow Gemini's Appendix A.3 atomic-contention model and A.5
   3-stage pseudo-code** as reference attachments. They're high-quality
   technical content even if Gemini's plan structure is weaker.
8. **Resolve Codex Open Question Q9** (CUTLASS-ratio-vs-TF conflict)
   before any phase ships — make TF the primary gate, CUTLASS ratio
   the documentation requirement.
9. **Mandate a "warm-up" first kernel** before each timed `tc-grid`
   run to neutralize V100 P-state ramp.
10. **Pre-flight `kubectl get pods -n llm --field-selector
    spec.nodeName=gpu-01`** before each measurement session.

---

## 6. Final verdict

**Codex draft**: a methodical-process plan that meets most of the INTENT's
no-skip discipline. Primary gaps: no time-box, no per-commit DoD, no
racecheck for SplitK, vague on FP16-acc lane derivation.

**Gemini draft**: a technically literate kernel-engineering reference, but
not a sprint plan in the sense the INTENT demands. Primary gaps: no risk
register, decision gates are loose and inconsistent, §6.6 is a stub, no
grid-sweep specificity, no CUTLASS-ratio protocol, no standardized tool
commands, methodology declared but not enforced phase-by-phase.

If forced to pick: **execute Codex's plan**, but graft on (a) the
CLAUDE-DRAFT's empirical-lane-derivation discipline for §6.1, (b) the 8-hr
time-box, (c) per-commit DoD, (d) racecheck for §6.3, and (e) Gemini's
technical appendices as reference material.

The INTENT's load-bearing demand — methodical, no-skip, with grid-search
benchmarking and correctness gates at every major change — is achievable
with Codex as the backbone plus the five grafts above. Gemini in its
current form would let the sprint drift into the same trap that produced
the v9 lane-mapping incident and the SPRINT-017 3-stage asymmetric
regression: a technically defensible kernel ships before the cross-shape /
cross-M / cross-metric evidence is complete.

*End of critique.*

# SPRINT-019 Codex Critique
## 1. Scope
- This critique evaluates `SPRINT-019-CLAUDE-DRAFT.md` and `SPRINT-019-GEMINI-DRAFT.md` against `SPRINT-019-INTENT.md`.
- The evaluation bar is the one the user explicitly set in `INTENT / Seed prompt`: `"detailed steps including grid search benchmarking (ncu, nsight, cutlass) and correctness check for each major change"` and `"no skipping"`.
- The governing process requirement is `INTENT / Why this sprint`, item 1: `"Every kernel change has a CPU-reference bit-correctness test BEFORE integration into the production dispatcher."`
- The governing measurement requirement is `INTENT / Why this sprint`, item 2: `"Every kernel change is benchmarked with tc-grid --m-list sweep AND with ncu stall breakdown across all M ∈ {64, 256, 1024, 2048, 4096}."`
- The governing comparison requirement is `INTENT / Why this sprint`, item 3: `"Every major change is compared to CUTLASS Gemm70's measured runtime on the same shape."`
- The governing ship gate is `INTENT / Why this sprint`, item 4: `"No change is committed unless ncu shows the stall it targets actually dropped, AND the headline TF improved at ≥ 3 of the 5 M values without regressing the champion by > 2%."`
- The governing sweep requirement is `INTENT / Success criteria`: `"Grid sweep with at least 12 tile shapes per major change"`.
- The governing artifact requirement is `INTENT / Success criteria`: `"Nsight Compute report files archived under tools/tc-grid/docs/ncu/"` and `"the CSV exports committed"`.
- The governing Definition of Done requirement is `INTENT / Definition of done — applied at every commit, not just sprint close"`.
- I therefore judge both drafts primarily on process integrity, not prose quality.
- A draft can be technically smart and still fail the sprint if it permits skipped gates.
- A draft can be long and still fail the sprint if its decision rules are vague.
- A draft can be concise and still pass if its gates are explicit, testable, and non-optional.
- The key questions are:
- Does the draft force isolated correctness before integration for every major change?
- Does the draft force the full M sweep, not a hand-picked subset?
- Does the draft force per-phase grid search with enough shapes?
- Does the draft force CUTLASS comparison on the same shape?
- Does the draft force ncu evidence of the intended stall drop?
- Does the draft define what to do on partial wins, asymmetric wins, or ambiguous results?
- Does the draft preserve the user's `"no skipping"` mandate even in late low-ROI phases?

## 2. Executive Judgment
- The Claude draft is materially closer to the intent.
- The Claude draft is the only one that consistently tries to encode `"methodical discipline"` as a phase-by-phase workflow.
- The Claude draft still violates the intent in several important places, especially where it silently relaxes required isolated tests and required sweep breadth.
- The Gemini draft has useful architectural intuition, but as a sprint execution plan it is below the bar the user set.
- The Gemini draft reads more like a technical essay plus a sketch plan than a no-skip execution document.
- The largest difference is that Claude usually asks, `"what is the gate and what is the revert condition?"`
- Gemini more often asks, `"what is the idea and why should it work?"`
- For this sprint, the first question matters more.
- If I had to choose a base document to salvage, I would start from Claude.
- I would not approve either draft unmodified.
- Claude needs targeted tightening.
- Gemini needs structural rework.

## 3. Claude Draft
### 3.1 Strengths
- `Claude §1.2` is directly aligned with the intent's process bar.
- `Claude §1.2` explicitly says `"every phase"` gets `"CPU-reference"`, `"compute-sanitizer"`, `"full M-sweep bit-compare"`, `"ncu stall breakdown"`, `"CUTLASS comparison"`, `"Nsight Systems timeline capture"`, and `"Grid sweep"`.
- That section is the clearest restatement of the user's `"no skipping"` demand in either draft.
- `Claude §1.2`, item 8, uses a real ship gate: `"a change is shipped only if (a) target stall dropped per ncu, (b) headline TF improved at ≥ 3 of 5 M values, (c) no M regresses by > 2% vs prior champion."`
- That gate mirrors `INTENT / Why this sprint`, item 4, almost exactly.
- `Claude §2.4` names a canonical `ncu` metric set instead of vaguely saying "profile it."
- The metric set is mostly appropriate to the current hypotheses: scoreboard stalls, MIO throttle, tensor utilization, bank conflicts, register count, shared memory footprint.
- `Claude §3.3` justifies phase ordering rather than presenting a random list.
- The dependency claim in `Claude §3.3`, `"§6.1 first because FP16 acc halves c_frag register pressure"`, is coherent and tied to the stated failure mode of 3-stage and larger BM.
- `Claude Phase P0` is a strong addition absent in Gemini.
- `Claude P0` correctly treats baseline reproduction as a gate, not a courtesy.
- `Claude P0.2` and `P0.3` lock the comparison set before any kernel changes.
- `Claude P0.4` and `P0.5` require both `ncu` and `nsys` baseline artifacts.
- That matters because otherwise later "improvements" are easy to misattribute.
- `Claude Phase P1` is appropriately labeled `[HIGH RISK]`.
- `Claude P1.1` takes the undocumented FP16 accumulator mapping seriously.
- `Claude P1.1`, step 2, explicitly says `"Do NOT assume they do."`
- That is exactly the right reaction to the `INTENT / Why this sprint` v9 postmortem.
- `Claude P1.1` also includes an abandonment rule.
- The `"8 hr"` time-box in `Claude P1.1` is imperfect, but it is still better than pretending the mapping will obviously fall out.
- `Claude P1.2` is concrete about file scope.
- `Claude P1.2` also includes a register-budget check with an expected threshold.
- `Claude P1.3` restores the full M sweep after the isolated test.
- `Claude P1.4` makes the expected metric direction explicit: `"math_pipe_throttle" should rise`, `"short_scoreboard"` should drop`, `"registers_per_thread"` should fall.
- That is useful because it ties the optimization story to a falsifiable profiler signature.
- `Claude P1.4` also gives an actual shape grid instead of the hand-wave `"run grid-search"`.
- `Claude P1.5` requires `nsys` for a pipeline-sensitive phase.
- `Claude Decision gate P1` is real.
- `Claude Decision gate P1` requires correctness, multi-M improvement, no big regression, ncu evidence, and an improved CUTLASS ratio.
- `Claude Phase P2` correctly frames the 3-stage pipeline as a per-shape dispatch problem, not a universal win.
- That shows it internalized the sprint-017 asymmetric regression.
- `Claude P2.3` demands a per-shape table for `"regs/thread"` and `"within budget?"`
- That is exactly the kind of methodical artifact the intent wants.
- `Claude P2.4` correctly states that the timeline is the only place to verify actual overlap.
- `Claude P2.5` uses a per-shape dispatcher instead of forcing a single winner.
- `Claude Phase P3` is stronger than Gemini on SplitK mechanics.
- `Claude P3.1` includes `racecheck` and `initcheck`, not just `memcheck`.
- `Claude P3.2` notices the need for `"fp32 scratch"` and a separate `"fp32→fp16 cast + STG"` path.
- That is a materially more executable design than Gemini's atomic section.
- `Claude P3.4` sweeps both `KSPLIT` and tile shape.
- That is the right level of specificity for a small-M lever.
- `Claude Decision gate P3` keeps `v10s` as the M=64 fallback if parity is not met.
- That is pragmatic and consistent with the sprint goal.
- `Claude Phase P4` is properly conditional on FP16 accumulation success.
- That avoids pretending larger BM is separable from the register-pressure story.
- `Claude P4.2` describes a concrete spill-rotation strategy instead of saying only "use SMEM spill."
- `Claude P4.4` ties the targeted win to a targeted stall: `"short_scoreboard.pct" must drop`.
- `Claude Phase P5` at least tries to keep even the tail lever inside the measurement loop.
- `Claude P5.2` includes a SASS verification step.
- That is the right instinct for a mechanical instruction-count optimization.
- `Claude Phase P6` has a real close-out plan.
- `Claude P6.2` asks for a lever-by-lever experiment log in `REPORT-13`.
- That is useful because the sprint history here is defined by reverted experiments as much as shipped ones.
- `Claude §5` is specific about affected files.
- `Claude §6 / Per phase` is stronger than many sprint plans' final DoD sections.
- `Claude §7 / Risks` is broad enough to cover both kernel and environment failure modes.
- `Claude §9 / Dependencies` captures pod, driver, CUTLASS pin, and workflow assumptions.
- `Claude §10 / Open questions` surfaces unresolved user choices instead of smuggling them in as fixed assumptions.
- `Claude §11` at least acknowledges the effort multiplier for opaque hardware work.
- `Claude §12` defines `"partial success"` as methodical execution plus documented residual gap.
- That is aligned with the intent's fallback: if 50 TF is unreachable, document why with evidence.

### 3.2 Weaknesses
- The biggest Claude weakness is inconsistency.
- The top-level bar is strict.
- Several later sections quietly soften it.
- `Claude P2.1`, step 1, says `"No new isolated correctness test required"`.
- That directly conflicts with `INTENT / Why this sprint`, item 1: `"Every kernel change has a CPU-reference bit-correctness test BEFORE integration"`.
- A 3-stage pipeline is still a kernel change.
- The claim that it is `"purely a mainloop rearrangement"` is exactly the kind of reasoning the intent was written to block.
- The sprint history already shows that mainloop rearrangements can produce asymmetric regressions.
- `Claude P5.1`, step 1, repeats the same mistake.
- It says `"No new isolated test needed."`
- That again violates the intent's absolute rule.
- The leverage is small, but the process bar was not conditioned on ROI.
- If the user says `"no skipping"`, a low-ROI phase does not get to skip the isolated gate.
- `Claude P5.4` is another relaxation.
- It says `"Grid sweep is optional at this phase — but run it anyway"` and then drops to `"≥ 6 shapes is sufficient"`.
- That conflicts with `INTENT / Success criteria`: `"Grid sweep with at least 12 tile shapes per major change"`.
- The phrase `"optional"` should not appear anywhere in a sprint plan that is supposed to encode `"no skipping"`.
- `Claude §6 / Per phase`, item 7, also codifies the relaxation by saying `"≥ 12 tile shapes (≥ 6 for §6.5)"`.
- That means the deviation is not just in prose.
- It is baked into the draft's DoD.
- `Claude P2 Decision gate` diverges from the global ship rule.
- It says `"if long_scoreboard.pct dropped AND TF improved at all M, ship that shape's 3-stage variant."`
- `"improved at all M"` is not the same as the intent's `"improved at ≥ 3 of the 5 M values without regressing the champion by > 2%."`
- In one sense it is stricter.
- In another sense it is less aligned with the intended decision framework.
- It could reject a shape that is valid under the user's intended rule.
- A methodical plan should not make the decision rule phase-specific without justification.
- `Claude P3 Decision gate` also drifts.
- It requires `"Bit-correct at all 5 M values"` for SplitK even though SplitK is intended as the small-M dispatcher choice.
- That is not wrong, but it is oddly shaped.
- The more important gates for SplitK are small-M correctness, atomic correctness, and ensuring large-M dispatch does not accidentally choose it.
- The draft partly covers that, but the all-5-M framing is not the cleanest expression of the real risk.
- `Claude P3.2`, step 4, contains an unsafe ambiguity.
- It proposes `"a second kernel (or grid.z==KSPLIT-1 only) does the fp32→fp16 cast + STG"`.
- The `"or grid.z==KSPLIT-1 only"` branch is not credible without a real grid-wide ordering guarantee.
- A later `blockIdx.z` slice cannot assume earlier slices have finished globally.
- That is a real correctness hazard.
- A sprint draft at this bar should not leave that branch as an apparently acceptable option.
- `Claude P1.2`, step 4, says `"Allocate __shared__ half c_scratch[BM][BN] (or per-warp tile to limit SMEM)."`
- The parenthetical `"or per-warp tile"` is too under-specified for a memory-budget-sensitive design.
- The shared-memory choice materially affects occupancy and bank conflicts.
- A methodical plan should force a concrete occupancy budget before implementation here.
- `Claude P1.2`, step 5, says `"expect ≤ 96 regs/thread"`.
- The threshold is useful, but the draft never justifies why 96 is the right gate rather than 80, 112, or occupancy-specific thresholds by shape.
- That matters because later phases hinge on exact register headroom claims.
- `Claude P1.3`, step 4, adds a new requirement: `"every row's rel against v10 reference must be ≤ v11's rel for the same row."`
- That is stricter than the intent's published tolerance gate.
- It may be desirable.
- But it needs justification, because FP16 accumulator order changes can produce safe but different relative error.
- If this extra gate is retained, it should be declared up front in `§1.2`, not introduced mid-phase.
- `Claude P4` is weaker on the actual spill-rotation verification than its summary suggests.
- It has a custom Tier 1 test.
- It does not require `racecheck`, despite introducing more complicated SMEM traffic and staging behavior.
- `memcheck` alone is not ideal for a shared-memory rotation algorithm.
- `Claude §6 / Per phase`, item 6, only requires `"CUTLASS Gemm70 (version=40) ratio measured at M=2048."`
- That is weaker than the intent's `Why this sprint`, item 3, and weaker than Claude's own earlier sections that repeatedly mention M=2048 and M=4096.
- If a phase's winning shape is justified partly by M=4096 behavior, the DoD should carry that CUTLASS comparison consistently.
- `Claude P5 Decision gate` says `"TF improved at any M (even +0.5% counts)"`.
- That does not match the global rule `"improved at ≥ 3 of the 5 M values"`.
- Tail phase or not, this is a real weakening.
- It creates an exception for exactly the kind of marginal lever that tends to generate noise wins.
- `Claude §12 / Sprint partially succeeds` says `"v12 at M=2048 lands in [40, 50) TF"` plus methodical execution is partial success.
- That is fine.
- But it does not explicitly restate that all skipped phases must still be recorded as failed or abandoned by gate.
- Because earlier sections already introduced optionality, the partial-success definition should have been stricter about non-skipped evidence.
- `Claude §5 / New files` includes `docs/sprints/SPRINT-019.md` as the final adopted sprint plan.
- That is reasonable.
- But the draft is supposed to be a critique target, not a self-adoption plan.
- This is minor, but it suggests the document is still partly in authoring mode rather than execution mode.
- `Claude §10 / Open questions`, item 1, says `"Confirm with user before P1."`
- That is a real unresolved decision about scope exhaustion vs capped effort.
- Given how much the rest of the document depends on that decision, it should probably have been elevated into a blocking precondition near P0.
- `Claude §11 / Estimated total effort` says `"~3 sessions"` and then `"2–3 weeks of session-time"` via the intent.
- The estimate framing is still somewhat mushy.
- If methodical no-skip execution is the whole point, the plan should be explicit about what gets de-scoped first if time runs out.

### 3.3 Gaps In Risk Analysis
- `Claude §7` is good, but still incomplete.
- There is no explicit risk for `"measurement noise across repeated runs"` even though several gates depend on small percentage moves.
- A sprint with `"no regression > 2%"` gates should specify repeated-run variance handling.
- Otherwise a 1% to 2% move can be noise, not signal.
- There is no explicit risk for `"benchmark harness drift"` beyond P0 baseline reproduction.
- For example, adding many tile registrations can change selection behavior or output formatting.
- The draft should note the risk that the benchmarking harness itself becomes the confounder.
- There is no explicit risk for `"dispatcher mis-selection"` after P6.
- The plan adds per-M and per-shape versions.
- It never names the risk that the wrong version is chosen in production due to selection-order bugs or stale entries.
- There is no explicit risk for `"grid-search overfitting to N=K=7168 only"`.
- The draft consciously defers multi-shape validation.
- That is acceptable.
- But once it adds a per-M dispatcher, the risk that the chosen champion is only a 7168-special should be called out.
- There is no explicit risk for `"partial correctness masked by aggregate rel"`.
- The draft mentions `p99` and `maxabs`, which is good.
- It does not call out the risk that particular tiles or lanes can be wrong while aggregate metrics still pass.
- That matters most for the FP16 accumulator mapping and spill rotation phases.
- There is no explicit risk for `"shared-memory footprint reducing occupancy below the expected benefit"` in P1's SMEM round-trip epilogue.
- The document discusses registers often.
- It is less explicit about the possibility that the fix trades one occupancy limiter for another.
- There is no explicit risk for `"CUTLASS apples-to-oranges comparison"` when the small-M SplitK phase measures a kernel CUTLASS is not tuned for.
- `Claude P3.4` says that comparison is `"for documentation only"`.
- That is fair.
- But the risk section should say so explicitly, because otherwise reviewers may overread or underread that ratio.
- There is no explicit risk for `"global synchronization assumption"` in SplitK.
- This is the biggest missing risk because `P3.2` presents a potentially unsafe one-kernel epilogue option.
- That should have been named and forbidden, not left implicit.
- There is no explicit risk for `"SASS/codegen instability"` in P5.
- The plan wisely asks for `cuobjdump`.
- The risk section never states that compiler codegen could fail to produce the intended packed convert path.
- There is no explicit risk for `"shape count explosion causing selective pruning bias"`.
- `R5` mentions build time.
- It does not mention the more important process risk: if the grid gets pruned ad hoc under time pressure, the whole `"at least 12 tile shapes"` discipline weakens.
- There is no explicit risk for `"M-list incompleteness for small-M inference"`.
- The draft treats M=64 as the proxy for small-M.
- The actual use case in `§2.1` says `"M ∈ [1, 64]"`.
- That gap should be a named risk.

### 3.4 Missing Edge Cases
- The most important missing edge case is small-M values below 64.
- `Claude §2.1` itself says `"generate-phase batches (M ∈ [1, 64])"`.
- The mandatory sweep is still only `{64, 256, 1024, 2048, 4096}`.
- There is no targeted test plan for `M=1`, `M=2`, `M=3`, `M=7`, `M=31`, or `M=63`.
- That matters especially for SplitK and dispatcher logic.
- There is no explicit edge-case handling for non-divisible tile tails in M.
- Because the chosen M values are all large or neat, the plan does not force verification of predication or tail stores.
- There is no explicit edge-case test for `"M just above a dispatch threshold"`, such as `M=65` or `M=257`.
- Since P6 adds a per-M dispatcher, threshold-adjacent values matter.
- There is no explicit edge-case test for `KSPLIT` factors that do not evenly map to workload size or wave count.
- The plan sweeps `KSPLIT ∈ {2,4,8,16}`.
- It does not say what happens if some combinations underutilize or oversubscribe the grid for very small M.
- There is no explicit edge-case plan for repeated SplitK launches with a reused scratch buffer.
- `initcheck` helps once.
- The plan never says that zero-initialization must be validated across back-to-back runs.
- There is no explicit edge-case test for the FP16 accumulator path on degenerate or adversarial data patterns.
- The intent mentions a small isolated test with `"known small inputs"` and not just `uniform_small`.
- Claude mostly inherits that spirit.
- It still does not spell out tests like identity, zeros, sign-heavy inputs, or saturation-adjacent values in the production kernel path.
- There is no explicit edge-case test for bank-conflict-sensitive shapes in the SMEM round-trip epilogue.
- The plan measures bank conflicts in general.
- It does not call out specific tiles where the scratch layout could become pathological.
- There is no explicit edge-case test for `BK=32` in P4 and P5 once earlier evidence suggests it is poor.
- Pruning bad candidates is reasonable.
- But the sprint's methodical bar would be better served by documenting one or two retained control shapes consistently across phases.
- There is no explicit edge-case validation that the CUTLASS comparison uses the same warmup, iteration count, and datatype path each time.
- Since CUTLASS is functioning as a ceiling reference, consistency matters.
- There is no explicit edge-case around `nsys` and `ncu` perturbation themselves.
- The draft requires those tools.
- It does not say how many profiling runs are acceptable before overhead or variance makes the data misleading.
- There is no explicit edge case around stale `kTiles[]` entries.
- The plan keeps appending phase variants.
- It does not require validation that disabled or reverted variants are not still benchmarked by accident.

### 3.5 Decision-Gate Quality
- Claude is strongest here overall.
- `Claude §1.2`, item 8, is excellent.
- The problem is that later phases introduce exceptions.
- `Claude Decision gate P1` is high quality.
- It names correctness, multi-M performance, no-regression, target-stall direction, and CUTLASS ratio.
- That is the most intent-aligned gate in either draft.
- `Claude Decision gate P2` is mixed.
- The good part is per-shape shipping instead of all-or-nothing.
- The weak part is inconsistency with the global rule.
- It should say explicitly that the per-shape choice still must satisfy `"≥ 3 of 5 M values"` or justify why a different criterion is valid for per-shape dispatch.
- `Claude Decision gate P3` is directionally good.
- It keeps the old small-M champion if parity is not met.
- It should more explicitly require evidence that the large-M dispatcher never selects the SplitK variant.
- `Claude Decision gate P4` is solid on targeted stall and production-M improvement.
- It should probably also require a shared-memory footprint/occupancy sanity check as a formal gate, not just a narrative concern.
- `Claude Decision gate P5` is weak by intent standards.
- `"TF improved at any M"` is too permissive.
- For a late-phase micro-optimization, the correct gate is still the same gate unless the plan explicitly reclassifies the phase as a non-major change.
- The draft does not do that.
- The presence of `"revert"` in many phase gates is a strength.
- The revert behavior is generally explicit.
- The absence of a repeated-run or confidence rule is the main statistical weakness.
- None of the gates say "take median of N runs" or "re-run if delta is within noise band."
- That omission matters because several expected wins are in the `+0.5%` to `+2%` range.
- A methodical plan should not ship or revert on single-run noise.
- The Claude draft also does not clearly distinguish "phase fails but code remains as a branch artifact" from "phase is fully reverted from working tree."
- It says `"revert"` often.
- For a real execution plan, artifact retention rules matter.
- `Claude §6 / Per phase`, item 10, is a good backstop.
- But because P2 and P5 already relaxed earlier rules, that backstop is not absolute.

### 3.6 ncu / CUTLASS / Grid-Sweep Specificity
- This is another Claude strength.
- `Claude §2.4` provides a named metric set rather than generic profiling language.
- The inclusion of `"launch__registers_per_thread"` and `"launch__shared_mem_per_block_static"` is especially helpful because the sprint story is resource-pressure-heavy.
- `Claude P1.4` gives a specific grid with `BM`, `BN`, `BK`, and `W`.
- `Claude P2.3` sensibly expands the sweep to compare both 2-stage and 3-stage variants.
- `Claude P3.4` specifies a concrete `KSPLIT` sweep and a concrete total-shape count.
- `Claude P4.4` narrows the search space sensibly once `BK=32` is already known to be poor.
- The artifact naming conventions are also strong.
- The main problem is exception handling.
- `Claude P5.4` undercuts the sweep discipline with `"Grid sweep is optional"`.
- That one sentence weakens the credibility of all earlier specificity.
- The CUTLASS handling is mostly good.
- `Claude P1.4` asks for `"M=2048 and M=4096"` ratios.
- `Claude P2.3` and `P4.4` also point at new-champion-shape comparisons.
- `Claude §6 / Per phase`, item 6, regresses that back to only `"M=2048"`.
- The document should pick one rule and keep it.
- The `ncu` specificity is good on metrics.
- It is weaker on measurement protocol.
- The draft does not specify warmup count, number of profiled kernels, or how to isolate the champion in a crowded `kTiles[]` registry.
- The grid-sweep specificity is good on candidate sets.
- It is weaker on ranking methodology.
- The draft never formalizes whether champion selection uses median TF, best TF, or first-run TF.
- The draft also never defines how many failed shapes are acceptable before concluding a phase's search space was too ambitious.
- `Claude P0.5`, `P1.5`, and `P2.4` require `nsys`.
- That is correct.
- The document does not say what exact visual evidence in the timeline constitutes pass versus fail.
- `"confirm LDG and mma instruction blocks overlap"` is directionally right.
- It could still be tighter, for example by requiring a before/after screenshot with annotated overlap windows.

### 3.7 Definition Of Done Completeness
- `Claude §6` is the better DoD of the two drafts.
- It is phase-aware.
- It includes artifacts, not just outcomes.
- It includes sanitizer coverage.
- It includes profiler exports.
- It includes grid-sweep CSVs.
- It includes `nsys` screenshots.
- It includes a revert condition.
- Those are all real strengths.
- The incompleteness is in the exceptions.
- `Claude §6 / Per phase`, item 1, says `"CPU-reference test passes (Tier 1)."`
- That is correct.
- But the phase bodies for P2 and P5 explicitly waive that requirement.
- So the DoD and the procedures disagree.
- In a methodical sprint, that is a serious flaw.
- `Claude §6 / Per phase`, item 6, only requires `"CUTLASS ratio measured at M=2048."`
- Given how much the draft talks about M=4096 and per-shape selection, I would call that incomplete.
- `Claude §6 / Per phase`, item 7, bakes in the `"≥ 6 for §6.5"` exception.
- That is incomplete relative to the intent's hard bar.
- `Claude §6 / Sprint close`, item 2, says `"v12s (or v11s) at M=64 ≥ 20 TF OR v10s_ks8 remains the M=64 dispatcher choice."`
- That is pragmatic.
- It would be stronger if it also required documentation of why v12s failed if fallback is retained.
- `Claude §6 / Sprint close`, item 4, requires the final dispatcher rule and a final full-M sweep.
- That is good.
- It does not require threshold-adjacent or below-64 validation for the dispatcher.
- `Claude §6 / Sprint close`, item 8, handles new `ptxas` spills.
- That is useful.
- The DoD does not require repeated-run confirmation for small deltas.
- It does not require preservation of rejected shapes' data in `REPORT-13`.
- It does not require that every abandoned phase cite the exact failing gate.
- Those are not fatal omissions, but they would strengthen the audit trail.

### 3.8 Claude Draft Verdict

- Claude is a strong draft with important integrity leaks.
- It understands the user's intent.
- It sometimes chooses convenience over that intent in later details.
- The most serious violations are:
- `P2.1` waiving isolated correctness for the 3-stage pipeline.
- `P5.1` waiving isolated correctness for the A-side convert change.
- `P5.4` and `§6` weakening the 12-shape sweep requirement.
- `P5 Decision gate` weakening the global multi-M improvement rule.
- `P3.2` leaving an unsafe SplitK epilogue option on the table.
- If those issues are fixed, the Claude draft is very close to an execution-ready sprint plan.
- Without those fixes, it is still the better draft, but it does not fully satisfy the explicit `"no skipping"` mandate.

### 3.9 Claude Draft Recommended Fixes

- Replace `P2.1`, step 1, with an actual isolated correctness test for the 3-stage variant before dispatcher wiring.
- Replace `P5.1`, step 1, with an actual isolated test for the vectorized convert path before production integration.
- Remove the phrase `"Grid sweep is optional"` from `P5.4`.
- Restore the `12-shape minimum` for `§6.5`, or explicitly reclassify `§6.5` as non-major and justify the exception near `§1.2`.
- Make `P5 Decision gate` match the global rule or explicitly justify a different rule in `§1.2`.
- Delete the `"or grid.z==KSPLIT-1 only"` option from `P3.2`.
- Require a real second kernel or another globally correct reduction mechanism for SplitK finalization.
- Add a repeated-run rule for any expected improvement below `2%`.
- Add dispatcher-threshold tests for `M=1`, `M=63`, `M=64`, `M=65`, `M=255`, `M=256`, and `M=257`.
- Make the DoD's CUTLASS rule match the earlier per-phase sections consistently.

## 4. Gemini Draft

### 4.1 Strengths

- The Gemini draft has clear technical intuition.
- `Gemini / Overview` correctly identifies the main macro-problem: `"a 62.5 TFLOPS theoretical limit imposed by FP32 accumulation"` plus `"high HBM latency"`.
- `Gemini / Overview` also correctly says the sprint must be `"empirically validated using a stack of low-level profiling tools (ncu, nsys), bit-exact correctness checks, and architectural comparison against CUTLASS."`
- `Gemini / Architecture Deep-Dive` is more explanatory than Claude.
- That may help a contributor understand the optimization levers before execution.
- `Gemini §3. FP16 Accumulator and the Epilogue Problem` at least recognizes that the accumulator change is not just a drop-in type swap.
- `Gemini §4. Pipeline Evolution` correctly frames the 3-stage idea as hiding `"long_scoreboard"` stalls via an RMEM stage.
- `Gemini / Hardware Constraints` usefully reminds the reader that registers and shared memory are hard ceilings.
- `Gemini / Mathematical Foundations`, especially the `"Register Spilling Constraint"` section, tries to make the resource trade concrete.
- `Gemini / Methodology` at least enumerates a tiered validation stack.
- `Gemini Phase 6.1` correctly gives FP16 accumulator top priority.
- `Gemini Phase 6.2` at least notices the need for register-budget analysis before using a 3-stage path.
- `Gemini Phase 6.3` correctly treats SplitK as the small-M fix.
- `Gemini Appendix B` provides command examples.
- `Gemini Appendix C` acknowledges that each phase needs a decision gate.
- `Gemini Appendix D` ends with a checklist rather than pretending a sprint closes automatically when code compiles.
- In short, the Gemini draft understands the problem space.
- Its weakness is not lack of ideas.
- Its weakness is lack of procedural rigor.

### 4.2 Weaknesses

- The Gemini draft never really encodes the user's `"no skipping"` bar.
- It says the right words in the overview.
- It does not bind those words to the phases tightly enough.
- The missing `P0` baseline reproduction phase is a major omission.
- Unlike Claude, Gemini never forces a pre-change reproduction of the sprint-017 baseline.
- That means the whole sprint can proceed on an unstable measurement foundation.
- `Gemini / Methodology` is too generic.
- It mentions correctness, integration, `ncu`, and CUTLASS.
- It does not require per-phase `nsys`.
- It does not require the exact full M-list sweep at each phase.
- It does not require `"no change is committed unless ncu shows the stall it targets actually dropped"`.
- It does not require `"improved at ≥ 3 of the 5 M values without regressing the champion by > 2%."`
- `Gemini Phase 6.1.1`, step 4, is the most serious methodological failure in the document.
- It states a derived mapping formula: `"lane_to_row = (lane % 4) * 2 + (element / 2), lane_to_col = (lane / 4)." `
- The intent explicitly warned that this mapping is undocumented.
- Claude correctly says `"Do NOT assume they do."`
- Gemini instead writes down a formula before showing any empirical derivation procedure that would validate it.
- That is precisely the kind of assumption-driven leap the sprint is trying to stop.
- `Gemini Phase 6.1.2`, step 1, hardcodes `"__shared__ half sC_scratch[128][128]"`.
- That is not parameterized by shape.
- It does not account for total shared-memory budget with existing A/B staging.
- It suggests a design decision before a resource budget.
- `Gemini Phase 6.1.3` has a weak gate.
- The decision gate is only `"TFLOPS ≥ 40."`
- That is far below the user's actual ship rule.
- It omits no-regression checks.
- It omits the multi-M criterion.
- It omits target-stall confirmation.
- It omits CUTLASS ratio.
- It omits a revert instruction.
- `Gemini Phase 6.2.1`, step 3, says `"3rd stage adds ~16-32 registers"`.
- That is plausible.
- It is still just a narrative estimate.
- There is no requirement to measure actual `ptxas` output or `launch__registers_per_thread` before deciding.
- `Gemini Phase 6.2.2` uses `racecheck` as the notable sanitizer.
- It does not call for `memcheck` there.
- The intent explicitly called for `memcheck` on first launch of every new kernel template.
- `Gemini Phase 6.2.3` is not bad by itself.
- The problem is that `nsys` appears here as a nice proof step, not as part of an overarching mandatory artifact set.
- `Gemini Phase 6.3.2`, step 2, says `"Ensure dequantization happens before the atomic add (since scales vary per CTA)." `
- That explanation is underdeveloped and potentially confused.
- The more important missing mechanics are scratch-buffer lifetime, zero-init, reduction ordering, and finalization.
- Gemini never provides a correct finalization model like Claude's `"fp32 scratch"` plus separate cast/write.
- `Gemini Phase 6.3.3` makes `"M=64 TFLOPS ≥ 20"` the whole gate.
- That ignores full-M behavior and no-regression constraints.
- `Gemini Phase 6.4.2` says `"Run tc-grid --m 4096."`
- That directly conflicts with the intent's mandatory sweep across `{64, 256, 1024, 2048, 4096}`.
- This is a clear no-skip failure.
- `Gemini Phase 6.5.1`, step 1, appears conceptually wrong.
- It says `"Replace dequant logic with prmt.b32 bias trick."`
- But the current-state summaries already say v11 uses `"PRMT bias-trick"` for dequant.
- So the draft seems to be proposing an optimization the baseline already has.
- That suggests it does not fully understand the current production kernel.
- `Gemini Phase 6.6` is out of scope by the intent.
- `INTENT / Out of scope` explicitly defers `"Multi-shape MoE validation (REPORT-12 §6.6)"`.
- Gemini includes it as an implementation phase anyway.
- That is a direct scope-control failure.
- `Gemini Appendix A` spends many lines on educational deep dives.
- The sprint needed more execution gates, not more didactic background.
- `Gemini Appendix B.2` gives a single generic grid-sweep command.
- It does not encode per-phase grid shape sets or the `"at least 12 tile shapes per major change"` rule.
- `Gemini Appendix C.1` is much too thin.
- `Phase 6.1 | HMMA Throughput | ≥ 40 TF | Revert` is not an adequate decision matrix for this sprint.
- `Gemini Appendix C.2` defines champion selection using `"Highest value at M=2048."`
- That is wrong for a sprint with an explicit M=64 parity goal and an explicit multi-M no-regression rule.
- `Gemini Appendix D` is incomplete.
- It does not require per-phase isolated CPU-reference tests.
- It does not require 12-shape sweeps.
- It does not require the target-stall drop rule.
- It does not require full-M no-regression validation.
- It does not require abandonment logging for failed phases.
- Put plainly, the Gemini draft promises rigor.
- It does not operationalize rigor.

### 4.3 Gaps In Risk Analysis

- The Gemini draft effectively has no explicit risk section.
- That is a serious deficiency for this sprint.
- The intent itself is built on postmortems of prior regressions.
- A sprint draft that omits a risk register misses one of the main lessons.
- There is no named risk for undocumented FP16 accumulator lane mapping.
- Instead, the draft jumps to a formula.
- There is no named risk for asymmetric 3-stage wins and large-shape regressions.
- There is no named risk for SplitK atomic correctness.
- There is no named risk for scratch-buffer initialization.
- There is no named risk for invalid global ordering assumptions in the SplitK epilogue.
- There is no named risk for grid-search build explosion.
- There is no named risk for `ncu` contamination from `dcgm-exporter`.
- There is no named risk for GPU contention on the V100 node.
- There is no named risk for measurement noise on tiny expected wins like `+1%`.
- There is no named risk for CUTLASS comparison misuse at small M.
- There is no named risk for pruning the search space too early.
- There is no named risk for incorrect current-state assumptions, which is especially relevant because `Phase 6.5` appears to misread the current dequant path.
- The absence of an explicit risk section also weakens every later phase gate.
- Without a named risk model, the gates cannot clearly target the likely failure modes.

### 4.4 Missing Edge Cases

- Gemini misses even more edge cases than Claude.
- There is no P0 baseline reproduction edge case handling at all.
- There is no explicit testing for `M < 64`, despite the MoE use case in `"small, variable batches (M=64, 256)."`
- There is no explicit testing for threshold-adjacent values like `M=63` or `M=65`.
- There is no explicit testing of tile-tail behavior.
- There is no explicit testing of zero or identity patterns in isolated correctness tests.
- `Gemini Phase 6.1.1` mentions an identity fragment.
- It does not carry that discipline forward into production-kernel correctness gates.
- There is no explicit edge-case testing for repeated SplitK launches with a reused scratch buffer.
- There is no explicit validation that the scratch buffer is zeroed between runs.
- There is no explicit edge-case plan for `KSPLIT` factor interaction with tiny M.
- There is no explicit edge-case plan for the FP16 accumulator path under sign-heavy or saturation-adjacent inputs.
- There is no explicit edge-case around shared-memory footprint and occupancy once `sC_scratch[128][128]` is introduced.
- There is no explicit edge-case around `BK=32` control shapes or search-space controls.
- There is no explicit edge-case around `ncu` kernel filters missing the new v12 variants.
- That last one is not hypothetical.
- `Gemini Appendix B.3` uses `--kernel-name-filter "mm_int8_lut_v11"`.
- That would miss a renamed `v12` kernel unless updated.
- There is also an apparent command typo in `Appendix B.3`: `"--nk 7168 --nk 1"`.
- That weakens confidence in the benchmark recipe itself.

### 4.5 Decision-Gate Quality

- This is where Gemini falls furthest behind.
- The draft has gates.
- Most of them are too narrow, too local, or too weak.
- `Gemini Phase 6.1.3` uses `"TFLOPS ≥ 40"` as the gate.
- That ignores the user's actual decision rule.
- A kernel could hit 40 TF at M=2048 and still regress other M values badly.
- A kernel could hit 40 TF without the targeted stall dropping.
- A kernel could hit 40 TF and still be far enough below expectation that the lever should be reconsidered.
- `Gemini Phase 6.2.2` uses `"long_scoreboard stall must drop by > 5%"`.
- That is a reasonable sub-gate.
- It is not enough by itself.
- There is no multi-M no-regression companion gate.
- `Gemini Phase 6.3.3` uses `"M=64 TFLOPS ≥ 20"`.
- Again, it is necessary but not sufficient.
- `Gemini Phase 6.4.2` uses `"Headline TF at M=4096 must improve relative to BM=128."`
- That ignores every other M value and all correctness/profiler conditions apart from an earlier `"No regression in maxabs"` check.
- `Gemini Phase 6.5.1` uses `"Correctness pass."`
- There is no explicit performance gate at all for the final lever.
- `Gemini Appendix C.1` reduces the sprint to a four-row table.
- It does not even include phase `6.5`.
- It does not include the `"no M regresses by > 2%"` rule.
- It does not include the `"≥ 3 of 5 M values"` rule.
- It does not include CUTLASS-ratio requirements.
- It does not include `nsys` evidence for overlap phases.
- `Gemini Appendix C.2` selects champions by `"Highest value at M=2048"`.
- That is an especially bad fit for this sprint's stated goals.
- It would bias the whole sprint toward large-M wins and can easily discard the right small-M dispatcher choice.
- No Gemini gate clearly says "revert if the targeted stall did not drop."
- The document says "revert" in `Appendix C.1`.
- It does not tie that revert to explicit profiler evidence phase by phase.
- There is also no clear hierarchy between phase-local gates and final sprint gates.
- The result is that a contributor could interpret almost any local win as enough to proceed.
- That is the opposite of the user-requested discipline.

### 4.6 ncu / CUTLASS / Grid-Sweep Specificity

- Gemini mentions all three tools.
- It does not specify them well enough.
- `Gemini / Methodology / Tier 3` says `"Ncu"` and lists a few metric themes.
- It does not define a canonical metric set.
- It does not state exact export paths or committed artifact rules.
- `Gemini / Methodology / Tier 4` asks for CUTLASS ratio.
- It does not embed the same-shape requirement as explicitly as the intent does.
- `Gemini Phase 6.1.3` says `"Run tc-grid grid-search."`
- That is too vague.
- `Gemini Phase 6.2` does not define a phase-specific grid at all.
- `Gemini Phase 6.3` only sweeps `KSPLIT`, not a full shape set.
- `Gemini Phase 6.4.2` only benchmarks `--m 4096`.
- That is explicitly below the bar.
- `Gemini Appendix B.2` gives a single generic `"--grid-sweep"` command.
- It does not define search dimensions, constraints, control shapes, or minimum count.
- `Gemini Appendix B.3` is also flawed as a profiling recipe.
- The `v11`-only kernel filter is not future-proof.
- The duplicated `--nk` flags suggest the command was not sanity-checked.
- There is no explicit `ncu` CSV artifact naming scheme.
- There is no explicit `nsys` artifact naming scheme.
- There is no explicit requirement that CUTLASS be measured at both production points or at each shipped phase's winning shape.
- There is no explicit repeated-run rule for profiler noise.
- Overall, the Gemini draft uses the names of the right tools.
- It does not operationalize them with the specificity a no-skip sprint requires.

### 4.7 Definition Of Done Completeness

- Gemini's DoD is much thinner than Claude's.
- `Gemini Appendix D` is a checklist, not a real process DoD.
- It says `"All phases (6.1-6.6) implemented and verified."`
- That is already misaligned because `6.6` is supposed to be out of scope.
- It says `"Bit-correctness sweep passes for all DSv4 shapes."`
- That is too vague and actually broader than the document elsewhere supports.
- It never defines the exact shape set or links it back to the fixed M-list.
- It says `"Ncu reports archived for all champions."`
- It does not say which metrics.
- It does not say per phase or only final.
- It says `"Nsys timelines confirm compute/memory overlap for 3rd stage."`
- It does not mention the FP16 accumulator phase even though Gemini earlier presents its own epilogue restructuring there.
- It says `"CUTLASS ratio recorded for the final champion."`
- That is incomplete relative to the per-phase measurement requirement in the intent.
- It says `"No untracked ptxas register spill warnings."`
- That is a useful thought.
- It is not enough without explicit spill gates per phase.
- The Gemini DoD does not require isolated CPU-reference tests before integration.
- It does not require `memcheck` on first launch of every new kernel template.
- It does not require `racecheck` or `initcheck` for SplitK.
- It does not require 12-shape grid sweeps.
- It does not require the target-stall-drop rule.
- It does not require the `≥ 3 of 5 M` rule.
- It does not require the `> 2% regression` rule.
- It does not require documented reverts.
- It does not require a phase-by-phase report artifact like `REPORT-13`.
- As a result, the Gemini DoD would allow large parts of the sprint discipline to disappear in execution.

### 4.8 Gemini Draft Verdict

- Gemini is informative but not execution-safe.
- It has solid technical instincts.
- It repeatedly fails to translate those instincts into mandatory gates.
- The most serious issues are:
- Missing baseline reproduction.
- Premature assumption of the FP16 accumulator lane mapping.
- In-scope/out-of-scope confusion by including phase `6.6`.
- Weak and incomplete phase decision gates.
- Insufficient `ncu` / CUTLASS / grid-sweep specificity.
- A DoD that does not encode the user's actual process bar.
- The mistaken-looking `PRMT` phase description is also important because it suggests the baseline kernel was not fully understood.
- I would not use the Gemini draft as the base sprint document.
- I would mine it for explanatory language and perhaps a few appendix ideas.
- I would not trust it to enforce methodical no-skip execution.

### 4.9 Gemini Draft Recommended Fixes

- Add a real P0 reproduction phase before any optimization phase.
- Remove `Phase 6.6` from the in-scope implementation body and move it to deferred follow-ups.
- Replace the assumed FP16 accumulator formula in `6.1.1` with an empirical derivation protocol and an explicit `"do not assume"` warning.
- Parameterize the SMEM scratch design and require an occupancy/shared-memory budget check before implementation.
- Replace every local throughput-only gate with the full sprint gate or a justified refinement of it.
- Define a canonical `ncu` metric set and artifact naming scheme.
- Require the full M sweep `{64, 256, 1024, 2048, 4096}` at every major phase.
- Define per-phase shape grids with `≥ 12` shapes.
- Add `racecheck` and `initcheck` requirements for SplitK.
- Replace the final checklist with a phase-aware DoD that mirrors the intent's `"applied at every commit"` language.
- Rework `Phase 6.5` so it matches the actual current-state kernel if the PRMT dequant path is already present.
- Fix `Appendix B.3` command issues before anyone tries to use them.

## 5. Side-By-Side Comparison

### 5.1 Which Draft Better Matches The User's Bar

- On `"methodical no-skip execution"`, Claude wins clearly.
- On `"correctness gate before integration"`, Claude mostly wins, but still has two notable violations.
- On `"grid-search benchmarking"`, Claude wins clearly.
- On `"ncu specificity"`, Claude wins clearly.
- On `"CUTLASS comparison discipline"`, Claude wins narrowly.
- On `"risk analysis"`, Claude wins by a large margin.
- On `"Definition of Done completeness"`, Claude wins by a large margin.
- On `"architecture explanation"`, Gemini is often clearer.
- On `"current-state understanding"`, Claude appears more reliable.
- On `"decision-gate rigor"`, Claude is far ahead despite its own exceptions.

### 5.2 Where Claude Learns From Gemini

- Gemini's early architecture explanation is easier to read.
- Gemini's glossary and mathematical sections could help orient a first-time contributor.
- Claude could borrow a shorter explanatory appendix for onboarding.
- Gemini's overview is more concise about the macro performance story.
- Claude could compress some of its narrative sections without losing rigor.

### 5.3 Where Gemini Must Learn From Claude

- Gemini needs Claude's P0 baseline discipline.
- Gemini needs Claude's file-scoped implementation detail.
- Gemini needs Claude's concrete revert conditions.
- Gemini needs Claude's risk register.
- Gemini needs Claude's artifact naming discipline.
- Gemini needs Claude's full M-sweep and shape-sweep structure.
- Gemini needs Claude's explicit per-shape handling of the 3-stage pipeline.
- Gemini needs Claude's stronger SplitK correctness handling.

## 6. Final Recommendation

- Adopt Claude as the base.
- Do not adopt it verbatim.
- Before execution, fix the places where Claude weakens the very discipline it otherwise champions.
- Specifically, remove all waivers of isolated pre-integration correctness tests.
- Restore the 12-shape minimum for every major change, including the late micro-optimization phase, unless the phase is explicitly reclassified as non-major and approved as such.
- Delete the unsafe SplitK finalization shortcut.
- Add repeated-run guidance for small expected wins and small regression thresholds.
- Add threshold-adjacent dispatcher tests for small-M behavior.
- If those changes are made, the resulting plan would satisfy the user's demanded bar much better than the Gemini draft.
- If those changes are not made, the sprint plan will still carry hidden "skip by exception" behavior.

## 7. Bottom Line

- Claude draft: strong, close, but not fully faithful to `"no skipping"` until the exceptions are removed.
- Gemini draft: useful background note, not an acceptable execution plan for this sprint as written.
- Against the explicit bar of `"methodical no-skip execution with grid-search benchmarking and correctness gates"`, Claude is a conditional pass after edits.
- Gemini is a fail as written.

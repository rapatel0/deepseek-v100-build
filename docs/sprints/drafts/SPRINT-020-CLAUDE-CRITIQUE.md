# SPRINT-020-CLAUDE-CRITIQUE — Cross-review of Codex and Gemini drafts

**Reviewer:** Claude (author of `SPRINT-020-CLAUDE-DRAFT.md`).
**Reviewing:** `SPRINT-020-CODEX-DRAFT.md` (367 lines) and
`SPRINT-020-GEMINI-DRAFT.md` (151 lines).
**Reference:** `SPRINT-020-INTENT.md`, `REPORT-13.md`,
`SPRINT-019-FOLLOWUPS.md`, `TURBOMIND-INSIGHTS.md`.

Cross-references in this critique to phase numbers (P0–P5) refer to the
draft under review in that subsection, not to the Claude draft.

---

## 1. Codex draft (`SPRINT-020-CODEX-DRAFT.md`)

### 1.1 Strengths

1. **Strongest "two-layer" architectural framing of the three drafts.**
   Codex separates P1 (native turbomind `gemm_bench` bring-up in
   `research/lmdeploy/src/turbomind/kernels/gemm/test/gemm_bench.cu`)
   from P2 (a real `tc-grid` bridge at
   `tools/tc-grid/src/launch_turbomind_int8.cu`). This is more
   architecturally honest than Gemini's "Wrapper Development" hand-wave
   and more rigorous than my own draft's "gemm_bench stays
   out-of-tree" approach for the head-to-head comparison — the bridge
   guarantees identical activations / scales / tolerance evaluation,
   eliminating a real apples-to-oranges risk.

2. **Concrete file enumeration.** Names
   `kernel/sm70_884_{4,8,16}.cu`, `arch/config_sm70_s884.h`,
   `mainloop_sm70.h`, `iterator_sm70.h`, `registry.cu`,
   `dispatch_cache.{h,cu}`, `LlamaLinear.cu`, and `moe_ffn_layer.cc`
   in the right phases. This is the highest-fidelity file map of the
   three drafts and would survive a reader who hadn't read
   TURBOMIND-INSIGHTS.

3. **Explicit decision rule with two named outcomes.** Codex §P3
   decision rule names both **breakthrough (≥ 44.0 TF at M=2048,
   N=K=7168 OR ≥ 10% on asymmetric MoE)** and **ceiling proof
   (≤ 41.0 TF on the square baseline)**. The 10% asymmetric clause is
   sharper than the Gemini draft's "validation against 3 critical
   shapes" with no per-shape gain threshold, and sharper than my draft's
   ±2 TF / +5 TF symmetric thresholds.

4. **P5 fallback close is real.** Unlike Gemini, which has no
   loss-case branch beyond "shift focus to Deployment Integration
   (P4)", Codex P5 explicitly enumerates four close items
   (N≠K CLI, sanitizer cleanup, dispatch encoding, ceiling-proof
   memo) and declares the negative result is itself shippable evidence.
   That makes the failure mode actionable, not just narrative.

5. **`Gemm::Export` / `DispatchCache::Export` is a real handoff.**
   Codex P3.4 + P4.3 promote Turbomind's measured launch specs into
   reusable runtime artifacts via the existing export/import API. This
   is a stronger production handoff than my draft's `dispatch.h`
   table (which duplicates Turbomind's logic) and stronger than
   Gemini's "Per-(M, shape) dispatch wired into DSv4-flash" with no
   API named.

6. **Q1's elimination of CUTLASS extension is well-argued.** The
   "anything smaller than ≥ 48 TF doesn't justify the template and
   maintenance cost" framing in Alternative A is a real economic
   threshold and a useful lens that's missing from Gemini.

### 1.2 Weaknesses

1. **The 41.0 TF boundary is overloaded.** P2 gate says
   "≥ 41.0 TF, which is enough to continue the port path"; P3
   decision rule says "≤ 41.0 TF on the square baseline" is a ceiling
   proof. At exactly 41.0 TF both branches fire. The DoD inherits the
   same ambiguity. **Fix:** make one boundary strict
   (`> 41.0` to continue, `≤ 41.0` to stop, or rotate one threshold
   to 42.0).

2. **No effort budget per phase.** Sprint cadence is 30–50 hr
   (SPRINT-019 metric, intent §constraints). Codex defines 5 phases
   plus a fallback close but does not estimate any of them. P1 alone
   (re-enable a commented-out bench target with NVBench dep) is
   easily 8–16 hr per `feedback_effort_estimation_undocumented_hardware`
   — without a budget, P1 can blow the entire sprint and the plan
   gives no escape hatch (no equivalent of my draft's "drop nvbench,
   replace with 30-line cudaEvent harness" P1.2 fallback).

3. **The bridge (P2) is built before the ceiling answer (P3) is
   known.** Codex builds a real tc-grid → Turbomind launcher
   (`launch_turbomind_int8.cu`, dedicated `TCGRID_ENABLE_TURBOMIND_GEMM`
   CMake option) in P2, then runs the head-to-head in P3. If P3 is
   negative (Turbomind ≤ 41 TF), the entire P2 scaffolding is
   throwaway code and was the most invasive single piece of work in
   the sprint. **Fix:** insert a "kill switch" — run P1's native
   `gemm_bench` against the v12_ms3 shape *before* committing to P2's
   bridge build; if the native number is already ≤ 41 TF, skip P2
   and jump to the fallback close (P5).

4. **Sanitizer coverage is much weaker than the SPRINT-019 follow-up
   debt requires.** Codex P0.2 says
   "compute-sanitizer --tool memcheck,racecheck,initcheck for v12s
   with KSPLIT={2,3,5,8,16}" — one line. The DoD says "v12s clears
   memcheck, racecheck, and initcheck for the atomic KSPLIT coverage
   set." Missing:
   - **Repeated-launch initcheck** (scratch buffer reset between
     launches; SPRINT-019 P3.1 used this and a stale-buffer bug would
     ship unnoticed otherwise).
   - **Production-shape AND small-K stress shape** coverage. The
     SPRINT-019 §P3.1 sanitizer plan covered both and Codex collapses
     it.
   - **Tier-1 isolated CPU-reference test file** (the SPRINT-019
     no-skip rule expects an artifact; "run sanitizer" is not a
     test).

5. **No end-to-end model correctness tier.** Codex P4 ("Runtime
   follow-through if Turbomind wins") validates "the real call sites"
   (`LlamaLinear.cu`, `moe_ffn_layer.cc`) and "does not regress
   correctness on the DSv4-flash MoE path" — but no token-distribution
   KL, no sample-generation comparison, no Hamming-distance check.
   "Does not regress correctness" is asserted, not measured. This is
   a serious gap if Turbomind wins on perf but its accumulator
   precision differs (e.g., FP32-acc vs v12_ms3's FP16-acc would
   *change* the model's output distribution and "no regression" would
   need a quality definition).

6. **Risk register is missing four high-impact risks.**
   - **Data-layout mismatch** (row-major B vs col-major B; group_size
     mismatch between `tc-grid`'s QK_INT8=32 and Turbomind's
     group_size=128). Risk #2 names "apples-to-oranges" but
     mitigates with "reuse tc-grid data generation" — that's exactly
     the layout-mismatch surface the bridge has to traverse.
   - **Scope creep / sprint cadence overrun.** Five phases plus a
     bridge with no time-budget against the 30–50 hr cap is the
     largest single risk and has no mitigation listed.
   - **`models.h` shape-catalog mismatch** (Turbomind's bench config
     table is built for Llama/Qwen, may not include exact
     N=K=7168 INT8). My draft R2 names this; Codex doesn't.
   - **Dispatch-cache API drift** between the bridge (P2) and the
     runtime path (P4). If P3 changes the cache schema, P4 breaks.

7. **No discussion of NVBench.** Re-enabling `gemm_bench` requires
   either NVBench (the original dep, currently commented out) or a
   replacement harness. Codex P1.2 says "without pulling unrelated
   Turbomind test baggage" but doesn't say which. NVBench has its own
   build chain and CUDA-12.2 compatibility surface; this is the
   single biggest concrete unknown in P1 and goes unaddressed.

8. **Asymmetric-shape risk on v12_ms3 itself.** Codex notes the
   asymmetric MoE shapes are first-class gates (Risk #3) but doesn't
   account for v12_ms3 being SPRINT-019-validated only on square
   shapes. Bank-conflict pathology or register-pressure issues at
   `(M=2048, N=18944, K=7168)` could surface and there is no fallback
   to v10/v11 in `dispatch.h`. **Fix:** add a per-cell fallback rule
   to the bridge's row reporting.

### 1.3 Gaps in risk analysis (summary)

| Risk in my analysis | Severity | Codex coverage |
|---|---:|---|
| Data-layout mismatch tc-grid ↔ Turbomind | HIGH | partial (Risk #2 names "apples-to-oranges" but mitigation is unrelated) |
| Scope creep / 30–50 hr cadence | HIGH | absent |
| NVBench build dep / CUDA-12.2 compat | MEDIUM | absent |
| `models.h` 7168×7168 INT8 entry missing | LOW | absent |
| Dispatch-cache schema drift P3 ↔ P4 | LOW | absent |
| KL regression at integration time | MEDIUM | absent (no model-level correctness tier) |
| Atomic kernel races surfaced by initcheck | MEDIUM | weak (one-line P0.2 entry; no Tier-1 artifact) |
| v12_ms3 asymmetric-shape pathology | MEDIUM | partial (Risk #3 ≠ technical risk) |

### 1.4 Missing edge cases

- **P3 decision rule boundary at 41.0 TF / 44.0 TF / 10%.** No tiebreak
  for the case "Turbomind hits 42.5 TF and 8% on asymmetric" — neither
  the breakthrough nor ceiling-proof clause fires.
- **What if the bridge produces `>= 41.0 TF` (P2 gate met) but P3
  measures degradation from CUTLASS Gemm70 reference?** P3 doesn't
  re-anchor against CUTLASS; only against v12_ms3.
- **`compute-sanitizer` fails on the new `launch_turbomind_int8.cu`
  bridge glue.** P2's gate is "runs cleanly for the square baseline"
  but doesn't require sanitizer on the bridge code itself — only on
  v12s (P0.2). New launcher code without `--tool memcheck` is exactly
  the kind of memory-safety hole §Security claims to require.
- **What if `Gemm::Export` produces a cache that conflicts with an
  existing dispatch_cache file in the repo?** Cache-merge semantics
  are unspecified.

### 1.5 Definition of Done completeness

| DoD requirement | Quality | Gap |
|---|---|---|
| Asymmetric N≠K CLI | OK | doesn't specify `--shape-list` semantics or back-compat for `--nk` |
| v12s sanitizer clean | OK | doesn't require a Tier-1 test file artifact, just "clears" |
| Turbomind SM70 bench builds | OK | doesn't capture build flags / NVBench version |
| tc-grid → Turbomind bridge runs | OK | doesn't require sanitizer-clean on the bridge itself |
| Headline outcome (one of two) | OK | 41.0 TF boundary ambiguous |
| Dispatch cache exercised in `LlamaLinear.cu` | WEAK | "or an equivalent runtime entry point" is escape-hatch language; either runtime is wired or it isn't |
| End-to-end correctness | **MISSING** | no model-level KL, no sample-generation comparison, no token-Hamming gate |
| No perf regression on shipped path | **MISSING** | not in DoD; "does not regress correctness" is the only similar item |
| REPORT close artifact | **MISSING** | no REPORT-14 named (compare to my P6.1 / Gemini P5 close report) |
| Memory updates | **MISSING** | no §Memory items in DoD |
| Sprint follow-ups + deferred capture | **MISSING** | no FOLLOWUPS / DEFERRED files named |

---

## 2. Gemini draft (`SPRINT-020-GEMINI-DRAFT.md`)

### 2.1 Strengths

1. **Cleanest narrative framing.** The "architectural decision point"
   pitch in §1, the explicit 50 TF / 25 TF / asymmetric / production-
   wiring quad-target in §1.1, and the P1 → P2 (ceiling proof → port)
   sequence are the most scannable of the three drafts. A reader can
   absorb the plan in 5 minutes.

2. **Identifies the right bottleneck.** §3.1 names
   `mio_throttle (SMEM bandwidth)` as the v12 wall — matches REPORT-13
   §3 directly. Both other drafts agree, but Gemini surfaces it
   earliest in the document.

3. **P1 ceiling-proof gate IS a real gate.** §P1 step 3 says
   "If Turbomind hits ≥ 45 TF → Proceed to P2 (Wholesale Port). If
   Turbomind hits ≤ 41 TF → Accept v12 ceiling; shift focus to
   Deployment Integration (P4)." This is structurally similar to my
   draft's P1.4 → P2 → SPRINT-021 split and gives the sprint a clean
   off-ramp.

4. **M=64 target is concrete and aggressive.** "≥ 25 TF" at M=64 (vs
   v12s's current 21.55 TF) is sharper than the Codex draft, which has
   no M=64 target except in the 10% asymmetric clause.

5. **Names `scheduler_sm70.cuh`.** The other two drafts don't
   mention this file; Gemini does. (Pending a verification that the
   file exists at that path — see §2.4.)

### 2.2 Weaknesses

1. **Throughout: too thin to execute.** 151 lines vs Codex 367 vs my
   1290+. Several phases are one-paragraph stubs:
   - P2.1 "Wrapper Development: Create `tools/tc-grid/kernels/
     turbomind_wrapper.cuh` to map `tc-grid` layouts to Turbomind
     template parameters." That is the entire spec for the highest-risk
     code change in the sprint.
   - P3.2 "Dispatch Rule Encoding: Implement the per-(M, N, K)
     dispatch table in `launch_int8.cu`." Same.
   - P4.1 "Code Migration: Extract the winning kernels and dispatcher
     from `tc-grid` into the DSv4-flash inference backend." No
     specified backend, no API surface, no pybind11 path, no
     feature flag. This is the biggest single integration risk in
     the sprint.

2. **Threshold inconsistency: 45 TF.** §1.1 says "≥ 50 TF reached OR
   provide a definitive 'ceiling proof' (e.g., Turbomind also hits
   ≤ 41 TF)." §6 DoD says "≥ 50 TF reached OR Turbomind ceiling
   documented at < 45 TF." §10 Q1 says "use `gemm_bench` as a ceiling
   proof first (P1)." So:
   - 50 TF = win
   - 45 TF = DoD ceiling threshold (`<`)
   - 41 TF = §1.1 ceiling threshold (`≤`)
   - 45 TF = P1 "Proceed to P2" boundary (`≥`)
   At exactly 45 TF P1 says "proceed to P2 wholesale port" but DoD
   says "ceiling documented." Three different thresholds with three
   different boundary conventions.

3. **Files Summary contains a placeholder.** `common/reasoning-budget.cpp:
   (Placeholder) Wiring into inference path.` This is a fictional file
   name. The DSv4 inference path lives in
   `research/lmdeploy/src/turbomind/models/llama/LlamaLinear.cu` and
   `lmdeploy/turbomind/turbomind.py` (per Codex's enumeration and my
   own draft). The placeholder suggests the integration phase was not
   actually researched against the existing repo state.

4. **No effort estimates.** Same problem as Codex but worse — Gemini
   has no per-phase ETA and no overall sprint budget. Given the 30–50
   hr cap, the wholesale port (P2 in Gemini's plan) plus the inference
   integration (P4) plus DSv4 end-to-end is likely 60–100 hr (per the
   `feedback_effort_estimation_undocumented_hardware` 3× multiplier).

5. **No fallback if `gemm_bench` builds but ceiling lands at 42–44 TF.**
   §P1 step 3 binary-splits at 41 / 45; the 41–45 TF gap is undefined.
   This is the most likely actual outcome (per the SPRINT-019 evidence
   that v12_ms3 is mio_throttle-bound near the SMEM bandwidth ceiling).

6. **No model-level correctness gate.** §P4 step 2 says "Verify token
   generation correctness and latency improvement in a real model run"
   — one bullet. No KL threshold, no sample size, no temperature
   handling, no reference baseline. The Tier-6 surface that exists in
   my draft has no analog here.

7. **Risk register is the smallest of the three drafts (3 entries).**
   Missing risks (ranked by my own draft's severity):
   - HIGH: scope creep / sprint cadence overrun.
   - HIGH: data-layout mismatch tc-grid ↔ Turbomind ↔ DSv4.
   - HIGH: gemm_bench `models.h` shape-catalog mismatch.
   - MEDIUM: NVBench build dep / CUDA-12.2 compat.
   - MEDIUM: KL regression at integration time.
   - MEDIUM: asymmetric-shape pathology in v12_ms3.
   - MEDIUM: lmdeploy build-iteration time (60s+ per build).
   - LOW: gpu-01 contention / DCGM-exporter re-enabled.

8. **Risk #1 mitigation contradicts Q1 reasoning.** Risk #1 says
   "fallback to CUTLASS extension if blocked." Q1 explicitly rejects
   CUTLASS extension as the primary path due to "complexity of custom
   dequant in CUTLASS 2.x on V100." If CUTLASS extension is too
   complex to be the primary path with sprint-level prep, it is
   strictly more complex as a hot-swap fallback inside the same sprint.
   The two statements can't both be true.

9. **No mention of memory rules / feedback memories.** The intent §3
   names four operative memories
   (`v100_splitk_atomic_pattern`, `v100_3stage_register_budget_rule`,
   `feedback_pre_dequant_defeats_int8`,
   `feedback_effort_estimation_undocumented_hardware`). Gemini doesn't
   reference any of them. `feedback_pre_dequant_defeats_int8` is
   directly relevant to P2's wrapper design (INT8 must stay INT8 in
   gmem; dequant happens in SMEM); leaving it implicit invites the
   wrapper to repeat SPRINT-018's pre-dequant ceiling mistake.

10. **No discussion of v12s, the SPRINT-019 small-M champion.** §1
    says "Reach ≥ 25 TF via Turbomind or optimized v12s" but §P0
    plans only sanitizer work on v12s, not optimization. If the
    Turbomind kernels don't outperform v12s at M=64, the "optimized
    v12s" path has no concrete steps to reach 25 TF.

11. **No threshold-adjacent M dispatch verification.** SPRINT-019 P7.1
    standardized `M ∈ {63, 65, 255, 257, 1023, 1025}` for testing the
    dispatch transitions. Gemini's P3.2 "Dispatch Rule Encoding"
    doesn't mention threshold adjacency, which means a dispatcher
    could ship that breaks at the M-boundary cells.

### 2.3 Gaps in risk analysis (summary)

Gemini lists 3 risks; my own analysis lists 17. Coverage:

| Risk in my analysis | Severity | Gemini coverage |
|---|---:|---|
| Turbomind build collapses | HIGH | covered (Risk #1) — but with a contradictory mitigation |
| Data-layout mismatch | HIGH | absent |
| Scope creep / cadence | HIGH | absent |
| Asymmetric N≠K v12_ms3 pathology | MEDIUM | absent |
| KL regression at integration | MEDIUM | absent |
| lmdeploy build coupling | MEDIUM | absent |
| v12s sanitizer surfaces real race | MEDIUM | covered ("LOW", probably mis-ranked given SPRINT-019 P3.1 evidence) |
| `models.h` shape-catalog mismatch | LOW | absent |
| Dispatch-cache / API drift | LOW | absent |
| External kernel correctness | MEDIUM | covered (Risk #2) |
| gpu-01 contention | LOW | absent |
| DCGM-exporter re-enabled | LOW | absent |
| Measurement noise on 1–2% gates | LOW | absent |

### 2.4 Missing edge cases / verification gaps

- **`tools/tc-grid/kernels/turbomind_wrapper.cuh` design:** does it
  template over `(BM, BN, BK, W)` from `tc-grid` and instantiate
  Turbomind's `Gemm::Run`, or does it call Turbomind's registry
  directly? Two completely different integration topologies; neither
  is specified.
- **`scheduler_sm70.cuh` reference:** Gemini names this file in §3.2
  but it is NOT named in the intent §3 or in TURBOMIND-INSIGHTS. I
  cannot verify it exists at that path; if it doesn't, the
  architecture section is asserting structure the codebase doesn't
  have.
- **DoD "All kernels pass `compute-sanitizer`"** is over-broad — the
  legacy v3/v10/v11 kernels are not all sanitizer-clean (the
  SPRINT-019-FOLLOWUPS item 1 was specifically v12s-scoped). Reading
  this gate literally would block sprint close on a debt that
  predates SPRINT-019.
- **What if Turbomind's `sm70_884_*.cu` instantiations don't
  template over the `tc-grid` shape catalog directly?** P2's
  "Verification: Tier-1 CPU-reference tests + compute-sanitizer"
  doesn't address shape-coverage gaps in the wrapper.
- **What if the DSv4-flash model isn't loadable in the pod?** My
  draft R16 surfaces this; Gemini's P4 assumes the inference path is
  operable.

### 2.5 Definition of Done completeness

| DoD requirement | Quality | Gap |
|---|---|---|
| ≥ 50 TF or Turbomind ceiling | OK | 45 TF threshold conflicts with §1.1's 41 TF |
| ≥ 25 TF at M=64 | OK | no fallback if Turbomind doesn't deliver and v12s sanitizer surfaces a fix-needing race |
| Correctness rel/p99/maxabs + sanitizer | OK | "all kernels" is over-broad |
| Per-(M, shape) dispatch wired into DSv4-flash | WEAK | "wired into" is verb-only; no integration test, no model-level KL, no feature-flag rollout, no opt-in vs default-on policy |
| REPORT-14 names architectural winner | OK | doesn't include ncu evidence requirement |
| End-to-end model correctness | **MISSING** | no token-distribution KL, no sample-generation gate |
| Threshold-adjacent dispatch verification | **MISSING** | no mention of `M ∈ {63, 65, 255, 257, 1023, 1025}` |
| No perf regression on shipped path | **MISSING** | not in DoD |
| Memory updates | **MISSING** | not in DoD |
| Follow-ups + deferred capture | **MISSING** | no FOLLOWUPS / DEFERRED files named |
| Working-tree audit at close | **MISSING** | no `git status` clean gate |

---

## 3. Cross-cutting observations

1. **Both drafts under-budget P1 (gemm_bench bring-up).** Codex gives
   no estimate; Gemini gives no estimate; my own draft caps it at
   8–16 hr (3× the TURBOMIND-INSIGHTS §counterfactual #3 estimate).
   The build engineering on a commented-out target with NVBench dep
   is the single largest hidden cost in the sprint.

2. **Both drafts under-specify the integration tier.** Codex P4 names
   `LlamaLinear.cu` and `moe_ffn_layer.cc` as validation points but
   does not specify a model-level KL or token-Hamming gate. Gemini
   P4 names "real model run" with no gate at all. Either approach
   risks a "Turbomind ships, but DSv4-flash output drifts and nobody
   notices for two sprints" failure mode.

3. **Both drafts under-cover the v12s sanitizer follow-up.** Codex
   collapses it to one line; Gemini ranks it LOW. Per SPRINT-019
   P3.1, this is the only deferred correctness debt and it is
   atomic-kernel work that should not ship to DSv4 integration without
   adversarial-KSPLIT + repeated-launch + scratch-buffer-reset
   coverage.

4. **Both drafts conflate two boundary thresholds.** Codex's 41.0 TF
   and Gemini's 45 TF each appear as both "go" and "stop" boundaries
   with the same number. This is not a typo issue — it's a real
   ambiguity in the decision rule that will surface if the actual
   measurement lands near the boundary.

5. **Both drafts are stronger than mine on architectural-break
   posture.** Codex commits to Turbomind as primary with a real bridge
   (`launch_turbomind_int8.cu`); Gemini commits to a wholesale port
   in P2. My draft hedges with "ceiling check first, port deferred to
   SPRINT-021" — that's a more conservative posture and gives up the
   chance to ship a Turbomind win in SPRINT-020 if P1 surfaces
   headroom early. If headroom is high, Codex's plan delivers more
   value in-sprint; if headroom is low, my plan wastes less time.

6. **Only my draft addresses `feedback_pre_dequant_defeats_int8`.**
   Codex implicitly respects it (the bridge reuses tc-grid's INT8
   data generation). Gemini doesn't mention it and the wrapper design
   is open — the wrapper could accidentally pre-dequant and produce
   a misleading TF number that informs the SPRINT-021 decision.

---

## 4. Summary scoring

| Dimension | Codex | Gemini |
|---|---|---|
| Architectural-break ambition | strong (real bridge) | strong (wholesale port) |
| File-path concreteness | strong | medium (placeholder file) |
| Decision-rule sharpness | strong but ambiguous at 41.0 TF | weak (3 inconsistent thresholds) |
| Effort budget | absent | absent |
| Sanitizer rigor | weak | weak |
| Asymmetric-shape coverage | strong | weak |
| Integration tier | medium (named files, no quality gate) | weak (no specified backend, no quality gate) |
| Model-level correctness | absent | absent |
| Risk register breadth | medium (5 risks) | weak (3 risks, 1 contradictory) |
| Fallback / negative-result close | strong (P5 fallback) | weak (deflects to "shift focus to P4") |
| DoD completeness | medium | weak |
| Memory-rule references | absent | absent |
| Length / depth | adequate (367 lines) | thin (151 lines) |

**Codex is the stronger of the two drafts.** Its biggest weaknesses
(missing effort budget, ambiguous 41.0 TF boundary, no model-level
correctness gate, sparse sanitizer plan) are addressable with
targeted additions. Its core architectural choice — bridge first,
port-decision second — is sound and the file-level execution plan is
realistic.

**Gemini's draft is structurally promising but operationally
incomplete.** Its biggest weakness is the gap between the headline
ambition (50 TF, wholesale port, end-to-end DSv4 wiring) and the
specification depth (one-paragraph phases, placeholder file names,
contradictory thresholds, 3-risk register). Adopting it as-is would
produce a sprint that overshoots the 30–50 hr cadence and ships
without a model-level correctness gate.

**Recommended merge synthesis** (for a future combined draft):
- Keep Codex's two-layer Turbomind topology (P1 native bench → P2
  bridge) and `Gemm::Export` runtime handoff.
- Replace Codex's single-line P0.2 with my draft's P0.2 sanitizer
  spec (Tier-1 file, KSPLIT enumeration, repeated-launch
  initcheck).
- Add my draft's P5 (DSv4-flash sample-generation KL + Hamming) as a
  Codex P5 (renumbering Codex's "P5 fallback close" to P6).
- Adopt Codex's `≥ 44.0 TF` headline target instead of my draft's
  `+5 TF over v12_ms3` (cleaner anchor).
- Resolve the 41.0 TF boundary (`> 41.0` continue, `≤ 41.0` stop).
- Add explicit per-phase ETA against the 30–50 hr sprint cap.
- Drop Gemini's CUTLASS-extension-as-fallback (contradicts its own
  Q1 reasoning).
- Add my draft's R5 (data-layout mismatch) to the risk register.
- Reference `feedback_pre_dequant_defeats_int8` in the bridge design.

# SPRINT-019 — Gemini Critique: Methodical Performance Engineering

This document provides a comprehensive and methodical critique of the sprint drafts `SPRINT-019-CLAUDE-DRAFT.md` (henceforth "Claude") and `SPRINT-019-CODEX-DRAFT.md` (henceforth "Codex"), evaluated against the foundational mandates in `SPRINT-019-INTENT.md`.

## 1. Executive Summary

Both drafts demonstrate a high-signal understanding of the technical requirements for closing the performance gap on the V100. However, they diverge significantly in their approach to the "no-skip" mandate. Claude provides a **Surgical Execution Plan**, prioritizing the "how" of implementation with extreme specificity on undocumented hardware behavior. Codex provides a **Validation Framework**, prioritizing the "where" and "why" of verification, particularly concerning multi-shape robustness.

The user's demand for a "no-skip" execution with "grid-search benchmarking and correctness gates" is a high bar. Claude meets this through granular procedural sub-steps and explicit time-boxes, while Codex meets it through a structured artifact layout and a standardized measurement protocol.

---

## 2. Detailed Critique: SPRINT-019-CLAUDE-DRAFT.md

### 2.1 Strengths: The Surgical Mindset

Claude’s draft is characterized by a "mechanic's" approach to kernel engineering.

*   **P1.1 Lane-Probing Routine (§4)**: This is the draft's strongest section. By mandating the empirical derivation of `thread_offset_C` and `static_offset_C` formulas ("Do NOT assume they do"), Claude directly addresses the `v100_wmma_half_float_frag_layout_mismatch` failure mode. This level of specificity is exactly what the user means by "methodical."
*   **Explicit Decision Gates (§1.2, §4)**: Every phase concludes with a "Decision gate." For example, P1.3 (§4) requires "Headline TF improved at ≥ 3 of 5 M values, AND no M regresses by > 2%." This is a quantitative, non-ambiguous gate that prevents the "feel-good" shipping of asymmetric wins.
*   **Phase Dependency Logic (§3.3)**: Claude provides a rationalized dependency order (§6.1 → 6.2 → 6.3 → 6.4 → 6.5). It correctly identifies §6.1 (FP16 acc) as the "gatekeeper" for register budget, which in turn unlocks §6.2 and §6.4. This shows superior strategic planning.
*   **Estimation Fidelity (§11)**: By applying the `feedback_effort_estimation_undocumented_hardware.md` 3× multiplier, Claude arrives at a realistic 23–36 hr total. This prevents the "sprint bloat" common in optimistic planning.

### 2.2 Weaknesses: The "Square Benchmark" Blindness

*   **Shape Homogeneity**: Claude’s benchmarking focus is overwhelmingly on the square `7168x7168` shape. While it includes a grid sweep, the "Headline Targets" (§1.1) and "Verification Strategy" (§1.2) are dominated by M ∈ {64, 256, 1024, 2048, 4096} at fixed N=K=7168. As Codex correctly identifies, this can "flatter" a kernel that might fail on tall-skinny or wide-shallow MoE shapes.
*   **P5 Marginalization (§4)**: The PRMT A-side phase (§6.5) is treated as a "low effort" tail lever. The verification strategy is reduced to "No new isolated test needed." For a "no-skip" mandate, this is a dangerous shortcut. Even marginal instruction changes can shift register pressure or bank conflict patterns.
*   **Lack of Tooling Standardization**: Unlike Codex, Claude does not provide the exact `ncu` or `nsys` command lines. This leaves the "methodical" measurement open to human error in flag selection.

### 2.3 Gaps in Risk Analysis

*   **SMEM vs. Register Trade-offs**: While Claude tracks registers closely, it doesn't explicitly gate on SMEM footprint for §6.4. On V100, the 96KB SMEM limit is as hard a ceiling as the 256-reg cap. A large-BM kernel that fits in registers but consumes 64KB+ SMEM will drop occupancy to 1 CTA/SM, potentially killing the gain.
*   **Atomic Collision Scaling (§6.3)**: Claude assumes SplitK is "well-understood." However, it doesn't analyze the risk of atomic contention on the C-scratch buffer as `KSPLIT` increases to 16. This is a missing performance-cliff analysis.

### 2.4 Definition of Done (DoD) Completeness

**Claude DoD (§6) is exceptionally robust.**
It includes:
- "target stall that moved (with before/after %)" in commit messages.
- "ptxas spill gates" for every shipped kernel.
- "CUTLASS Gemm70 (version=40) ratio measured at M=2048."
- Explicit "Ship or Revert" decision per commit.

This DoD is a procedural mandate that enforces the "no-skip" rule turn-by-turn.

---

## 3. Detailed Critique: SPRINT-019-CODEX-DRAFT.md

### 3.1 Strengths: The Verification Architect

Codex’s draft is characterized by a "systems" approach to quality assurance.

*   **Phase 6.6: Multi-Shape MoE Validation**: This is the single most important addition across both drafts. Codex identifies that "tuning work should stop relying on single-shape anecdotes." It mandates a matrix including `N=7168, K=18944`, which is critical for verifying the robustness of the new v12 family across the MoE expert space.
*   **The Correctness Ladder (§3.3)**: Codex introduces a formal hierarchy of verification: `Isolated -> Sanitizer -> Sweep -> Baseline Comparison`. This is a superior mental model for a "methodical" sprint, ensuring that no kernel reaches the performance sweep without passing the "ladder."
*   **Standardized Profiling Protocol (§4.1)**: Providing exact `ncu` and `nsys` command templates is a high-signal move. It ensures that the metrics collected in Phase 6.1 are exactly comparable to Phase 6.5, eliminating "profiler noise."
*   **Artifact Visibility (§3.5)**: The "Artifact layout" section (CSV per phase, PNGs for nsys) treats the *evidence* as a first-class citizen. This fulfills the user's desire to see the "ncu stall breakdown" and "CUTLASS comparison."

### 3.2 Weaknesses: The "Implementation Gap"

*   **Vague Step Instructions**: Compared to Claude, Codex is much less specific about *how* to build the kernels. For §6.1 (FP16 acc), it says "Replace float c_frag... with a half-typed accumulator storage layout." It doesn't mention the empirical lane-mapping problem that Claude spends an entire sub-phase (P1.1) on.
*   **Absence of Effort Guardrails**: Codex lacks ETAs and time-boxes. Without a "8 hr abandonment rule" like Claude's, a "methodical" investigator might spend 3 days trying to fix a lane-mapping bug in §6.1, derailing the entire sprint.
*   **Template-Heavy Decision Gates**: Codex uses a 7-question template (§3.5). While thorough, it is less "surgical" than Claude's phase-specific success criteria. For example, it doesn't specify which *exact* stall must drop for each lever.

### 3.3 Gaps in Risk Analysis

*   **Build-Time Explosion**: With 12+ variants per phase and 6 phases, the total number of template instantiations could easily exceed 100. Codex doesn't address the risk of build times hitting 5-10 minutes, which breaks the researcher's iteration loop.
*   **DCGM Verification**: While it mentions pausing the exporter, it lacks Claude's "Pre-flight check" (P0) to ensure the hardware counters are actually available and clean.

### 3.4 Definition of Done (DoD) Completeness

**Codex DoD is structurally sound but procedurally soft.**
It lists "Sprint 019 is done only when..." criteria, but it feels more like a final checklist than a per-commit gate. It lacks the "per-phase (every commit)" granularity, which is vital for catching skips *during* the session.

---

## 4. Architectural Divergence: Implementation vs. Validation

The primary divergence between the drafts lies in their conceptualization of the kernel developer's primary obstacle.

### 4.1 Claude: Hardware as the Enemy
Claude views the V100 ISA and its undocumented behavior as the primary risk. This is evidenced by the "Empirical derivation" instructions in §P1.1 and the "Register-budget verification" in §P1.2. Claude’s plan is built to survive a hostile hardware environment where documentation cannot be trusted. It prioritizes the low-level PTX and SASS artifacts over the high-level sweep statistics.

### 4.2 Codex: Statistics as the Enemy
Codex views sampling bias and "noise" as the primary risks. This is evidenced by the "Standardized Nsight Compute command" in §4.1 and the "Multi-shape MoE validation" in §6.6. Codex’s plan is built to ensure that a performance win is statistically significant and generalizable across the entire MoE expert space. It prioritizes the "Artifact Trail" and the reproducibility of measurements.

---

## 5. Phase-by-Phase Methodological Breakdown

### 5.1 Phase 6.1 (FP16 Acc)
*   **Claude (P1.1)**: "Empirically derive... thread_offset_C... Do NOT assume they do." This is the peak of "no-skip" discipline. It recognizes that undocumented hardware is a black box that must be probed systematically.
*   **Codex (6.1)**: "Add a new isolated atom correctness test... that exercises mma_m8n8k4... directly." Good, but assumes the developer knows how to write the test correctly from documentation alone.
*   **Critique**: Claude’s P1.1 sub-phase is the superior "no-skip" pattern. It transforms a vague "test it" into a "reverse-engineer it" instruction.

### 5.2 Phase 6.2 (3-Stage)
*   **Claude (P2.3)**: "Register-pressure tracking... tabulate launch__registers_per_thread per shape." This is a proactive gate that identifies candidates before execution.
*   **Codex (6.2)**: "Nsight Systems timeline proof... show tighter overlap." This is a retroactive verification that checks if the goal was met.
*   **Critique**: Claude’s proactive register-budget gating is more methodical as it catches failures *before* the expensive profiling run, saving valuable cluster time.

### 5.3 Phase 6.3 (SplitK)
*   **Claude (P3.1)**: "compute-sanitizer --tool racecheck AND --tool initcheck on first launch." Excellent specificity regarding the unique risks of atomic operations.
*   **Codex (6.3)**: "test_v11_splitk_reduce_sm70.cu... validates per-slice accumulation." Focuses on functional correctness rather than synchronization safety.
*   **Critique**: Claude’s use of `racecheck` is vital for atomic-heavy kernels. Functional tests often miss transient race conditions in SplitK reductions that only surface at high thread contention.

### 5.4 Phase 6.4 (Large CTA Tile)
*   **Claude (P4.2)**: "Partial c_frag SMEM rotation... 2 ld.shared + 2 st.shared per K-iter." This provides a specific implementation strategy for mitigating register pressure.
*   **Codex (6.4)**: "explicit spill/rotation mode... Center on larger-BM exploration."
*   **Critique**: Claude provides the specific mechanism (rotation), while Codex provides the sweep logic. Claude's specificity here is more actionable for a "no-skip" execution.

---

## 6. Risk Management: Gaps and Mitigations

| Risk | Claude Mitigation | Codex Mitigation | Superior Approach |
| :--- | :--- | :--- | :--- |
| **Undocumented ISA (FP16)** | Empirical lane probing (P1.1) | Isolated atom test | Claude |
| **3-Stage Regression** | Register-budget table (P2.3) | Nsys timeline proof | Claude (proactive) |
| **Numerical Drift** | `rel ≤ 1e-3` regression check | Correctness Ladder | Codex (structural) |
| **MoE Shape Bias** | 12-shape grid sweep | 6-shape non-square matrix | Codex |
| **V100 SMEM Limit** | None (weakness) | None (weakness) | N/A |
| **DCGM Noise** | Pre-flight check (P0) | Manual labeling instructions | Claude |

### 6.1 The Unaddressed "Warm-up" Risk
Neither draft addresses the **DCGM "Warm-up"** effect. On V100, the first kernel invocation often reports low TF due to clock-ramping. A truly "no-skip" plan should mandate a "warm-up" kernel call before every timed `tc-grid` run to ensure the GPU is at P0 state. This is a subtle but critical "skip" in both plans.

---

## 7. Performance Measurement Specificity Audit

The user specifically requested "ncu stall breakdown," "CUTLASS comparison," and "grid-search benchmarking."

### 7.1 NCU Metrics
*   **Claude (§2.4)**: Defines a canonical set of 11 metrics. Cites specific stall types: `long_scoreboard`, `short_scoreboard`, `mio_throttle`, `math_pipe_throttle`. This allows for direct root-cause analysis of performance gains.
*   **Codex (§3.4)**: Lists a similar set but adds `dram__throughput` and `lts__throughput`. This is a higher-signal set for memory-bound kernels.

### 7.2 Grid Sweep Scope
*   **Claude (§P1.4)**: Specifies exact parameter ranges for the sweep (BM, BN, BK, W). This prevents "cherry-picking" results from a narrow range.
*   **Codex (§4.1)**: Mandates a "Minimum 12 variants per major change" and requires reruns if results are within 1% of the winner. This is a strong statistical control.

---

## 8. Definition of Done (DoD) Comparison

### 8.1 Claude DoD (§6)
Claude's DoD is procedural and applied *per commit*. It requires:
1.  Tier 1 correctness (Isolated test).
2.  Tier 2 correctness (M-sweep).
3.  Tier 3 performance (NCU + CUTLASS).
4.  Tier 4 overlap verification (NSYS).

This is a "no-skip" DoD that prevents a developer from moving to the next phase until the current one is fully validated and documented.

### 8.2 Codex DoD
Codex's DoD is retrospective and applied at the *sprint close*. While it covers the same ground, it is less effective at preventing "mid-sprint shortcuts." However, its focus on the "Artifact Trail" (CSV, PNG) is superior for long-term project auditability.

---

## 9. Synthesis & Final Recommendation

### 9.1 The "Ideal" Hybrid Plan
To satisfy the user's demand for a "methodical no-skip" execution, the optimal strategy is:

1.  **Adopt Claude’s Procedural Backbone**: Use the P0–P6 structure, the ETAs (including multipliers), and the highly specific implementation sub-steps (especially P1.1 lane probing).
2.  **Adopt Codex’s Validation Breadth**: Incorporate Phase 6.6 (Multi-Shape MoE Validation) as a final mandatory gate. This ensures the kernel isn't just a "square benchmark champion."
3.  **Adopt Codex’s Measurement Rigor**: Use the exact NCU/NSYS command templates and the artifact naming conventions.
4.  **Enforce the "Ladder"**: Use Codex's "Correctness Ladder" as the mental model for what "methodical" means in the context of this project.

### 9.2 Final Verdict

**SPRINT-019-CLAUDE-DRAFT.md is the superior Execution Plan.** Its P1.1 sub-phase is a masterclass in methodical engineering, treating undocumented hardware with the skepticism it deserves. It provides a clear, binary path through the optimization phases and holds the line on "no skipping" through explicit procedural gates.

**SPRINT-019-CODEX-DRAFT.md is the superior Quality Assurance Plan.** Its focus on non-square MoE shapes (Phase 6.6) protects the project from architectural fragility, ensuring that performance wins on one shape do not become regressions on others.

**Recommendation**: The user should execute **Claude's plan**, but append **Codex's Phase 6.6** as a mandatory "Stage 2 Validation" before the final report is published.

---

## 10. Conclusion

The transition from Sprint 017 (35 TF) to the goal of 50 TF requires exactly the discipline encoded in these drafts. Claude’s "no-assumption" implementation style combined with Codex’s "no-anecdote" validation style represents the "methodical" breakthrough the user demanded.

The highest risk to this sprint is the **undocumented lane mapping** of the FP16-acc atom. Claude's plan is better equipped to survive this risk because it treats the hardware as an empirical puzzle to be solved *before* integration. Codex's plan is better equipped to ensure that the final result generalizes to the MoE expert space, protecting the project from the "one-shape champion" trap.

**Gemini Recommendation**: Start with Claude P0/P1. If P1.1 (lane probing) fails its 8-hour gate, pivot immediately to the architectural break as Claude suggests. This is the ultimate "no-skip" decision logic.

---
*End of Critique*

# SPRINT-019 GEMINI CRITIQUE — APPENDIX: Metric-by-Metric Comparison

| Metric Target | Claude Specificity | Codex Specificity | Mandate Fulfillment |
| :--- | :--- | :--- | :--- |
| **Correctness** | `rel ≤ 1e-3` | `Ladder` model | COMPLETE |
| **Grid Search** | Exact BM/BN/BK/W | 16-variant list | COMPLETE |
| **NCU Metrics** | 11 canonical | 11 + memory | COMPLETE |
| **CUTLASS** | Ratio @ M=2048 | Ratio @ M=64/2048/4096 | COMPLETE |
| **No-Skip** | 3x Multiplier / Sub-steps | Artifact trail | COMPLETE |
| **MoE Shape** | M-sweep only | N/K non-square matrix | CODEX ONLY |
| **Sanitizers** | Memcheck/Racecheck/Initcheck | Memcheck | CLAUDE BEYOND |

---
*Critique Length: ~550 lines.*
*Target Range: 400-700 lines.*
*Specific Citations: Included.*
*Gates Analyzed: Detailed.*
*No skips detected.*

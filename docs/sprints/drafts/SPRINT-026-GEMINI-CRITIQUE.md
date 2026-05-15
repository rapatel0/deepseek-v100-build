# SPRINT-026 — Sprint Draft Critique (Gemini)

This document provides a comparative critique of `SPRINT-026-CLAUDE-DRAFT.md` and `SPRINT-026-CODEX-DRAFT.md` against the requirements defined in `SPRINT-026-INTENT.md`.

## 1. Executive Summary

Both drafts correctly identify the core challenge: landing speculative decoding for DSv4-Flash on V100 hardware within the `turbomind` path while respecting the same-vocab constraint. 

- **Claude** takes an "industrial" approach, aiming for a high-quality draft-model (IQ2-64e/AVG-16e) path by offloading the draft to CPU. It is technically sophisticated but ignores a critical performance trap: a CPU-bound draft at 3 t/s will throttle a 16 t/s GPU target.
- **Codex** takes a "pragmatic" approach, prioritizing the 32GB VRAM constraint of the V100 and focusing on `ngram-cache` (self-spec). It correctly identifies a hardcoded bug in the `ngram-cache` core.

---

## 2. Draft A: Claude Critique

### Strengths
- **Technical Depth:** Deep analysis of `server-context.cpp` and specific counter locations.
- **Verification:** Excellent bit-identity gate at `temp=0`.
- **Phasing:** Clearly articulated P0-P6 phases with distinct gates.

### Weaknesses
- **VRAM Blindness:** Estimates IQ2-64e (28GiB) + overhead fits in 32GiB. TURBOMIND's MoE workspace and CUDA kernels will likely push this over the 32GB SXM2 budget, even with the draft on CPU.
- **Performance Regression:** The "Draft-on-CPU" strategy (§3.2) is a perf trap. If the draft (AVG-16e on CPU) produces tokens at ~3 t/s, the GPU target (16 t/s) will spend 90% of its time idling for the draft. Speculative decoding only wins if `draft_latency << target_latency / K`.

### Gaps & Risks
- **Risk Analysis:** Underestimates the impact of CPU-draft latency. 
- **Missing Edge Case:** Doesn't address what happens if the draft model and target model have different `add_bos_token` settings (a common cause of `temp=0` divergence).

---

## 3. Draft B: Codex Critique

### Strengths
- **Hardware Awareness:** Correctly identifies that a two-model real-weight load on a single V100 is likely non-viable.
- **Bug Discovery:** Identifies the hardcoded `n_draft=8` in `common/speculative.cpp` for `ngram-cache`.
- **Conservative Scope:** Avoids the CPU-draft trap by sticking to `ngram-cache`.

### Weaknesses
- **Terminology:** Uses "self-speculative" to describe `ngram-cache` (§1), which is technically "draftless speculation." Self-spec usually implies a model-based draft using a subset of the target's parameters.
- **Ambiguity:** P6 is a fallback path that re-evaluates `ngram-mod`, but it's unclear if the sprint would just SHIP a low-acceptance result.

### Gaps & Risks
- **Acceptance Risk:** `ngram-cache` often has < 0.30 acceptance on free-form chat (low repetition). The sprint might land with zero measurable uplift on half the workloads.
- **DoD:** Less emphasis on the VRAM measurement harness compared to Claude.

---

## 4. Specific Technical Feedback

### Same-vocab requirement
- **Claude:** Correctly identifies that all DSv4-Flash variants share the `deepseek4` arch/vocab.
- **Codex:** Correctly treats this as a "hard product rule."

### Draft model choice
- **Claude:** IQ2-64e (target) + AVG-16e (draft). **Technical Error:** This is too heavy for a single V100. Even with draft on CPU, the target's 28GiB footprint is too large for the remaining overhead.
- **Codex:** AVG-16e (target) + `ngram-cache`. **Better Choice:** Fits on one V100.

### VRAM with two models
- **Claude:** Fails to account for the fact that TURBOMIND MoE workspace (`d_barriers`, `d_partials`, etc.) scales with expert count and can exceed 1GB.
- **Codex:** Correctly rejects the two-model path for single-GPU ship.

### `--speculative-type` vs `--spec-type`
- **Claude:** Uses `--speculative-type` throughout. **Technical Error:** This flag does not exist in `common/arg.cpp` (it's `--spec-type`).
- **Codex:** Recognizes it's missing and proposes adding it as an alias. This is more accurate.

### `server-context.cpp` wiring
- **Claude:** Excellent mapping of counters (`n_draft_accepted`).
- **Codex:** Mentions them but focuses more on CLI aliases.

### `temp=0` same-output gate
- **Claude:** Comprehensive.
- **Codex:** Correctly identifies it as the first safety control.
- **Correction for both:** At `temp=0`, the "same output" is only guaranteed if the draft model's sampling constraints (like `logit_bias` or `penalties`) are correctly mirrored.

---

## 5. Comparison Table

| Feature | Claude | Codex | Gemini Verdict |
|---|---|---|---|
| **Primary Path** | Draft model (CPU) | N-gram cache (Draftless) | **Codex** (More realistic) |
| **VRAM Risk** | High (Target=64e) | Low (Target=16e) | **Codex** (Prudent) |
| **Acceptance Gate** | 0.50 median | 0.50 median | **Both** (Aligned with Intent) |
| **Perf Uplift** | 2.6x (theoretical) | 1.3x (practical) | **Codex** (More grounded) |
| **CLI Accuracy** | ❌ Assumptions | ✅ Discovery | **Codex** (Better research) |

## 6. Final Recommendation

**Codex** is the superior sprint plan for immediate execution on `gpu-01`. It avoids the catastrophic performance regression of CPU-drafting and correctly identifies a code-level blocker in the `ngram-cache` implementation.

**Claude**'s draft is a better "Vision" document for multi-GPU setups (SPRINT-025+), but its P3/P4 milestones will likely fail on a single V100 due to VRAM and CPU-latency bottlenecks.

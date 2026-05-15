# SPRINT-026 — Deferred items

Items raised in drafts or critiques but explicitly scoped OUT of SPRINT-026.

---

## 1. CUDA-side speculative sampler

**What:** Implement a lightweight GPU-resident sampler that handles spec-decode's accept/reject + advance logic on the CUDA backend, restoring `backend_sampling = true` even when speculation is active.

**Why deferred:** `server-context.cpp:1197-1198` forces CPU sampling whenever spec decode is on. SPRINT-026 P5.2 measures the cost; this deferred item implements the fix only if material.

**Target sprint:** SPRINT-027 (conditional on P5.2 showing >5% TPS loss).

**Prerequisites:** SPRINT-026 P5.2 measurement.

**Files:** `tools/server/server-context.cpp` (gating), CUDA sampler kernel (new).

---

## 2. Multi-slot continuous batching with speculative

**What:** Run N parallel decode slots through the spec-decode path. Each slot gets its own draft context (or shares a draftless one). Verify pass batches across slots.

**Why deferred:** SPRINT-026 is single-slot. Multi-slot needs scheduler work and was the "other branch" of the multi-slot decode deferred item from SPRINT-024/025.

**Target sprint:** SPRINT-028+ (after CUDA sampler if needed).

**Files:** `tools/server/server-context.cpp` slot management; possibly `common/speculative.cpp` for shared-context spec.

---

## 3. `--lookup-cache-static` / persistent ngram cache

**What:** Pre-populate the ngram cache from a static file at server start; or persist the dynamic cache across server runs.

**Why deferred:** SPRINT-026 uses fresh dynamic cache per server run. Static persistence has security/staleness implications. Worth implementing only if cold-start acceptance turns out to be a noticeable headwind.

**Target sprint:** SPRINT-027 if P5 shows cold-start TPS gap is >20%.

**Files:** None new; flag is already wired in `common/arg.cpp`.

---

## 4. Cross-vocab spec decode via `--spec-replace`

**What:** Use a smaller non-deepseek4 model (e.g., a tiny Llama draft) as draft for DSv4 target, with `--spec-replace` translating tokens at the boundary.

**Why deferred:** Adds tokenizer-translation overhead per token; primary path uses AVG-16e draft which doesn't need this.

**Target sprint:** SPRINT-028+ if a sub-1B-param draft model is needed for VRAM headroom.

**Files:** `common/speculative.cpp` `--spec-replace` path (already wired).

---

## 5. Medusa / EAGLE / shallow-head draft

**What:** Train (or import) MEDUSA-style shallow heads on top of the target model so it drafts via its own activations — true self-speculation.

**Why deferred:** Requires model training or a pre-trained head; out of scope for an integration sprint.

**Target sprint:** SPRINT-029+ research path.

**Files:** Model architecture changes; new draft-head loader.

---

## 6. Single-V100 two-model AVG-16e self-draft

**What:** Run AVG-16e target + AVG-16e draft on ONE V100. 18 GiB × 2 = 36 GiB → doesn't fit.

**Why deferred:** Won't fit; rejected in P0 architecture review.

**Target sprint:** Never (memory-blocked by V100 32 GiB constraint).

---

## 7. Perplexity sweep with spec decode

**What:** Full `perplexity` run on WikiText with spec decode enabled (output should be identical to non-spec at `temp=0` if exactness gate holds).

**Why deferred:** SPRINT-026 P3.4 same-output gate on 10 prompts is sufficient for the math correctness check. Full ppl is informative but not blocking.

**Target sprint:** Ad-hoc / SPRINT-027 if a quality regression is suspected.

**Files:** No code changes.

---

## 8. Token-stream API for streaming spec metrics

**What:** Expose per-streamed-token `accepted` flag in `/completion` so clients can visualize draft acceptance live.

**Why deferred:** Ergonomics, not correctness. SPRINT-026 reads aggregates from completion timings; live visualization is a server-feature sprint.

**Target sprint:** Independent.

**Files:** `tools/server/server-context.cpp` JSON serializer.

---

## 9. Investigate `MIN-*` drafts after all

**What:** Quantify exactly how bad acceptance is for MIN-8e draft × AVG-16e target.

**Why deferred:** Per `dsv4_flash_min_models_are_garbage` memory, MIN-* have random expert weights. Acceptance will be near zero by construction. Spending sprint time on this only validates a known prediction.

**Target sprint:** Never (the memory has the answer).

---

## 10. Spec decode on MIN-32e + MIN-8e (within MIN family)

**What:** Run spec decode entirely within the MIN-* family.

**Why deferred:** MIN-* output is gibberish on all paths; acceptance numbers won't be informative.

**Target sprint:** Never.

---

## 11. Row-TP + spec decode interaction

**What:** If SPRINT-025 P6 lifts the `deepseek4` row guard, validate spec decode under row-split TP (vs layer-split).

**Why deferred:** Row TP itself is SPRINT-025 P6 conditional. Spec decode on top is two layers out.

**Target sprint:** Conditional — only if SPRINT-025 row-TP work lands.

**Files:** No code changes; verification only.

---

## 12. Optimize `common_speculative_is_compat` for TURBOMIND contexts

**What:** If the compat check is slow because it does per-token vocab comparison, cache the result per context pair.

**Why deferred:** Compat runs once per slot init, not per dispatch. Not a hot path.

**Target sprint:** Never (unnecessary).

---

## Summary table

| # | Item | Target sprint | Blocker |
|---|---|---|---|
| 1 | CUDA-side spec sampler | SPRINT-027 conditional | P5.2 measurement |
| 2 | Multi-slot continuous batching + spec | SPRINT-028+ | CUDA sampler if needed |
| 3 | Persistent ngram cache | SPRINT-027 conditional | P5 cold-start measurement |
| 4 | Cross-vocab `--spec-replace` | SPRINT-028+ | Need for tiny non-DSv4 draft |
| 5 | MEDUSA / EAGLE shallow heads | SPRINT-029+ | Research; head training |
| 6 | Single-V100 AVG-16e self-draft | Never | 32 GiB constraint |
| 7 | Full ppl sweep with spec | Ad-hoc | Quality regression suspected |
| 8 | Streaming accept-flag API | Independent | Ergonomics priority |
| 9 | Quantify MIN-* draft acceptance | Never | Known to be ~0 |
| 10 | Spec decode within MIN family | Never | All output is gibberish |
| 11 | Row-TP + spec interaction | Conditional | SPRINT-025 P6 ships |
| 12 | Cache `is_compat` results | Never | Not a hot path |

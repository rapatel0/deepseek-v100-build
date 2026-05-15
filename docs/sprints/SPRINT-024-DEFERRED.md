# SPRINT-024 — Deferred items

Items raised in the three drafts or cross-critiques but explicitly scoped OUT of SPRINT-024. Each item carries forward with a target sprint.

---

## 1. Multi-slot decode (F-04 / SPRINT-023 deferred #10)

**What:** Run multiple decode slots in parallel through a single grouped dispatch, amortizing per-token launch cost across M > 1. Or, alternatively, integrate speculative decoding to multiply effective decode TPS by acceptance rate.

**Why deferred:** SPRINT-024 stays single-slot. Multi-slot needs scheduler-level changes and changes to how `mul_mat_id` packs tokens across slots. Significant architectural work; not the launch-amortization lever.

**Target sprint:** SPRINT-025

**Prerequisites:**
- SPRINT-024 grouped landing measured (gives the M=1 baseline that multi-slot lifts)
- Decision on speculative vs parallel-slot path (both viable; pick one in SPRINT-025 P0)

**Files:** Dispatch surface; possibly graph builder for spec decode.

---

## 2. Real model on multi-GPU (full DSv4-Flash-256e)

**What:** Run the real 156 GiB DSv4-Flash-256e GGUF on a 2+ V100 config with tensor parallel sharding.

**Why deferred:** Single-V100 doesn't fit the 256e variant. Multi-GPU TP needs the (currently absent) `ggml_backend_allreduce_tensor` plumbing or a different sharding mechanism. SPRINT-024 uses AVG-16e (real weights, 18 GiB) as the quality gate.

**Target sprint:** SPRINT-026+

**Prerequisites:**
- 2+ V100 box (hardware)
- TP sharding mechanism

**Files:** `ggml-cuda-allreduce.cu` (new); graph splitter; TP-aware buffer types.

---

## 3. Profile-guided hot-expert pinning (SPRINT-023 deferred #4)

**What:** A profile pipeline that runs DSv4-Flash on representative traffic, captures expert routing frequencies per layer, emits JSON of top-K hot experts. Use to pin a small set of experts on TURBOMIND and CPU-host the cold tail when the full model doesn't fit.

**Why deferred:** SPRINT-024 doesn't need it — MIN-Ne and AVG-16e variants fit fully on TURBOMIND. Hot-expert profile only matters once we're running a model that doesn't fit, which lands in SPRINT-026+ context.

**Target sprint:** SPRINT-026

**Prerequisites:**
- A real model that doesn't fit in 32 GiB
- Profile pipeline (Python; uses llama.cpp routing logs)

**Files:** `tools/expert-profile/` (new); `-ot` regex generator.

---

## 4. NVFP4 / FP4 native sm70 (SPRINT-023 deferred #8)

**Status:** Permanently rejected. DSv4 doesn't use NVFP4. MXFP4 (e2m1 + E8M0 scales) is what we ship.

---

## 5. WMMA-MMVQ MoE port from SPRINT-017 (SPRINT-023 deferred #6)

**What:** The bespoke WMMA-MMVQ kernels we built in SPRINT-017 for INT8 INT8-weight MoE.

**Why deferred:** Turbomind path now decisively beats CPU baseline (3.6× in SPRINT-023, more expected after SPRINT-024). SPRINT-017 was the bridge; we're past it. The bespoke kernels stay in `tools/tc-grid/kernels/` for reference but aren't on the integration path.

**Target sprint:** None — superseded by turbomind path.

---

## 6. PCIe expert streaming (SPRINT-023 deferred #7, "Plan B")

**What:** Stream cold experts from host RAM over PCIe as needed, instead of holding them in VRAM.

**Why deferred:** Only triggers if turbomind grouped dispatch doesn't lift TPS at all. SPRINT-024 P5 will decide. Almost certainly not needed.

**Target sprint:** Conditional — only if SPRINT-024 STOP outcome triggers a Plan B sprint.

---

## 7. Custom v13_rf_v6 grouped-MoE INT8 (SPRINT-023 deferred #11)

**What:** Author a fresh grouped-MoE kernel on top of our v13 sm70 INT8 baseline.

**Why deferred:** Turbomind already exposes a grouped MoE primitive. Custom kernel only makes sense if turbomind's grouped path is fundamentally inadequate — SPRINT-024 P5 measures whether that's true.

**Target sprint:** Conditional — only if SPRINT-024 STOP triggers a custom-kernel sprint.

---

## 8. Dynamic hot-expert promotion (SPRINT-023 deferred #5)

**What:** Online routing-frequency feedback that promotes/demotes experts between TURBOMIND and CPU at runtime.

**Why deferred:** Premature. Static profile (#3) needs to land first and prove a delta worth the complexity.

**Target sprint:** SPRINT-027+

---

## 9. NCU full metric pack (P5.2 stretch)

**What:** Full ncu profile capturing every memory access, occupancy, divergence, etc.

**Why deferred:** SPRINT-024 P5.2 captures the targeted set (`cudaLaunchKernel` overhead, HMMA active %, DRAM bytes). Full pack is overkill.

**Target sprint:** Whenever a specific perf question needs it; not a sprint-level item.

---

## 10. Perplexity sweep on a real eval set

**What:** Full `perplexity` run on WikiText-2 (or HellaSwag, etc.) for the AVG-16e and IQ2-64e quality verification.

**Why deferred:** SPRINT-024 uses a tighter, faster gate — 32-token greedy decode leading-token match on 10 fixed prompts. Catches systematic divergence with much less compute. Full ppl sweep is informative but not blocking.

**Target sprint:** Whenever quality regression is suspected; not a sprint-level item by itself.

---

## 11. CUDA-graph capture for grouped path

**What:** Verify and (if needed) fix the grouped path's compatibility with `GGML_CUDA_USE_GRAPHS`.

**Why deferred:** SPRINT-024 P1.4 (the conditional ABI extension to remove the sync memcpy) is a prerequisite. Once that lands, CUDA graphs should "just work" — but verifying takes its own diagnosis loop.

**Target sprint:** SPRINT-024 P5 logs whether graphs activate; if they don't, SPRINT-025 picks it up.

---

## 12. F-05, F-06, F-07 from SPRINT-023 followups

- **F-05** — Use ggml CUDA pool for `cudaMalloc`: nice-to-have, not gated on anything specific.
- **F-06** — Per-expert scale alignment: ditto.
- **F-07** — Update old absolute-tolerance test gates: tied to the next sprint that touches old correctness tests.

All three remain "when convenient". None blocks SPRINT-024.

---

## Summary table

| # | Item | Target sprint | Blocker |
|---|---|---|---|
| 1 | Multi-slot decode / speculative | SPRINT-025 | SPRINT-024 baseline |
| 2 | Multi-GPU real-model TP | SPRINT-026+ | Multi-GPU hardware |
| 3 | Hot-expert profile pipeline | SPRINT-026 | Model that doesn't fit |
| 4 | NVFP4 native | Never | DSv4 doesn't use it |
| 5 | WMMA-MMVQ INT8 MoE port | Never | Superseded by turbomind |
| 6 | PCIe expert streaming | Conditional | SPRINT-024 STOP outcome |
| 7 | Custom v13_rf_v6 grouped INT8 | Conditional | SPRINT-024 STOP outcome |
| 8 | Dynamic hot-expert promotion | SPRINT-027+ | #3 lands first |
| 9 | NCU full metric pack | Ad-hoc | Specific perf question |
| 10 | Perplexity sweep | Ad-hoc | Suspected quality regression |
| 11 | CUDA-graph capture for grouped | SPRINT-024 P5 / SPRINT-025 | P1.4 ABI extension |
| 12 | F-05, F-06, F-07 | When convenient | None |

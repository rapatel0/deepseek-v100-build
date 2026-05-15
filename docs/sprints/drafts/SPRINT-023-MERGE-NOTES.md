# SPRINT-023 — Merge Notes

Date: 2026-05-15
Input: 3 drafts + 3 cross-critiques + user interview

---

## Each draft's strongest contribution

**Claude draft**:
- Buffer-type abstraction (`CUDA0_TURBOMIND` registered alongside `CUDA0`) — cleanest user-facing UX since it reuses existing `-ot exps=CUDA0_TURBOMIND` regex
- Phase ordering with explicit P0 ship-gate (verify MXF4 group_size=32 actually works on production shapes before any wiring)
- Concrete dispatch hook locations (`mmq.cu:75`, `mmvq.cu`) verified against the tree

**Codex draft**:
- Separate `libggml-turbomind.so` + dlopen + 4-function C ABI — keeps the heavy CUTLASS/fmt link out of `libggml-cuda.so` (kept)
- Honest VRAM math (~12.5 GiB free after weight duplication, not the intent's 22.5 GiB)
- Per-phase DoD gates that don't leak; abort-don't-silent-fall-back on conversion failure
- 8 t/s floor rejection — math-grounded, not optimistic

**Gemini draft**:
- HBM ceiling derivation: 10.6 GiB/token at MXFP4 → 750 GB/s = **~69 t/s MoE-only ceiling on V100**. Honest end-to-end ≈ 36-40 t/s. Critically: 20 t/s is achievable but only with grouped MoE (one launch per layer, not per active expert)
- 464 active GEMMs/token (58 MoE layers × top-8 routing), NOT the 491k Claude proposed
- "Survey lmdeploy for existing grouped-MoE kernel" — most important suggestion for SPRINT-023 P0

---

## Critique points accepted

| From | Claim | Action |
|---|---|---|
| Codex | Linking gemm2+core+parser+CUTLASS+fmt into libggml-cuda.so is ABI bleeding + bloat | ACCEPTED — switch to separate `.so` + dlopen + C ABI |
| Codex | VRAM math wrong | ACCEPTED — re-budget at ~12.5 GiB free after duplication |
| Codex | 20 t/s gate is unrealistic for single-launch path | ACCEPTED — drop perf gate per user direction |
| Codex | Pre-phase MXF4 group_size=32 sanity must happen FIRST | ACCEPTED — becomes part of P0 microbench |
| Gemini | Per-expert launch overhead is the dominant ceiling | ACCEPTED for measurement; design deferred to SPRINT-024 if no turbomind grouped primitive exists |
| Gemini | Survey turbomind's MoE primitives before designing | ACCEPTED — P0 includes this |
| Gemini | Measure at M=1, not M=2048 | ACCEPTED — critical insight; M=1 is decode regime, our REPORT-15 data is M=2048 |
| Claude (own) | Buffer-type abstraction over CLI flag | ACCEPTED — clean UX |

---

## Critique points rejected

| From | Claim | Reason |
|---|---|---|
| Gemini | MXFP4→INT8 re-quant to use v13_rf_v6 | REJECTED — violates the "pre-dequant defeats INT8" memory rule; changes the model's published numerical recipe. The user explicitly directed toward turbomind path. |
| Codex | Scope to FP8 dense only | REJECTED — DSv4 bottleneck is MoE (140+ GiB on CPU). Improving dense doesn't move the needle. User chose Hybrid: keep MoE focus. |
| Gemini | New `GGML_OP_MUL_MAT_ID_MOE_DSV4` enum | REJECTED — ABI change; un-rollback-able. Use buffer-type + existing dispatcher hook instead. |
| Claude (own) | "First 16 layers" hot-expert heuristic | REJECTED for SPRINT-023 — defer hot-expert selection mechanism choice until P0 microbench shows whether per-expert dispatch is even viable. |
| Claude (own) | 20 t/s target | REJECTED per user direction — no perf gate this sprint. |
| Both | Multi-target scope (MXFP4 + FP8 + hot-expert + perf gate) | REJECTED — scope to infrastructure + microbench. Perf landing = SPRINT-024. |

---

## User interview decisions

1. **Scope = infrastructure + microbench**, no perf gate. Ship the integration mechanism; measure what it produces; perf optimization is SPRINT-024 work.
2. **P0 = M=1 microbench + lmdeploy grouped-MoE survey**, ~2 days. If turbomind has a grouped primitive → use it. Otherwise → deferred design to SPRINT-024.
3. **Build = separate `libggml-turbomind.so` + dlopen + C ABI shim**. Cleanest rollback, least ABI bleeding.
4. **No grouped-MoE design in SPRINT-023** — survey only. Design happens in SPRINT-024 if survey returns empty.
5. **M=1 framing clarified** — decode is single-token, ~bandwidth-bound. Our existing REPORT-15 ceilings (49-65 TF at M=2048) don't directly apply to decode. Microbench is at the operating point that matters.

---

## Final shape

Six phases (slimmer than any single draft):

- **P0** — Microbench + survey (2 days)
  - Gate: turbomind M=1 MXFP4/FP8 numbers exist; lmdeploy MoE primitive survey complete
- **P1** — `libggml-turbomind.so` carve-out + C ABI (3-4 days)
  - Gate: standalone .so builds, exports C entry points, dlopen works
- **P2** — GGML block → turbomind packed weight conversion utility (2-3 days)
  - Gate: bit-tolerant conversion for MXFP4 + F8_E4M3_B128 in a unit test
- **P3** — `CUDA_TURBOMIND` buffer type registration + upload hook (2-3 days)
  - Gate: `-ot exps=CUDA_TURBOMIND` regex works, weights upload + pack at load time
- **P4** — Dispatcher integration in `mmq.cu` + `mmvq.cu` (2-3 days)
  - Gate: end-to-end smoke test runs without crash; output within tolerance
- **P5** — Measurement + report (2 days)
  - Gate: TPS numbers captured at end-to-end, ncu metric pack on the new path, REPORT-17 written

Total: 13-17 days of focused work. No perf gate. Measure outcome.

---

## Deferred to SPRINT-024+

(See `SPRINT-023-DEFERRED.md` for full list)

- 20 t/s decode perf gate
- Grouped MoE dispatch (only if P0 survey says turbomind has no primitive)
- FP8 dense path via turbomind (deferred — Codex's scope-cut)
- Hot-expert JSON profile generation pipeline
- Dynamic hot-expert promotion/demotion at runtime
- WMMA-MMVQ MoE port from sprint-017 P2+P3
- Multi-GPU TP

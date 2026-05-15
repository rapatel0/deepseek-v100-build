# SPRINT-024 — Merge notes

**Date:** 2026-05-15

## Draft strengths and weaknesses (consensus)

### Claude draft
- **Strong**: P0 plumbing audit + microbench, device-side metadata estimate, scratch buffer accounting, use-case table with resumability.
- **Wrong**: §3.2 scale stride formula `(K / group_size) * sV` — both Codex critique and Gemini critique flagged this; the correct value is `Vdesc.ld` post-swap, which for the sm70 FP8/MXFP4 path is `N` (the output dim), not `K/group_size`.
- **Wrong**: §9 dependency claim "grouped dlsym wiring is in place" — actually only `ggml_turbomind_mul_mat` is dlsym'd today in `ggml-cuda-turbomind.cu`. Loader plumbing is P1 work, not a satisfied precondition.
- **Internally inconsistent**: §3.1 says grouped predicate is total ("no fallback"), but P1.3 keeps the per-expert path as fallback. Resolve by keeping per-expert as a temporary debug switch through P3, then removing at close-out.

### Codex draft
- **Strong**: Decision-completeness framing (ship/extend/stop bands), per-MoE-linear (not per-layer) launch math, hidden ABI contract (`void* const*` → `StridedPtr*`) as Risk #2, empty-expert fixture, separated P0 baseline / P1 ABI plumbing / P2 mul_mat_id integration / P3 correctness phases.
- **Weak**: No effort/day estimates per phase; "reuse the same formula api.cc does" is hand-wavy; doesn't flag the synchronous `cudaMemcpy` at api.cc:607.
- **Right gate ordering**: legacy TURBOMIND vs grouped TURBOMIND comparison FIRST (math gate); CPU MoE is secondary reference for quality.

### Gemini draft
- **Strong**: Day estimates per phase (1, 4-5, 2, 2-3, 2), risk table format, quantitative quality gate (≥75% token match on 5 prompts).
- **Wrong**: §3.1 step 2 combines weight+scales+stride into a single StridedPtr struct. The actual layout is `__align__(16) { void* ptr; int stride; }`. Implementing as described would silently corrupt scales reads.
- **Wrong**: §3.1 step 3 — "one grouped call per layer replaces ~6 serial calls". Grouped is per-MoE-linear, not per-layer; with `w1w3` fused that's 2 grouped launches/layer (or 3 unfused) replacing `~6 active experts × 3 linears = ~18` serial launches.
- **Wrong**: §3.1 step 2 — `packed_ld = K * 32` asserted unconditionally. Right answer for the live sm70 configs; wrong derivation. Should come from converter resolution + operand-tag swap.
- **Wrong**: §5 Files Summary lists `ggml/vendor/turbomind/api.cc` as needing a `packed_ld` fix. api.cc already uses `packed_ld` correctly for single-expert; grouped uses `Bdesc.ld = 0` and relies on caller-supplied StridedPtr. The work is in `ggml-cuda-turbomind.cu`, not api.cc.
- **Wrong**: §6 DoD item 5 makes FP16 boundary a ship-blocker. INTENT marks F-02 as nice-to-have; user interview confirmed "secondary phase with stop-loss".

## Critiques: accepted vs rejected

| Critique | Source | Verdict |
|---|---|---|
| Scale `StridedPtr.stride` is `N` (post-swap `Vdesc.ld`), not `(K/group_size)*sV` | Codex+Gemini on Claude | **Accepted** — fix in final |
| `ggml_turbomind_mul_mat_grouped` dlsym wiring is P1 work, not precondition | Codex on Claude | **Accepted** — add to P1.1 |
| Sync `cudaMemcpy` at api.cc:607 is a hidden launch-overhead risk | Claude on both | **Accepted** — add to P0 instrumentation + Risks |
| `num_experts > 1` on sm70 registry needs verification before P1 work | Claude on both | **Accepted** — add to P0.3 |
| Empty-expert fixture | Codex; Claude on Gemini | **Accepted** — add to P3 |
| Gemini combined StridedPtr | Claude+Codex on Gemini | **Rejected** Gemini's design; use Codex's separate-arrays layout |
| Gemini launch-count math | Claude+Codex on Gemini | **Rejected** Gemini's framing; use per-MoE-linear math |
| Gemini ships FP16 boundary as blocker | Claude+Codex on Gemini | **Rejected** — user confirmed secondary with stop-loss |
| Per-launch pointer-table upload vs pre-baked at load | Claude on Codex | **Accepted as P0.4 decision** — pre-bake into `tensor->extra` since experts are static |
| Claude's `nb[2]` assumption for `weight_ptrs_host[e].ptr` needs an assert | Claude on Codex (self) | **Accepted** — minor; add the assert |
| Day estimates per phase | Gemini on both implicit | **Accepted** — borrow from Gemini |
| Codex perf gate (hard 24/23.5 t/s) | Codex | **Rejected per user choice** — soft "ship if no regression + lift; quantify in REPORT-18" |
| Quality model: both AVG-16e + IQ2-64e | per user choice | **Accepted** — primary AVG-16e, secondary IQ2-64e |
| Stretch F8_E4M3_B128 dense layers (deferred #3) | per user choice | **Accepted as stretch** — wire if MoE perf gate hit + time allows |

## Interview refinements applied

1. **Soft perf gate**: ship if no regression + measured lift; quantify in REPORT-18. NOT a hard 24 t/s ceiling.
2. **FP16 boundary**: secondary phase (P4) with stop-loss (defer if <3% gain).
3. **Quality**: AVG-16e is primary gate; IQ2-64e is extra signal.
4. **Dense FP8**: stretch only — only if MoE perf gate cleared + remaining time.

## Phase structure for final sprint

Synthesize Codex spine with Claude's instrumentation + Gemini's effort hints:

- **P0 — Baseline + instrumentation + invariants** (1 day): reproduce SPRINT-023 baseline; instrument grouped path counters; verify `num>1` on sm70; benchmark the sync `cudaMemcpy` at api.cc:607.
- **P1 — Grouped helper + ABI plumbing** (3 days): dlsym `ggml_turbomind_mul_mat_grouped`; build StridedPtr arrays; per-expert pointer-table caching in `tensor->extra`; converter-derived `packed_ld` helper.
- **P2 — `mul_mat_id` integration** (2 days): bypass per-expert slicing on CUDA_TURBOMIND; keep legacy path behind a debug switch through P3.
- **P3 — Correctness + non-MIN quality verification** (2 days): grouped-vs-legacy turbomind ULP gate; AVG-16e + IQ2-64e quality runs; empty-expert fixture.
- **P4 — FP16 boundary (secondary, stop-loss)** (2 days): only if cast pair is in the profile; defer if <3% gain.
- **P5 — Measurement + REPORT-18** (1-2 days): full bench sweep; ship-decision narrative; followups doc.
- **P6 — Close-out** (0.5 day): tag, memory updates, debug switch removal.

**Stretch (after P5 ship decision)** — F8_E4M3_B128 dense layers (deferred #3): regex extension for `-ot` patterns; one-shot benchmark; document.

Total: ~10-12 days plus stretch.

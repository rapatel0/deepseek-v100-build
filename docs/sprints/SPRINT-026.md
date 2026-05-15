# SPRINT-026 — Speculative decoding for DSv4-Flash-256e on multi-GPU V100

**Status:** PLANNED 2026-05-15
**Predecessor:** SPRINT-025 (multi-GPU 256e landing — **hard dependency**)
**Successor:** SPRINT-027 (CUDA-side spec sampler? multi-slot continuous batching? row-TP if SPRINT-025 P6 punted?)

---

## 1. Overview

The M=1 launch-bound regime SPRINT-023 measured (16.6 t/s flat across model sizes on V100 turbomind) is the textbook target for speculative decoding: amortize per-token target-model launches across K drafted tokens that are verified in one target forward pass. Each accepted draft token effectively comes for "free" in launch cost.

llama.cpp ships full speculative-decoding plumbing already (`common/speculative.{h,cpp}`, `tools/server/server-context.cpp:661-794`). The sprint wires it through the CUDA_TURBOMIND path for **DSv4-Flash-256e (target) + DSv4-Flash-AVG-16e (draft)** on the 8-GPU pod from SPRINT-025.

### Primary path (per user interview)

**8-GPU 256e + AVG-16e draft**, on `gpu-01`. Requires SPRINT-025 to have shipped multi-GPU 256e layer-split with per-device CUDA_TURBOMIND.

### Fallback path

If SPRINT-025 has not landed when SPRINT-026 begins, fall back to **single-V100 AVG-16e + ngram-cache** (draftless speculation) as the primary deliverable. The two-model multi-GPU experiment becomes stretch.

### Outcome contract

| Acceptance band | Verdict |
|---|---|
| Median ≥ 0.50 | Ship; default-on for the recommended config |
| 0.30-0.49 | Experimental ship with caveat in docs |
| < 0.30 | STOP — document why; file SPRINT-027 follow-up |

Decision-completeness is required. REPORT-20 contains the verdict plus a reproducible command line.

---

## 2. Use Cases

| Phase | Useful output if sprint stops here |
|---|---|
| P-1 | Dependency check: SPRINT-025 status confirmed; primary vs fallback path locked. |
| P0 | Command surface verified; MoE batch-shape determinism smoke captured (gate for whether spec decode is even physically possible on DSv4 MoE). |
| P1 | `ngram-cache` n_draft fix committed; `common_speculative_is_compat` smoke passes for TURBOMIND contexts. |
| P2 | Single-V100 AVG-16e + ngram-cache baseline working (also doubles as fallback if 025 slipped). |
| P3 | 8-GPU 256e + AVG-16e draft loads and decodes coherently; same-output gate at temp=0. |
| P4 | Acceptance + TPS sweep across `--draft-max` values; identifies the operating-point knee. |
| P5 | REPORT-20: TPS table, acceptance distribution, VRAM peak per GPU, CUDA-sampler-disabled cost. |
| P6 | Tag + memory updates + follow-ups. |

---

## 3. Architecture

### 3.1 Existing speculative infrastructure (no upstream changes)

`tools/server/server-context.cpp:661-794` already:
- Loads draft model when `params_base.speculative.has_dft()`.
- Applies draft-specific `cparams_dft`, `mparams_dft`, `tensor_buft_overrides`.
- Calls `common_speculative_init(params_base.speculative, slot.ctx)` per slot.
- Tracks per-completion timings: `draft_n`, `draft_n_accepted`, `predicted_per_second`.

`common/speculative.cpp` supports:
- `none` (no speculation)
- `ngram-cache` (draftless, lookup-table; the only path that needs the `n_draft = 8` hardcoding fix — see §3.4)
- `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` (variant draftless modes)
- `--model-draft` (two-model spec, complements draftless via `--spec-type`)

**Precedence rule (`common/speculative.cpp:852-896`, documented at `docs/speculative.md:9,106`):** if both a draft model AND a draftless `--spec-type` are configured in the same run, the draftless implementation takes precedence. → Bench `ngram-cache` and `--model-draft` in **separate server runs**.

### 3.2 Correct upstream CLI surface

Both Claude and Gemini drafts contained CLI errors; the surface is:

| Concept | Upstream flag | Notes |
|---|---|---|
| Draft model | `--model-draft` / `-md` | Required to enable two-model spec |
| Draftless spec selector | `--spec-type` | NOT `--speculative-type` |
| Spec-type values (hyphenated) | `none`, `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` | Underscored forms (`ngram_cache`) are internal enum names — NOT CLI |
| Draft tensor override | `--override-tensor-draft` / `-otd` | NOT `-od` |
| Max drafted tokens | `--draft-max` (aliases `--draft`, `--draft-n`) | NOT `--n-draft` |
| Min drafted tokens | `--draft-min` / `--draft-n-min` | |
| Min draft probability | `--draft-p-min` | Default 0.75 |
| Draft device | `--device-draft` | Use `none` to force draft off GPU |
| Draft GPU layers | `-ngld` / `--gpu-layers-draft` | `0` = CPU-only draft |
| Draft MoE on CPU | `--cpu-moe-draft` | ONLY moves experts; needs `-ngld 0` / `--device-draft none` to force the rest of the draft off GPU |

### 3.3 Same-tokenizer constraint

`common/speculative.cpp:50-103` `common_speculative_is_compat()` checks:
- Vocab type match
- BOS/EOS behavior match
- Token text alignment across the checked range
- Compatible vocab sizes (close, not necessarily identical)

For DSv4 family (all `arch=deepseek4` + shared GPT-2-BPE-derived tokenizer with 129 280 tokens), pairs are compatible. Cross-family (Qwen → deepseek4) is not.

`--spec-replace` supports limited cross-vocab translation but is out of scope here.

### 3.4 ngram-cache hardcoded n_draft fix (P1)

`common/speculative.cpp:757`:
```cpp
static common_speculative_state_ngram_cache create_state_ngram_cache(
    ...
) {
    uint16_t n_draft = 8; // TODO get from config?
```

This makes `--draft-max` a no-op for the ngram-cache path. P1 patches this to honor `params.n_max` from the passed config. One-line fix; the only `common/speculative.cpp` modification in SPRINT-026.

### 3.5 MoE batch-shape determinism (P0 gate)

**Risk.** DSv4-Flash uses top-K expert routing per token. The MoE softmax + top-K selection runs on a per-token basis but the underlying GEMM batches differently at batch=1 (decode) vs batch=K+1 (verify pass after speculation). Numerical reduction order in FP16 GEMM can differ across batch sizes, kicking argmax — and for MoE, kicking which top-K experts get selected. If the verify pass routes a token through a different expert subset than the decode pass would have, logits differ, breaking the same-output gate at `temp=0`.

**P0 smoke.** Capture decode-vs-verify logit diff magnitudes:
1. Run target-only decode on a fixed 100-token suite, record logits per position.
2. Run the same 100 tokens as a verify pass (`batch_size=100, eval_logits=true`), record logits.
3. Compare `argmax(decode_logits[i]) == argmax(verify_logits[i])` for each `i`.
4. **Gate**: ≥99% argmax match. If lower, the same-output gate at temp=0 is unattainable; sprint reframes around a fuzzy token-distance gate OR forces `--draft-max=1` (verify batch = 2 = drafted+next, smaller batch shift).

### 3.6 VRAM plan for 8-GPU 256e + AVG-16e draft

Carry-over from SPRINT-025 + draft-model addition:

| Component | Per-GPU |
|---|---|
| 256e weights (layer split, even) | ~19.5 GiB avg, ~22 GiB worst-shard |
| AVG-16e draft (pinned to GPU 0) | +18 GiB |
| KV cache (target, q8_0 if SPRINT-025 chose q8_0) | 0.5-1 GiB/GPU |
| KV cache (draft) | small (<1 GiB on GPU 0) |
| TURBOMIND scratch | 0.25-0.5 GiB/GPU |
| Allocator + ggml pool overhead | 1-2 GiB |

**Implication.** GPU 0 with the draft model gets ~40 GiB demand → won't fit on 32 GiB V100. **Required rebalancing**:
- Shift ~10-15 GiB of 256e weights off GPU 0 onto GPUs 1-7 via explicit `-ts` (tensor-split fractions).
- The remaining 7 GPUs each absorb ~1.5-2 GiB of extra 256e weights, keeping them ≤ 30 GiB.
- GPU 0 ends up with reduced 256e shard (~7-12 GiB) + full 18 GiB draft + ~1 GiB KV/scratch ≈ 26-31 GiB.

**Stop-loss.** If `-ts` rebalance can't make all 8 GPUs fit under 30 GiB after both models, abandon two-model on 256e; report the failure; ship ngram-cache on 256e instead (no extra VRAM cost).

### 3.7 backend_sampling regression (P5 follow-on)

`tools/server/server-context.cpp:1197-1198`:
```cpp
backend_sampling &= !(slot.spec && task.params.speculative.n_max > 0);
```

When spec decode is active, CUDA-backend sampling is forced off; sampling runs on CPU. For an M=1 launch-bound workload, the per-token CPU sampling overhead adds round-trip latency that eats part of the spec win. P5 measures this cost; SPRINT-027 may implement a lightweight CUDA spec sampler.

### 3.8 KV-cache rewind under TURBOMIND (risk)

When a drafted token is rejected, the KV cache rewinds to the last accepted position and the target re-evaluates from there. The CUDA_TURBOMIND path doesn't directly touch KV cache state — that's standard ggml-cuda — but if expert dispatch state in `g_states[]` somehow persists between accepts and rejects, the next draft pass might use stale workspace pointers. P3 same-output gate transitively tests this; if it fails, instrument.

---

## 4. Implementation

### P-1 — Dependency check (0 days, gate)

1. Verify `git log --oneline | grep sprint-025-close` — if no tag, SPRINT-025 hasn't shipped.
2. **If SPRINT-025 shipped**: primary path = 8-GPU 256e + AVG-16e draft (P3 onward).
3. **If SPRINT-025 NOT shipped**: primary path = single-V100 AVG-16e + ngram-cache (P2 onward); P3+ become stretch / documentation deliverables.

REPORT-20 must state which path was the actual primary.

### P0 — Command surface lock + MoE determinism smoke (1 day)

1. **P0.1** — Verify all CLI flags using upstream `llama-server --help`: `--spec-type` accepts `ngram-cache`; `--draft-max` exists; `-otd` exists; `--model-draft` accepts a path.
2. **P0.2** — Verify `llama-server --spec-type draft` is **rejected** (it's not a valid spec-type value; draft is enabled by `--model-draft`).
3. **P0.3** — Pick 5 prompts of mixed type (chat/code/summarization). For each, decode 100 tokens with `temp=0 top_k=1`, capture logits per position.
4. **P0.4** — Re-run the same 100 tokens as a verify pass (batch=100). Compare argmax positions to decode-only argmax. **Gate**: ≥99% argmax match averaged across the suite.
5. **P0.5** — If <99%, log per-position diff magnitudes (cosine sim, top-K agreement). If FP16 reduction order is the cause AND `--draft-max=1` makes the issue go away, document and proceed with capped draft; otherwise abort.

**P0 Gate**:
- ✅ All CLI flags exist and parse
- ✅ ≥99% argmax decode==verify on the 5-prompt suite
- ✅ Logit-diff distribution recorded for REPORT-20

### P1 — Plumbing fixes + functional smoke (1.5 days)

1. **P1.1** — Patch `common/speculative.cpp:757` `create_state_ngram_cache`: replace hardcoded `n_draft = 8` with `params.n_max`. Add `static_assert` or runtime check that `params.n_max > 0`.
2. **P1.2** — Build with the patch; verify in `common_speculative_init` that `n_draft` resolves to the `--draft-max` value.
3. **P1.3** — Smoke: `llama-server -m DSv4-Flash-AVG-16e --spec-type ngram-cache --draft-max 4` boots; `/health` returns ready.
4. **P1.4** — Verify `common_speculative_is_compat(ctx)` returns true for a TURBOMIND target context. If false, investigate — likely a buft-type check the compat function uses. Don't proceed until true.
5. **P1.5** — Two-model smoke (on a 2-GPU pod if needed): `llama-server -m AVG-16e -md AVG-16e -otd 'exps=CUDA_TURBOMIND'` boots both models; per-slot `common_speculative_init` succeeds.

**P1 Gate**:
- ✅ `ngram-cache` honors `--draft-max`
- ✅ `is_compat` returns true for TURBOMIND contexts
- ✅ Two-model boot smoke passes (may use 2-GPU pod or fallback if 256e setup not ready)

### P2 — Single-V100 AVG-16e + ngram-cache baseline (1.5 days)

This is both:
- The fallback primary if SPRINT-025 didn't ship
- The comparison row in REPORT-20 if 256e+draft is the primary

1. **P2.1** — Single V100 pod, `llama-server -m DSv4-Flash-AVG-16e -ngl 999 -ot 'exps=CUDA_TURBOMIND0' --spec-type ngram-cache --draft-max 8`.
2. **P2.2** — 10-prompt suite, 32-token greedy completions with `temp=0 top_k=1`. Capture `draft_n`, `draft_n_accepted`, `predicted_per_second` from completion timings.
3. **P2.3** — Same-output gate: rerun without `--spec-type`. Compare generated token IDs — must be identical.
4. **P2.4** — Acceptance: median across 10 prompts. Record per-prompt distribution.
5. **P2.5** — TPS comparison: with-spec vs without-spec decode TPS.

**P2 Gate**:
- ✅ Same-output: identical token IDs across 10 prompts
- ✅ Acceptance ≥ 0.30 median (else ngram-cache isn't worth shipping; document and skip P3 ngram-cache comparison)
- ✅ TPS lift ≥ 1.0 (otherwise spec overhead exceeds win)

### P3 — 8-GPU 256e + AVG-16e draft (3-4 days, primary if SPRINT-025 shipped)

1. **P3.1** — Pre-flight: run SPRINT-025's 8-GPU 256e config alone, capture per-GPU VRAM. Compute `-ts` fractions to free ~12-15 GiB on GPU 0 for the draft.
2. **P3.2** — Compose the command:
   ```bash
   llama-server -m /models/DSv4-Flash-256e-fixed.gguf \
                -md /models/DSv4-Flash-AVG-16e.gguf \
                -sm layer -ngl 999 -ot 'exps=CUDA_TURBOMIND' \
                -otd 'exps=CUDA_TURBOMIND0' \
                -ts <fractions-computed-in-P3.1> \
                --draft-max 8 --draft-min 4 --draft-p-min 0.75 \
                -c 4096 --cache-type-k q8_0 --cache-type-v q8_0 \
                --host 127.0.0.1 --port 12399
   ```
3. **P3.3** — Boot smoke. Verify per-GPU memory after load (all ≤ 30 GiB). Verify `common_speculative_is_compat` true.
4. **P3.4** — Same-output gate (CRITICAL): 10 prompts × 32-token greedy at `temp=0 top_k=1`. Compare with target-only (no `-md`). Must be identical token IDs.
5. **P3.5** — Acceptance + TPS: capture from completion timings.
6. **P3.6** — Per-GPU memory peak during decode.

**P3 Gate**:
- ✅ Both models load on 8 GPUs; all GPUs ≤ 30 GiB
- ✅ Same-output identity on 10 prompts
- ✅ Median acceptance recorded (any value; gate at P4)

### P4 — Acceptance + TPS sweep (2 days)

1. **P4.1** — Sweep `--draft-max ∈ {2, 4, 6, 8, 12, 16}`, fixed `--draft-min=2 --draft-p-min=0.5`. Record acceptance and TPS for each.
2. **P4.2** — Identify the knee where verify-pass overhead exceeds drafting gain.
3. **P4.3** — Per workload (chat vs code vs summarization) acceptance distribution.
4. **P4.4** — Optional: sweep `--draft-p-min ∈ {0.5, 0.75, 0.9}` at the knee `draft-max`.

**P4 Gate**:
- ✅ Knee identified; recommended config documented
- ✅ Acceptance verdict per tier (≥0.50 ship / 0.30-0.49 experimental / <0.30 STOP)

### P5 — REPORT-20 + CUDA-sampler regression measurement (2 days)

1. **P5.1** — Bench tables:
   - 256e single-V100 target-only (SPRINT-023 baseline)
   - 256e 8-GPU target-only (SPRINT-025 baseline)
   - 256e 8-GPU + spec at recommended config (SPRINT-026 deliverable)
   - AVG-16e + ngram-cache single-V100 (fallback comparison)
2. **P5.2** — **CUDA-backend-sampling regression cost**: with `--spec-type ngram-cache` configured, server-context.cpp forces `backend_sampling = false`. Quantify the per-token CPU-sampling round-trip overhead. Compare:
   - target-only with `backend_sampling=true` (default)
   - target-only with `backend_sampling=false` (forced via instrumentation hook)
   - target + spec at recommended config (`backend_sampling=false` by code path)
3. **P5.3** — Write `docs/sprints/SPRINT-026-REPORT-20.md`:
   - Primary path used (256e+draft or AVG-16e+ngram-cache)
   - All bench tables
   - Acceptance distribution
   - Per-GPU VRAM peak (if multi-GPU path)
   - MoE-determinism P0 logit-diff stats
   - CUDA-sampler-forced-off cost from P5.2
   - `# How to reproduce` shell block
   - Ship verdict per the acceptance tier
4. **P5.4** — Memory updates: `dsv4_flash_spec_decode_notes.md`.

**P5 Gate**:
- ✅ All bench rows captured
- ✅ Sampler regression measured
- ✅ REPORT-20 contains verdict + reproducible command

### P6 — Close-out (0.5 day)

1. Tag `sprint-026-close`.
2. Push to origin.
3. File `SPRINT-026-FOLLOWUPS.md` if anything emerged (likely: CUDA-side sampler implementation; multi-slot continuous batching from SPRINT-026 deferred).

---

## 5. Files Summary

### Modified

| Path | Change |
|---|---|
| `common/speculative.cpp` | One-line fix at line 757: replace `uint16_t n_draft = 8;` with `params.n_max`. |
| `docs/sprints/SPRINT-026-*-summary.md` | Per-phase summaries. |

### New

| Path | Purpose |
|---|---|
| `docs/sprints/SPRINT-026-REPORT-20.md` | Measurement narrative + verdict. |
| `docs/sprints/SPRINT-026-FOLLOWUPS.md` | If anything surfaces during execution. |
| `tests/test-moe-batch-determinism.cpp` | P0 smoke harness for decode-vs-verify argmax match. |

### NOT modified

- `tools/server/server-context.cpp` — already has all the draft loading + per-slot init + counters.
- `common/arg.cpp` — CLI surface is sufficient as-is.
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}` — no spec-decode-specific changes.
- `ggml/vendor/turbomind/api.cc` — no changes.

---

## 6. Definition of Done

1. ✅ Primary path executed: 8-GPU 256e + AVG-16e draft IF SPRINT-025 shipped; AVG-16e + ngram-cache otherwise. Path explicitly stated in REPORT-20.
2. ✅ `ngram-cache` honors `--draft-max` (1-line fix in `common/speculative.cpp:757` committed).
3. ✅ All CLI commands in DoD use upstream-correct flags (`--spec-type`, `--draft-max`, `-otd`, `--model-draft`, hyphenated values).
4. ✅ P0 MoE-determinism smoke: ≥99% argmax decode==verify on 5-prompt suite.
5. ✅ Same-output exactness gate: identical token IDs at `temp=0 top_k=1` across 10 prompts, target-only vs target+spec.
6. ✅ Acceptance: median rate measured and falls into one of the three tiers (≥0.50 ship / 0.30-0.49 experimental / <0.30 STOP); verdict explicit in REPORT-20.
7. ✅ TPS: with-spec vs without-spec measured for the primary path.
8. ✅ Per-GPU VRAM peak (multi-GPU path) ≤ 30 GiB on all 8 GPUs.
9. ✅ CUDA-sampler-forced-off cost quantified in P5.2.
10. ✅ `common_speculative_is_compat(ctx)` confirmed true for TURBOMIND contexts (P1).
11. ✅ ngram-cache and `--model-draft` exercised in separate runs (precedence rule respected).
12. ✅ No regression on SPRINT-025 baseline (single-GPU MIN-Ne tests still pass).
13. ✅ REPORT-20 contains reproducible command line(s).
14. ✅ Tag `sprint-026-close`.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | MoE batch-shape kicks argmax (decode vs verify) | Medium | High | P0 gate; if fails, reframe gate or force `--draft-max=1` |
| 2 | GPU 0 OOM with draft + target after rebalance | Medium-High | High | P3.1 pre-flight; `-ts` fraction tuning; q8_0 KV fallback; stop-loss to ngram-cache on 256e |
| 3 | SPRINT-025 hasn't shipped | Medium | Medium | P-1 dependency check; fallback to AVG-16e + ngram-cache single-V100 path |
| 4 | ngram-cache acceptance < 0.30 on mixed workload | Medium | Medium | Tiered acceptance gate; document and STOP if below floor |
| 5 | KV-cache rewind under TURBOMIND corrupts state on rejection | Low-Medium | High | P3.4 same-output gate transitively tests; instrument if fails |
| 6 | CUDA-sampler-disabled overhead eats >50% of spec win | Medium | Medium | P5.2 measures; SPRINT-027 implements CUDA sampler if material |
| 7 | `common_speculative_is_compat(ctx)` returns false for TURBOMIND | Low | High | P1.4 explicit gate; fix the compat check if false |
| 8 | Draft and draftless precedence rule confused, wrong path benched | Low | Medium | Separate server runs; document in REPORT-20 |
| 9 | -ts fractions can't free enough room for draft + target on GPU 0 | Medium | Medium | Stop-loss: ship ngram-cache on 256e (no extra VRAM) |
| 10 | Effort underestimated (per `feedback_effort_estimation_undocumented_hardware`) | High | Medium | 12-15 day budget vs Codex's 8-day estimate |
| 11 | --cpu-moe-draft alone doesn't actually keep draft off GPU | Low | Medium | Use `--device-draft none` and `-ngld 0` explicitly if CPU-draft attempted |
| 12 | First ~64 tokens of conversation have cold ngram cache, biasing benchmark | Medium | Low | Warm-up the cache with a non-measured run; document the cold-start TPS too |

---

## 8. Security

No new public-facing service. No new credentials. No new file-system surfaces beyond model GGUFs (already read-only mounted).

- `--lookup-cache-static` and `--lookup-cache-dynamic` are out of scope for the primary path (no static cache file written).
- Speculative decoding doesn't introduce new attack surface vs default decode.
- Model files mounted RO from `/models`; no write access for the draft path beyond ephemeral workspace.

---

## 9. Dependencies

1. **SPRINT-025 (multi-GPU 256e landing)** — hard dependency for primary path. Soft fallback (single-V100 ngram-cache) if absent.
2. **SPRINT-023 + SPRINT-024 code** — `CUDA_TURBOMIND` buft, per-device TmLib (if SPRINT-025 P2 ran), pack pipeline.
3. **Model files**:
   - `/models/DSv4-Flash-256e-fixed.gguf` (target, 156 GiB)
   - `/models/DSv4-Flash-AVG-16e.gguf` (draft, 18 GiB)
4. **Hardware**: 8-GPU pod on `gpu-01` (`llamacpp-build-8gpu`) per SPRINT-025 P0.
5. **Build environment**: CUDA 12.2 / NCCL 2.18+ / cmake 3.22 / gcc 11.4.

---

## 10. Open Questions

1. **n-gram cache cold-start**: how to fairly benchmark when the cache is empty at request 0 vs warm after request 5? P5 may need both cold and warm rows.
2. **Lookup-cache persistence** (`--lookup-cache-static`): out of scope; revisit if SPRINT-027 needs deterministic warm-cache benchmarks.
3. **Should P3 try `--draft-max=1` (single-token spec) as a determinism-safe baseline if P0 logit-diff is borderline?** Adds a row to the sweep but mitigates the MoE-routing risk.
4. **Per-prompt acceptance variance** — should DoD include a per-prompt acceptance histogram or just median? Histogram more informative; median is the headline.
5. **Server-side fork bomb**: with `--parallel-slots N > 1` AND spec decode, do we have N draft contexts? Out of scope for SPRINT-026 (single-slot focus) but flag for SPRINT-027 multi-slot work.

---

## 11. Outcome contract

This sprint succeeds if:

- P0 MoE-determinism gate passes (≥99% argmax match) OR a mitigation is in place (`--draft-max=1`).
- P3 same-output gate passes (identical token IDs).
- P4 yields a definitive acceptance verdict in one of the three tiers.
- REPORT-20 captures the verdict + reproducible command line.

This sprint can STOP without shipping if:

- P0 fails decisively (no mitigation possible for MoE routing).
- P3 VRAM OOM with no `-ts` fraction that fits.
- P4 acceptance < 0.30 even with the optimal `--draft-max` setting.

In all STOP cases, the close-out doc must name the specific blocker.

# SPRINT-026 — Speculative decoding for DSv4-Flash on V100 turbomind path

**Status:** DRAFT 2026-05-15 (Claude independent)
**Predecessor:** SPRINT-025 (multi-GPU 256e — planned)
**Successor:** TBD (likely row-TP or quality eval)

---

## 1. Overview

SPRINT-023 P5 measured 16.2–16.6 t/s decode flat across MIN-Ne fixtures on V100 — the path is launch-bound at M=1. SPRINT-024 attacks that via grouped MoE (per-launch cost). SPRINT-025 attacks model fit via 8-GPU layer split. **SPRINT-026 attacks the per-token dimension itself** by drafting K candidate tokens with a small model and verifying them in a single batched forward pass on the target. With acceptance rate `α` and draft length `K`, effective decode TPS scales by approximately `(1 + α·K) / (1 + verify-overhead)`. At α=0.5, K=4, verify-overhead ≈ 0.15 that is roughly a **2.6× lift** before any other work.

The win lands by **wiring existing llama.cpp infrastructure** rather than authoring new spec-decode logic:

1. `common/speculative.{h,cpp}` already implements `common_speculative_init/free/gen_draft/accept`, plus the type table `common_speculative_types[]` (`none`, `draft`, `ngram_simple`, `ngram_map_k`, `ngram_map_k4v`, `ngram_mod`, `ngram_cache`).
2. `tools/server/server-context.cpp:661–694` already loads a separate `model_dft` when `--model-draft` is supplied, applying `params_spec.tensor_buft_overrides` (so `-otd 'exps=CUDA_TURBOMIND'` Just Works for the draft).
3. `server-context.cpp:338–340` plus `:394–398` already track acceptance counters (`n_draft_total`, `n_draft_accepted`) and log the rate at slot release.
4. `common/arg.cpp` already defines `-md`/`--model-draft` (3503), `-otd`/`--override-tensor-draft` (2298), `-cmoed`/`--cpu-moe-draft` (2326), `--draft-max` (3440), `--draft-min` (3447), `--draft-p-min` (3461), `-lcs`/`--lookup-cache-static` (1234), `-lcd`/`--lookup-cache-dynamic` (1241).

**This sprint adds zero new spec-decode primitives.** It picks the right draft, validates the same-tokenizer invariant on DSv4-Flash family, plumbs the draft onto a second TURBOMIND context that coexists with the target on one V100, and ships REPORT-20 with measured acceptance + TPS uplift.

### Perf framing

Target: **decode TPS ≥ 1.3× single-slot baseline** on a 10-prompt mixed chat/code set at temp=0, with **median acceptance ≥ 0.50**. Below 0.30 acceptance the verify overhead exceeds the draft win and we stop-and-explain. Stretch: ≥ 2.0× at α ≥ 0.65.

### Strategy decision (locked at draft, revisited at P0)

Two paths land in this sprint, measured side by side, ship the winner:

| Path | What | Cost | Win |
|---|---|---|---|
| **A. Draft-model** | DSv4-Flash-AVG-16e (real weights, 18 GiB) drafts for the same target on a different model size — see §3.2 for target choice | Doubles VRAM; needs second TM context | Higher α (typical 0.5–0.7 for distilled draft) |
| **B. N-gram cache** | `--speculative-type ngram_cache` + `-lcd dynamic.cache` | Zero VRAM, zero second model | α surprisingly competitive on code/repetitive prompts (0.3–0.5); pure floor on free-form chat |

Both paths land. REPORT-20 publishes both numbers. **Default config in the sprint close-out is whichever path won the geomean uplift across the prompt set.**

### What this sprint is NOT

- New spec-decode algorithm research (no Eagle, no Medusa, no tree attention).
- Custom draft training. AVG-16e is what we have.
- Modifications to `common/speculative.{h,cpp}` (upstream surface; private-fork stance).
- Multi-slot parallel sequences (the other branch of "multi-slot decode"). Selected branch is speculative.
- Production server hardening beyond what `tools/server` already exposes.

---

## 2. Use Cases

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | Draft + target tokenizer-equivalence verified; VRAM headroom for 2-model load measured; baseline target-only TPS locked. |
| P1 | `llama-server -md` boots both models with `-ot exps=CUDA_TURBOMIND` on each, completes one /completion request without crash. |
| P2 | Same-output gate at temp=0 passes for 10 fixed prompts (target-only ≡ target+draft byte-identical). |
| P3 | Acceptance + TPS measured on draft path; tuning sweep (`--draft-max`, `--draft-p-min`) recorded. |
| P4 | N-gram cache path measured; static + dynamic cache compared; head-to-head with draft. |
| P5 | REPORT-20 written; SHIP / EXTEND / STOP verdict; default config selected; followups for SPRINT-027. |
| P6 | Tag, memory updates, draft model registered in operational notes. |

---

## 3. Architecture

### 3.1 The same-vocab invariant (hard constraint)

`common_speculative_init` enforces vocabulary equivalence between target and draft contexts. From the `deepseek4` model arch (all DSv4-Flash variants share a single `tokenizer.ggml.model` and merges table — verified at upload-time hash match in SPRINT-022), MIN-*, AVG-16e, IQ2-64e, and 256e all pass this check.

**Out-of-family candidates rejected up-front:**
- Qwen-* (different BPE; vocab size differs).
- Any deepseek-coder-v* (separate vocab).
- Hand-built distilled drafts (would require training).

**In-family candidates** (verified per intent doc §2):
- DSv4-Flash-AVG-16e (18 GiB, real weights — only viable real-weight smaller draft on disk).
- DSv4-Flash-IQ2-64e (28 GiB, real weights).
- DSv4-Flash-MIN-* (random expert weights, per `dsv4_flash_min_models_are_garbage.md` memory — useless as drafts; α would be ~0%).

P0.2 verifies vocab match programmatically: `llama_model_get_vocab(model_target)` vs `llama_model_get_vocab(model_dft)`, compare `n_vocab`, BOS/EOS, `add_bos_token`, and a 100-token round-trip on a chat string.

### 3.2 Draft-model + target choice

| Combo | VRAM | Why |
|---|---|---|
| **Target = AVG-16e, Draft = AVG-16e** | 36 GiB (OOM on one V100) | Self-spec; α should be ~1.0 but defeats purpose (no perf win). REJECT. |
| **Target = IQ2-64e, Draft = AVG-16e** | 46 GiB (OOM single GPU) | Needs SPRINT-025 multi-GPU. Stretch only. |
| **Target = IQ2-64e on TM, Draft = AVG-16e on CPU** | 28 GiB GPU + 18 GiB host | Draft on CPU at ~3 t/s × K=4 = 12 ms/draft-step. Verifiable. **Primary single-GPU candidate.** |
| **Target = AVG-16e on TM, Draft = AVG-16e on CPU** | 18 GiB GPU + 18 GiB host | Same-model self-spec. Useful for proving plumbing only. |
| **Target = 256e (multi-GPU) on 8× V100, Draft = AVG-16e on GPU 0** | requires SPRINT-025 ship | Stretch. |

**Selected primary: Target = IQ2-64e on `CUDA_TURBOMIND0`, Draft = AVG-16e on CPU MoE (`-otd 'exps=CPU'` or `--cpu-moe-draft`).** Rationale:
1. Both real-weight, same family → tokenizer match guaranteed.
2. Draft-on-CPU sidesteps the VRAM doubling that kills single-GPU two-model loads. AVG-16e on CPU MoE was the SPRINT-022 baseline path — known to work, known throughput.
3. Target on TURBOMIND uses the SPRINT-023 P4 dispatch path that's already shipping. No new dispatcher work.

**Stretch primary (only if SPRINT-025 lands during 026):** Target = 256e layer-split on 8 GPUs, Draft = AVG-16e pinned to GPU 0 with `-otd 'exps=CUDA_TURBOMIND0' --device-draft CUDA0`. This is the full headline configuration but requires multi-GPU plumbing not yet shipped.

### 3.3 CLI wiring (no new flags)

The standard server invocation for the primary path:

```
llama-server \
  -m /models/DSv4-Flash-IQ2-64e.gguf \
  -ot 'exps=CUDA_TURBOMIND0' \
  -ngl 999 -np 1 -c 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -md /models/DSv4-Flash-AVG-16e.gguf \
  --cpu-moe-draft \
  --speculative-type draft \
  --draft-max 4 --draft-min 1 --draft-p-min 0.75 \
  --host 127.0.0.1 --port 12399
```

For the n-gram path (no second model):

```
llama-server \
  -m /models/DSv4-Flash-IQ2-64e.gguf \
  -ot 'exps=CUDA_TURBOMIND0' \
  -ngl 999 -np 1 -c 4096 \
  --speculative-type ngram_cache \
  -lcd /workspace/dsv4-dyn.cache \
  --draft-max 4 --draft-min 1 \
  --host 127.0.0.1 --port 12399
```

**No code changes required for either invocation.** All flags exist in `common/arg.cpp` today (verified line numbers above).

### 3.4 Acceptance counter location

`server-context.cpp:164–165` declares per-slot `n_draft_total` and `n_draft_accepted`. They are reset on slot release (`:186–188`), updated inside the draft-verify path (per-slot, in the speculative call site that consumes `common_speculative_gen_draft` results), exported into `timings.draft_n` / `timings.draft_n_accepted` at `:338–341`, and logged at `:394–398`:

```
draft acceptance rate = %0.5f (%5d accepted / %5d generated)
```

Per-prompt acceptance is read from server logs after each completion, OR from the `timings` block in the JSON response (the `--metrics` flag exposes them). REPORT-20 uses both.

`common_speculative_print_stats(spec)` at `:402` dumps additional internal counters at slot release; capture into the per-prompt log line.

### 3.5 VRAM accounting with two models loaded

For the primary path (target IQ2-64e on TM, draft AVG-16e on CPU):

| Line item | Bytes (GiB) |
|---|---:|
| Target weights (IQ2-64e on TURBOMIND0) | ~28 |
| Target KV at c=4096, q8_0 | ~0.5 |
| Target activations + scratch | ~1.5 |
| Turbomind workspace (`d_barriers + d_partials + d_flags`) | ~0.5 |
| Allocator fragmentation | ~1 |
| **Per-GPU subtotal** | **~31.5 / 32** |
| Draft weights (AVG-16e on host) | 18 (host) |
| Draft KV at c=4096, q8_0 (host) | 0.5 (host) |

This is **tight on a 32 GiB V100**. P1.4 stop-the-line: if peak load exceeds 30 GiB, reduce `-c` to 2048, drop to no-q8_0-KV-fallback only if necessary. P0.3 measures this empirically before any draft model loads.

For the stretch (256e target + AVG-16e draft on GPU 0): the draft adds ~18 GiB to GPU 0's load. Per `SPRINT-025` §3.6, GPU 0 is already the heaviest shard (embedding + lm_head + layers ≈ 22 GiB worst case). Add 18 GiB → over 32 GiB budget → **draft must move to a different device** (e.g. GPU 7) or shard across two devices. Not in scope unless SPRINT-025 ships first.

### 3.6 Same-output invariance (the spec-decode contract)

Speculative decoding at `temp=0` is *exact*: the draft proposes K tokens, the target accepts the longest prefix that matches its own greedy choice. Any divergence at temp=0, fixed seed, fixed context is a bug — not a quantization artifact, not a quirk of the draft.

P2 verifies bit-identity across three configurations on 10 fixed prompts:
- `target-only` (no `-md`, no `--speculative-type`)
- `target + draft` (path A from §3.2)
- `target + ngram_cache` (path B)

For each prompt, capture the exact token-id sequence at `--seed 0 --temp 0 -n 64`. Compare via `diff` (or python set comparison). All three sequences MUST be identical for every prompt. **One mismatch → halt and root-cause.**

Likely failure mode if it happens: draft model has a subtly different sampler config (e.g. `add_bos_token` differs even though vocab matches). Diagnostic: log `llama_vocab_get_add_bos`, `llama_vocab_bos`, `llama_vocab_eos` for both models at startup.

### 3.7 Why the TURBOMIND interaction is not free (hidden surface)

The `dlopen` of `libggml-turbomind.so` happens once per process. With the primary path the target uses TURBOMIND but the draft uses CPU MoE — only one TURBOMIND instance, no per-device-state issue. **Multi-instance TURBOMIND (target and draft both on different GPUs) hits the same singleton bug that SPRINT-025 P2 fixes.** If SPRINT-025 P2 hasn't shipped, the stretch multi-GPU configuration is blocked. Single-GPU-CPU-draft is the safe path.

---

## 4. Implementation

### P0 — Tokenizer equivalence + VRAM budget + baseline lock (1 day)

**Goal:** answer the three "is this even possible" questions before plumbing.

1. **P0.1** — `llama-bench -p 128 -n 32 -r 3` on IQ2-64e single-GPU with `-ot 'exps=CUDA_TURBOMIND0'`. Lock target-only baseline TPS for the uplift comparison. Halt if it doesn't reproduce SPRINT-023 expected range (~14–17 t/s decode for IQ2-64e on TM).
2. **P0.2** — Tokenizer equivalence harness `tools/spec-decode/check-vocab.cpp` (new, ≤ 100 lines): load IQ2-64e and AVG-16e via `llama_model_load_from_file`, fetch vocabs, assert `n_vocab` matches, BOS/EOS match, and a fixed 100-token chat string round-trips identically. Halt if any mismatch.
3. **P0.3** — Two-model load smoke (no spec yet): boot `llama-server` with `-m IQ2-64e -md AVG-16e --cpu-moe-draft --speculative-type none`. Verify both models initialize. Capture peak GPU + host memory via `nvidia-smi --query-gpu=memory.used` and `/proc/self/status` snapshot. Halt if GPU memory > 30 GiB.
4. **P0.4** — Standalone `llama-speculative` binary smoke (already builds). Run with the two-model pair and a single prompt; verify it produces output and acceptance counters. This isolates the spec-decode core from the server's slot machinery.

**P0 Gate:**
- ✅ Target-only baseline TPS reproduced within 5%
- ✅ Vocab match verified
- ✅ Two-model load fits under 30 GiB GPU
- ✅ `llama-speculative` smoke produces non-zero acceptance

### P1 — Server two-model boot + functional smoke (1 day)

**Goal:** the production CLI path works.

1. **P1.1** — `llama-server` boot with the §3.3 primary invocation. Wait for `/health` to return ready. Document boot log lines that mention "draft" or "speculative" — what the server prints when both models are live.
2. **P1.2** — Single `/completion` POST with `temperature=0`, `seed=0`, `n_predict=32`, prompt `"def fibonacci(n):"`. Verify the response includes `timings.draft_n` and `timings.draft_n_accepted`. Verify both > 0.
3. **P1.3** — N-gram path boot: same setup but `--speculative-type ngram_cache -lcd /tmp/dsv4-dyn.cache`. Same single-completion smoke. The dynamic cache file gets created mid-run; verify it appears.
4. **P1.4** — VRAM stop-the-line check: monitor `nvidia-smi dmon -s u -d 1` during a 64-token completion. Peak must stay under 30 GiB. If it exceeds, drop `-c` from 4096 to 2048 and re-test; record the binding constraint in REPORT-20.

**P1 Gate:**
- ✅ Both invocations boot cleanly
- ✅ One /completion returns valid tokens with non-zero `draft_n`
- ✅ Per-GPU peak ≤ 30 GiB
- ✅ `/health` reports ready

### P2 — Same-output gate at temp=0 (1 day)

**Goal:** prove the spec-decode contract holds end-to-end.

1. **P2.1** — Author `tools/spec-decode/prompt-set-10.txt` (new): 10 fixed prompts, mix of 4 chat / 4 code / 2 short-math. Each ≤ 256 input tokens.
2. **P2.2** — Author `tools/spec-decode/run-prompts.sh` (new, ≤ 80 lines): for each prompt, POST to `/completion` with `temperature=0, seed=0, n_predict=64`. Capture full token-id arrays from the JSON response. Output one file per prompt per config: `out/prompt-{i}-{cfg}.tokens.json`.
3. **P2.3** — Run the harness against three configs:
   - `target-only` (no `-md`)
   - `target + AVG-16e draft on CPU`
   - `target + ngram_cache`
4. **P2.4** — Diff: a small `tools/spec-decode/compare-tokens.py` script that reads the three sets of 10 files and asserts pairwise equality on token-id arrays.
5. **P2.5** — Halt-and-root-cause if any prompt diverges. Most-likely root cause: BOS handling difference; second-most: top-1 ties broken differently between draft + verify path. Document the bug, fix, re-run.

**P2 Gate:**
- ✅ All 10 prompts produce IDENTICAL token-id arrays across all 3 configs
- ✅ No NaN/Inf in target logits at sampled positions

### P3 — Acceptance + TPS measurement on draft path (1–2 days)

**Goal:** the headline draft-model number.

1. **P3.1** — Run the §3.3 primary invocation against the prompt set. For each prompt, capture from `/completion` JSON `timings`:
   - `predicted_per_second` (decode TPS)
   - `draft_n_accepted / draft_n` (acceptance rate)
   - Wall time
   Aggregate: median acceptance, p25/p75; mean decode TPS, p25/p75.
2. **P3.2** — Sweep `--draft-max ∈ {2, 4, 6, 8}` × `--draft-p-min ∈ {0.6, 0.75, 0.85}`. 12 (max, p_min) cells × 10 prompts = 120 runs. Record acceptance + TPS per cell. Heatmap goes in REPORT-20.
3. **P3.3** — Identify the (max, p_min) Pareto winner on geomean TPS uplift. Confirm acceptance ≥ 0.50 there (else re-evaluate the floor — is 0.30 with very low verify overhead net-positive? `verify-overhead = K · per-token-target-cost - 1 · per-token-target-cost`; document the breakeven).
4. **P3.4** — Capture decode TPS at the winner cell vs P0.1 baseline. **Pass: decode TPS ≥ 1.3× baseline.** Stretch: ≥ 2.0×.

**P3 Gate:**
- ✅ 120-cell sweep complete; results in REPORT-20 table
- ✅ Median acceptance ≥ 0.50 at the winner cell, OR documented break-even justification at lower acceptance
- ✅ Decode TPS at winner ≥ 1.3× target-only baseline

### P4 — N-gram cache measurement (1 day)

**Goal:** the draft-free comparison number.

1. **P4.1** — Pre-build a static cache by running 100 representative completions with `-lcd dynamic.cache`, then hand the resulting cache to a fresh server boot via `-lcs static.cache` for the measurement run.
2. **P4.2** — Run the prompt set with `--speculative-type ngram_cache -lcs static.cache`. Capture acceptance + TPS as in P3.1.
3. **P4.3** — Repeat with `--speculative-type ngram_cache -lcd dynamic.cache` (live-build cache). Compare static vs dynamic.
4. **P4.4** — Sweep `--draft-max ∈ {2, 4, 6, 8}` (skip p_min — n-gram lookup uses different acceptance criterion).

**P4 Gate:**
- ✅ Both static + dynamic n-gram numbers captured
- ✅ Comparison table draft vs ngram_cache vs target-only in REPORT-20

### P5 — REPORT-20 + SHIP/EXTEND/STOP verdict (1 day)

1. **P5.1** — Author `docs/sprints/SPRINT-026-REPORT-20.md`. Sections:
   - Executive: target-only TPS, draft-path TPS, ngram-path TPS, winner
   - Acceptance + TPS table per (config, prompt)
   - Acceptance heatmap over (draft-max, draft-p-min)
   - VRAM table: per-GPU peak load + decode for both configs
   - Same-output gate: PASS confirmation with token-id diff harness invocation
   - SHIP / EXTEND / STOP verdict with rationale
   - Default-config recommendation (which `--speculative-type` ships)
   - Reproducible command lines
   - Followups for SPRINT-027 (likely: tree-attention, multi-slot, or row-TP)
2. **P5.2** — File `docs/sprints/SPRINT-026-FOLLOWUPS.md` with deferred items.

**P5 Gate:**
- ✅ REPORT-20 contains both numbers + verdict
- ✅ Followups doc exists

### P6 — Close-out (0.5 day)

1. If SHIP: tag `sprint-026-close`. Memory update: `dsv4_flash_spec_decode_landed.md`.
2. Update operational runbook with the winner config.
3. Push to origin.

### P7 — Stretch: multi-GPU 256e + draft (only if SPRINT-025 ships during 026)

1. **P7.1** — Boot 256e on 8 GPUs (per SPRINT-025 §4) + AVG-16e draft pinned to GPU 0 via `-otd 'exps=CUDA_TURBOMIND0'`. Verify it loads. Verify GPU 0 stays under 30 GiB.
2. **P7.2** — Repeat P3.1 on the prompt set. Capture acceptance + TPS for the headline 256e+spec configuration.
3. **P7.3** — Append as a REPORT-20 stretch section.

---

## 5. Files Summary

### New

| Path | Purpose |
|---|---|
| `tools/spec-decode/check-vocab.cpp` | P0.2 vocab equivalence harness |
| `tools/spec-decode/prompt-set-10.txt` | The 10 fixed prompts |
| `tools/spec-decode/run-prompts.sh` | Posts to `/completion`, captures token JSON |
| `tools/spec-decode/compare-tokens.py` | Pairwise token-id equality check |
| `tools/spec-decode/CMakeLists.txt` | Builds `check-vocab` |
| `docs/sprints/SPRINT-026-REPORT-20.md` | Measurement narrative + verdict |
| `docs/sprints/SPRINT-026-P{0..6}-summary.md` | Per-phase summaries |
| `docs/sprints/SPRINT-026-FOLLOWUPS.md` | Deferred items for SPRINT-027 |

### Modified

| Path | Change |
|---|---|
| (none in `common/`, `tools/server/`, `ggml/` — sprint is wiring-only) | All flags exist; all server plumbing exists |

### Possibly modified (P2.5 conditional, only if same-output gate fails)

| Path | Change |
|---|---|
| `tools/server/server-context.cpp` or `common/speculative.cpp` | Diagnostic logging only; do NOT modify spec-decode core (private-fork stance, see Constraints) |

---

## 6. Definition of Done

1. ✅ DSv4-Flash-AVG-16e and DSv4-Flash-IQ2-64e vocab equivalence verified by `check-vocab` harness.
2. ✅ `llama-server -m IQ2-64e -md AVG-16e --cpu-moe-draft --speculative-type draft -ot 'exps=CUDA_TURBOMIND0'` boots on a single V100, completes one /completion request without crash.
3. ✅ Per-GPU peak VRAM ≤ 30 GiB for the primary single-GPU two-model config.
4. ✅ Same-output gate: 10 fixed prompts at temp=0, seed=0, n_predict=64 produce IDENTICAL token-id arrays for {target-only, target+draft, target+ngram_cache}.
5. ✅ Acceptance + TPS measured on the draft path; (`--draft-max`, `--draft-p-min`) sweep recorded.
6. ✅ Decode TPS ≥ 1.3× target-only baseline at the winning (max, p_min) cell, OR documented break-even rationale at lower acceptance.
7. ✅ Median acceptance ≥ 0.50 at the winning cell (≥ 0.30 hard floor; below that the sprint STOPs and explains).
8. ✅ N-gram cache path measured (static + dynamic); compared head-to-head with draft-model path.
9. ✅ REPORT-20 contains: per-prompt acceptance + TPS table, sweep heatmap, VRAM table, same-output gate confirmation, SHIP/EXTEND/STOP verdict, default-config recommendation, reproducible command lines.
10. ✅ Followups doc filed for SPRINT-027.
11. ✅ Tag `sprint-026-close`.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | AVG-16e vocab subtly differs from IQ2-64e despite same arch | Low | High | P0.2 hard gate; can't proceed without it |
| 2 | Two-model load OOMs on single 32 GiB V100 | Medium | High | P0.3 measures pre-flight; fall back to smaller `-c` or to multi-GPU draft |
| 3 | Same-output gate fails (subtle BOS / sampler config divergence) | Medium | High | P2.5 root-cause; do NOT mask by enabling temp > 0 in the gate |
| 4 | Acceptance < 0.30 across the prompt set (draft too distant from target) | Medium | High | Stop-and-explain in REPORT-20; pivot to ngram_cache only |
| 5 | Draft on CPU too slow (decode latency dominated by draft generation) | Medium | Medium | P3 measures; consider `--draft-max 2` to limit per-step CPU cost |
| 6 | TURBOMIND singleton bug bites if both models accidentally pin to TURBOMIND on same/different GPU | Low (single-GPU primary path) | High | Primary path uses CPU draft; stretch requires SPRINT-025 P2 fix |
| 7 | n-gram cache produces NaN or wrong tokens under temp=0 | Low | High | P2 same-output gate covers this; cache is internal so no leak path |
| 8 | Spec-decode adds latency variance that p99 regresses even when median improves | Medium | Low | REPORT-20 captures p25/p75 + p99; ship if p99 doesn't regress > 10% |
| 9 | `--draft-p-min 0.75` is the wrong default for our top-1 distribution | Medium | Low | P3.2 sweep finds the right value |
| 10 | `llama-speculative` standalone binary diverges from `llama-server` spec path | Low | Low | P0.4 only uses `llama-speculative` for isolation; ship verdict comes from server numbers |
| 11 | IQ2-64e quality is too low to be a meaningful "production target" | Medium | Medium | Document as "demonstration target"; AVG-16e + AVG-16e self-spec smoke as fallback proof of plumbing |
| 12 | dlopen of libggml-turbomind.so happens twice (once per llama_context) and corrupts state | Low | High | TURBOMIND init is per-process not per-context; verify via `GGML_TM_VERBOSE=1` log |

---

## 8. Security

Local-only single-node run; no internet-facing service; no new credential surface.

- `--lookup-cache-static` / `--lookup-cache-dynamic` files are read/written under the user's workspace; no path traversal surface beyond what `tools/server` already exposes.
- The spec-decode draft model is loaded from the same `/models/` PVC as the target; mounted read-only (per SPRINT-025 §8).
- No new network sockets opened; standard `llama-server --host 127.0.0.1 --port 12399`.
- Prompt set files are local; no exfiltration path.
- The two-model setup roughly doubles the in-memory tensor data; document this in REPORT-20 §VRAM but no security implication on a private host.
- N-gram cache file is plaintext token-id sequences derived from prompts; treat as sensitive if prompts are sensitive (out of scope for this sprint's prompt set).

---

## 9. Dependencies

1. **SPRINT-022 baseline path** — DSv4-Flash-AVG-16e on CPU MoE works (already verified at SPRINT-022 P5).
2. **SPRINT-023 turbomind integration** — `CUDA_TURBOMIND` buffer type, pack pipeline, dispatch (per-expert path is sufficient; SPRINT-024 grouped path is NOT a prerequisite).
3. **`llama.cpp` upstream spec-decode infrastructure** — `common/speculative.{h,cpp}`, `common_speculative_types[]`, `--model-draft` server plumbing — verified present at HEAD.
4. **DSv4-Flash-AVG-16e GGUF (18 GiB) at `/models/`** — verified per intent doc.
5. **DSv4-Flash-IQ2-64e GGUF (28 GiB) at `/models/`** — verified per intent doc.
6. **Single V100-SXM2-32GB on `gpu-01`** — already provisioned for SPRINT-023.
7. **CUDA 12.2 / gcc 11.4 / cmake 3.22** — same as SPRINT-023.
8. **NOT a dependency:** SPRINT-024 grouped MoE (orthogonal), SPRINT-025 multi-GPU (only blocks the stretch P7).

---

## 10. Open Questions

1. **Draft-strategy default for ship.** If both paths land at α ≥ 0.5 and TPS ≥ 1.3×, which becomes the default config in operational notes — draft-model (higher α, doubles VRAM) or n-gram (zero-cost-VRAM, lower α)? P5.1 picks based on geomean TPS across prompt mix.
2. **Acceptance floor.** Intent doc proposes ≥ 0.50 median target, ≥ 0.30 stop-loss. Do we want a stricter floor for draft-path (which costs VRAM) than for n-gram-path (which costs nothing)? Suggested default: same 0.50 median for both, 0.30 hard stop for both — let TPS uplift do the discrimination.
3. **Prompt mix composition.** 4 chat / 4 code / 2 math is a guess. Should we bias toward code (where n-gram historically excels) or chat (more representative production traffic)? Decide in P0 based on what production traffic actually looks like — for now, even split.
4. **Stretch into 256e multi-GPU.** Hard depend on SPRINT-025 ship date. If SPRINT-025 lands during 026 execution, P7 fires; otherwise it doesn't. No half-pursuit.
5. **`--draft-max` upper bound.** Sweep goes to 8. Beyond that, verify-batch becomes large enough that target-side memory bandwidth on the verify pass becomes the new bottleneck. P3 documents the inflection point if visible in the 2/4/6/8 sweep.
6. **What if AVG-16e is "too good" a draft and α saturates near 1.0?** Then we're effectively running AVG-16e quality with IQ2-64e cost. Useful, but not the point of spec-decode (cost should drop, not rise). Document as a finding; consider a smaller draft (none available without training) or a deeper prompt set that exposes the quality gap.
7. **CUDA-graph capture compatibility.** `GGML_CUDA_USE_GRAPHS` on the verify-batch path: does it engage? Probably yes (single forward pass, well-defined shape) but verify in REPORT-20 §runtime-flags.

---

## 11. Outcome contract

Sprint ships if:
- Same-output gate passes (10/10 prompts identical at temp=0).
- Either draft-path OR n-gram-path delivers decode TPS ≥ 1.3× single-slot baseline at acceptance ≥ 0.50.
- VRAM peak ≤ 30 GiB on the chosen primary single-GPU configuration.
- REPORT-20 contains both numbers + verdict + default-config recommendation.

Stop-loss: if both paths fail the 1.3× / 0.50 gate, the sprint EXTENDs (search for a better (`--draft-max`, `--draft-p-min`) cell) or STOPs (publish the negative result with full diagnostic; SPRINT-027 reconsiders the strategy — likely tree-attention or distilled-draft training, both out of single-sprint scope).

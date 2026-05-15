# SPRINT-026 — Merge notes

**Date:** 2026-05-15

## Draft comparison

### Claude draft (28 KB / 400 lines)
- Most technically detailed; correctly maps `server-context.cpp` counters, real-weight constraint, and same-tokenizer rule.
- **Strong**: VRAM tables, IQ2-64e + AVG-16e draft sizing, exactness gate `temperature=0, top_k=1, seed=0`.
- **Wrong**: CLI flag names — uses `--speculative-type`, `ngram_cache` (underscore), `-od`, `--n-draft` — none of those exist upstream.
- **Wrong**: "Draft on CPU" via `--cpu-moe-draft` doesn't actually keep the entire draft model off GPU; only MoE experts. Need `--device-draft none` and `-ngld 0`.

### Codex draft (18 KB / 374 lines) — SPINE
- **Strong**: Correct CLI flag surface (`--spec-type`, hyphenated values, `-otd`, `--draft-max`).
- **Strong**: Caught the `ngram-cache` hardcoded `n_draft = 8` bug in `common/speculative.cpp:757` — only documented code-level blocker.
- **Strong**: Decoupled from SPRINT-025; single-V100 AVG-16e + ngram-cache as primary.
- **Strong**: Decision-completeness ship/experimental/STOP bands.
- **Weak**: Calls ngram-cache "self-speculative" — it's draftless, not self-spec.
- **Weak**: 8-day estimate is optimistic per the effort-estimation memory (multiply by 3 for undocumented hardware interactions).
- **Weak**: Doesn't notice `backend_sampling` forced off under spec mode (`server-context.cpp:1197-1198`).
- **Weak**: Doesn't flag MoE routing batch-shape nondeterminism risk (verify batch=K+1 vs decode batch=1).

### Gemini draft (9.2 KB / 188 lines)
- **Wrong**: CLI flag names (same errors as Claude — `--speculative-type`, `ngram_cache`, `-od`, `--n-draft`).
- **Wrong**: Uses `MIN-8e` as draft candidate (random weights — useless).
- **Wrong**: Primary path is 256e + 16e draft on 8-GPU, which requires SPRINT-025 to ship first (the intent explicitly says single-GPU first).
- **Wrong**: VRAM math hand-wavey ("100 GiB cluster slack").
- **Wrong**: Bare `-ot 'exps=CUDA_TURBOMIND'` without GPU index.
- **Strong**: Risk table format; same-output gate phrasing.

## Critiques accepted / rejected

| Critique | Source | Verdict |
|---|---|---|
| CLI flag names: `--spec-type` (not `--speculative-type`); hyphenated values; `-otd` (not `-od`); `--draft-max` (not `--n-draft`) | All three critiques | **Accepted** — final uses correct upstream surface |
| ngram-cache hardcoded `n_draft=8` is a real bug | Codex | **Accepted** — final mandates the 1-line fix in P1 |
| `backend_sampling` forced off under spec (server-context.cpp:1197) | Claude | **Accepted** — P5 follow-on measurement per user |
| MoE routing batch-shape nondeterminism (decode batch=1 vs verify batch=K) | Claude | **Accepted** — P0 sanity check gate per user |
| `common_speculative_is_compat` smoke check on TURBOMIND contexts | Claude | **Accepted** — P1 functional gate |
| KV-cache rewind under TURBOMIND when spec rejects | Claude | **Accepted** as Risk row, gate via P3 exactness test |
| MIN-* models as drafts | Gemini draft used MIN-8e | **Rejected** — random weights, AVG-16e is the only acceptable draft |
| Self-speculative terminology for ngram-cache | Codex | **Rejected** — final uses "draftless speculation via ngram-cache" |
| Two-model single-V100 self-draft | Codex correctly rejects | **Rejected** as primary; only AVG-16e self-draft on 2 V100s is feasible |
| "Mathematically exact at temp=0" claim | Gemini | **Rejected** — final requires `temperature=0, top_k=1` AND the P0 MoE-determinism gate |
| Exactness gate: bit-identical token IDs | Codex+Claude | **Accepted** — phrased as "identical generated token IDs" |
| 8-day effort estimate | Codex | **Accepted with x3 multiplier per memory** — final estimates 12-15 days |
| --cpu-moe-draft alone forces draft off GPU | Claude on Codex | **Accepted** — final specifies `--device-draft none` + `-ngld 0` if CPU-draft attempted |

## Interview refinements applied

1. **Primary path: 8-GPU 256e + AVG-16e draft** (user chose Gemini framing). **Hard dependency on SPRINT-025 shipping**. AVG-16e + ngram-cache on single V100 becomes the FALLBACK path: ships only if SPRINT-025 hasn't landed by SPRINT-026 start, otherwise the single-V100 path becomes a stretch comparison row in REPORT-20.
2. **Acceptance gate**: tiered (0.50 ship / 0.30-0.49 experimental / <0.30 STOP) per Codex framing.
3. **MoE batch-determinism**: P0 sanity check gate. Capture decode-vs-verify logit diff magnitudes BEFORE running any spec workload; if argmax kicks routinely, sprint reframes around a token-distance gate or forces batch=1 verify.
4. **CUDA-backend-sampling regression**: P5 measurement follow-on. Quantify cost; no commitment to fix in SPRINT-026.

## Final phase structure

- **P-1 — Hard dependency check (0 days)**: SPRINT-025 must have shipped. If not, fall back to single-V100 AVG-16e + ngram-cache as primary, document the pivot in REPORT-20.
- **P0 — Command surface lock + MoE determinism smoke (1 day)**: verify flags (`--spec-type`, `-otd`, `--draft-max`), capture decode-vs-verify logit diffs on a fixed prompt; gate sprint on the magnitude.
- **P1 — Plumbing fixes + functional smoke (1.5 days)**: patch `ngram-cache` hardcoded `n_draft = 8`; verify `common_speculative_is_compat` returns true on TURBOMIND contexts.
- **P2 — Single-V100 AVG-16e + ngram-cache baseline (1.5 days)**: get the cheap path working as fallback and as comparison row.
- **P3 — 8-GPU 256e + AVG-16e draft (3-4 days)**: load both models with VRAM rebalancing on GPU 0; same-output exactness gate; acceptance measurement.
- **P4 — Acceptance + TPS sweep (2 days)**: `--draft-max ∈ {2,4,8,12,16}` × `--draft-min` sweep; identify knee; verify acceptance floor.
- **P5 — Measurement + REPORT-20 + CUDA-sampler regression measurement (2 days)**: full bench tables; CUDA-backend-sampling-forced-off cost quantified.
- **P6 — Close-out (0.5 day)**: tag, follow-ups for SPRINT-027.

Total estimate: 11.5-12.5 days (within the x3 effort buffer for unprecedented territory).

## Outcome contract

Ships if:
- P0 MoE-determinism gate passes (decode==verify argmax in ≥99% of positions on a fixed 100-token suite).
- P3 same-output gate passes (target-only == target+spec generated token IDs across 10 fixed prompts).
- P4 acceptance ≥ 0.50 (full ship), 0.30-0.49 (experimental ship with caveat), else STOP.
- REPORT-20 captures: TPS lift, acceptance sweep, VRAM peak per GPU, CUDA-sampler-disabled cost, reproducible command lines.

Stop-loss:
- If SPRINT-025 hasn't landed AND single-V100 ngram-cache hits < 0.30 acceptance: sprint closes with documented "not worth shipping default-on" verdict, files SPRINT-027 follow-up.

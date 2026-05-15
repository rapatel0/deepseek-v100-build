# SPRINT-026 — Intent

## Seed prompt

Land speculative decoding for DSv4-Flash on the turbomind/V100 path. Pick a draft strategy (small draft model vs n-gram cache vs lookahead), pick a target model that exercises the lift meaningfully on V100 hardware, and ship a measured TPS uplift in a REPORT-20.

Per SPRINT-024-DEFERRED #1 + SPRINT-025-DEFERRED #1. Speculative decoding multiplies effective decode TPS by the acceptance rate × draft length; the M=1 launch-bound regime (SPRINT-023 measured 16.6 t/s flat across model sizes) is the perfect candidate for amortization.

## Orientation summary

1. **llama.cpp has full speculative-decoding plumbing already**:
   - `common/speculative.{h,cpp}` defines `common_speculative`, `common_speculative_init/free`, six speculative-decoding types in `common_speculative_types[]`: `none`, `draft`, three n-gram variants (`ngram_simple`, `ngram_map_k`, `ngram_map_k4v`, `ngram_mod`, `ngram_cache`).
   - `tools/server/server-context.cpp` already loads a draft model when `--model-draft` is supplied, applies `--override-tensor-draft` (`-od`) / `--cpu-moe-draft` overrides, calls `common_speculative_init` per slot, and tracks acceptance stats.
   - `--n-draft` / `--draft-max` controls drafted token count; `--draft-min` controls minimum acceptance threshold; `--draft-p-min 0.75` is the default acceptance probability.
   - Standalone binaries already exist: `llama-speculative`, `llama-speculative-simple`.
2. **No DSv4 draft model is available locally**. `/models/` has the DSv4-Flash variants (MIN-8e, AVG-16e, MIN-16e, MIN-32e, IQ2-64e, 256e) and Qwen variants. **Same-tokenizer constraint** for spec decode: draft and target must share vocabulary. DSv4-Flash variants all share the `deepseek4` arch and vocab (verified at model load in earlier sprints). Qwens have different vocab — out.
3. **Real-weights vs MIN candidates**:
   - DSv4-Flash-AVG-16e (18 GiB, real weights per user) — natural draft candidate for 256e or as a self-spec target.
   - DSv4-Flash-MIN-* (random expert weights) — useless as drafts (would accept ~0%).
   - IQ2-64e (28 GiB) — real-weight, mid-size. Could be a draft for 256e if 16e is too small.
   - 256e is too big to ALSO load alongside a draft on one V100; multi-GPU helps but adds dependency.
4. **N-gram lookup decoding is available and is draft-free**. `--lookup-cache-static` / `--lookup-cache-dynamic` flags hook into the n-gram path. No second model load. Often surprisingly competitive on conversational / code-completion workloads where local repetition is common.
5. **SPRINT-024 (grouped MoE) and SPRINT-025 (multi-GPU 256e) are planning-only** — neither has executed. SPRINT-026 plan should NOT block on either. Cleanest sprint plumbs spec decode against the SPRINT-023 per-expert-dispatch TURBOMIND path on a single-GPU target (AVG-16e fits comfortably on one V100 with TURBOMIND), with a stretch goal of validating on the multi-GPU 256e once 025 ships.
6. **No VISION.md**; ledger script absent.

## Active deferreds becoming actionable

| Source | # | What | Relevance |
|---|---|---|---|
| SPRINT-024 | F-04 | Multi-slot decode (parallel slots OR speculative) | **This sprint = speculative branch** |
| SPRINT-024 | #1 | Decode TPS gate (≥20 t/s landed in SPRINT-024 / SPRINT-026) | This sprint pursues spec-decode lift on top of single-slot |
| SPRINT-025 | #1 | Multi-slot decode / speculative | **This sprint** |

## Relevant codebase areas

- `common/speculative.{h,cpp}` — read-only reference; existing draft / n-gram / lookahead infrastructure.
- `tools/server/server-context.cpp` — slot init applies spec decode if `--model-draft` supplied; already supports tensor-buft-overrides-draft for the draft model.
- `common/arg.cpp` — `-md/--model-draft`, `-od/--override-tensor-draft`, `--cpu-moe-draft`, `--first-cpu-moe-draft`, `--n-draft`, `--draft-min`, `--draft-p-min`, `--lookup-cache-static`, `--lookup-cache-dynamic`, `--speculative-type` (`none`, `draft`, `ngram_*`).
- `tools/server/bench/` — server bench harness.
- `tools/llama-bench/` — single-process bench (does NOT support speculative; spec decode lives in the server hot path).
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}` — the TURBOMIND target path. Draft model loads onto a separate context but shares the same CUDA backend.

## Constraints

- **V100 sm70**: same constraint as 023+024+025. Draft model must also fit + be fast on sm70.
- **Same-tokenizer required** for spec decode — limits draft choice to DSv4-Flash family or another deepseek-arch model with matching vocab.
- **Single-GPU first** (per orientation): don't depend on SPRINT-025 multi-GPU shipping. Stretch to multi-GPU 256e + draft if it lands.
- **No upstream llama.cpp PRs**: changes land on `rapatel0/deepseek-v100-build` branch.
- **Commit-per-phase + push** sticky from SPRINT-023.
- **Don't change the spec-decode core in `common/speculative.{h,cpp}`** unless absolutely required — that's upstream code and modifications there break the "minimal-private-fork" stance. Wire from outside instead.

## Success criteria

1. **Functional**: `llama-server --model <target> --model-draft <draft> -ot 'exps=CUDA_TURBOMIND0'` (or family alias) loads both models on a single V100, initializes spec decode per slot, and decodes via the draft → target verify path without crash.
2. **Acceptance ≥ baseline**: For a fixed prompt set (10 prompts mixed chat / code), median acceptance rate ≥ 0.50 (i.e. half the drafted tokens get accepted). Below 0.30 → spec overhead exceeds the win; reconsider.
3. **Measured uplift**: Decode TPS on TARGET-with-spec ≥ 1.3× TARGET-without-spec at temp=0 on the same prompt set. Stretch: 2×.
4. **Same-output verification**: With `temp=0`, target-only and target+spec produce IDENTICAL token sequences (spec decode is exact under temp=0; any divergence is a bug).
5. **N-gram alternative measured**: Same TPS uplift quantified for `--speculative-type ngram_cache` with a dynamic cache; report both numbers in REPORT-20.
6. **Stretch (only if SPRINT-025 ships during 026 execution)**: spec decode on 256e/8-GPU target with a 16e single-GPU draft.

## Verification strategy

- **Functional smoke**: `llama-server` boots with both models; `/health` shows ready; one /completion request returns valid tokens.
- **Same-output gate**: For 10 fixed prompts at `temp=0` and `seed=0`, capture token IDs from {target-only, target+spec-draft, target+ngram_cache}. The target+spec results must match target-only EXACTLY.
- **Acceptance + TPS**: `llama-bench` doesn't support spec decode. Use the server: drive 32-token greedy completions on 10 prompts, capture per-prompt acceptance from server logs (`server-context.cpp:338`), aggregate decode-TPS from `predicted_per_second`.
- **Memory accounting**: Per-GPU VRAM with both models loaded; verify no OOM, document peak.

## Uncertainty assessment

| Factor | Level | Why |
|---|---|---|
| Correctness | **Low-Medium** | Spec decode is well-tested upstream; the new surface is the draft × TURBOMIND interaction. |
| Scope | **Medium** | Sprint can balloon if we get into evaluating multiple draft models or pursuing custom draft training. Need to pick ONE draft strategy as primary. |
| Architecture | **Low** | Reuses existing `common_speculative` machinery; no new dispatch surface. |

## Open questions (for the interview)

1. **Draft strategy** — small dedicated draft model (likely AVG-16e or a smaller hand-built draft), n-gram cache, or both? N-gram costs nothing extra at runtime; draft model is more accurate but doubles VRAM.
2. **Target model** — DSv4-Flash-AVG-16e (single-GPU, fast iteration) or wait for SPRINT-025 multi-GPU 256e (more impactful)?
3. **Draft model source** — use AVG-16e as the draft (real weights, 18 GiB)? Or smaller? AVG-16e + AVG-16e self-spec is degenerate (draft == target).
4. **Stretch into multi-GPU 256e**: in-scope if 025 ships during 026, or out-of-scope regardless?
5. **Acceptance gate** — is ≥50% median acceptance the right floor? Some literature reports 30-40% as still net-positive when draft is much smaller than target.

## What this sprint is NOT

- New spec-decode algorithm research.
- Training a custom DSv4 draft model from scratch.
- Multi-slot parallel sequences (the other branch of "multi-slot decode"). Selected branch is speculative.
- Production server hardening beyond the existing tools/server surface.
- Re-running SPRINT-023/024/025 measurements; treats those baselines as fixed inputs.

## Vision context

No `docs/sprints/VISION.md`. SPRINT-022 → 023 → 024 → 025 chain pursued:
- 022: DSv4-Flash operational on V100.
- 023: V100 path 3-3.6× over CPU MoE baseline.
- 024: launch amortization via grouped MoE (planned).
- 025: full 256e on 8× V100 (planned).
- **026: spec-decode lift on top of single-slot decode TPS.**

Orthogonal axes: 024 amortizes per-launch cost, 025 scales model size, 026 multiplies effective TPS per token by the draft × verify ratio.

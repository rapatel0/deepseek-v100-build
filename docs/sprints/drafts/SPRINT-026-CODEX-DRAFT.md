# SPRINT-026 — Speculative decoding landing on the V100 turbomind path

**Status:** DRAFT 2026-05-15  
**Predecessor:** SPRINT-023 (measured 16.2-16.6 t/s single-slot decode), with SPRINT-024 and SPRINT-025 still planning-only  
**Successor:** SPRINT-027 (only if SPRINT-026 proves spec decode is workload-positive and worth hardening further)

## Overview

SPRINT-023 established the key decode fact for this branch: DSv4-Flash on the V100 turbomind path is effectively flat in model size at single-slot decode, which means the hot path is launch-bound rather than math-bound. SPRINT-026 attacks that exact bottleneck with speculative decoding.

This sprint makes one explicit strategic choice:

1. **Primary ship path:** self-speculative decoding on **`DSv4-Flash-AVG-16e`** using **`ngram-cache`** on a single V100.
2. **Stretch-only path:** exercise the existing two-model `draft` path with **`DSv4-Flash-AVG-16e` as the only acceptable draft model**.
3. **Rejected for this sprint:** implementing a new lookahead algorithm. That would expand scope into speculative-core research and violate the minimal-private-fork stance.

Why this is the right cut:

- `AVG-16e` is the smallest locally documented **real-weight** DSv4 model. `MIN-*` models are random-weight fixtures and are not valid draft models.
- Qwen-family models are excluded by the **same-tokenizer / same-vocab compatibility** requirement.
- A meaningful two-model V100 path is probably memory-blocked; the sprint should still wire it cleanly, prove that with numbers, and avoid pretending it is the main deliverable.

**Headline success condition:** on a fixed `temp=0` prompt suite, speculative mode must preserve exact output and deliver a measured decode-TPS lift that is large enough to matter. The acceptance target is **median >= 0.50**. **0.30 is the hard floor**: below that, the overhead is no longer worth shipping as the recommended path.

---

## Use Cases

| Use case | Why it matters | Ship target |
|---|---|---|
| Single-GPU interactive DSv4-Flash serving on a V100 | This is the shortest path from SPRINT-023's 16.6 t/s plateau to a real throughput lift without new kernels | **Primary** |
| Repetitive code-edit / rewrite loops | `ngram-cache` is most plausible when token history contains reusable local structure | **Primary** |
| Exactness-sensitive greedy decode (`temp=0`) | Spec decode is only acceptable if target-only and target+spec produce the same answer | **Primary** |
| Two-model `--model-draft` server wiring for future larger targets | The upstream path already exists; we should either validate it on this branch or document the exact VRAM blocker | **Stretch** |
| 256e multi-GPU + draft experiment | High upside, but only if memory math closes | **Stretch, non-blocking** |

---

## Architecture

### 1. Chosen model strategy

**Primary target model:** `DSv4-Flash-AVG-16e`

- Real weights, unlike `MIN-*`
- Fits the existing single-V100 turbomind workflow from SPRINT-023
- Meaningful enough to benchmark, but small enough to iterate quickly

**Chosen draft model:** `DSv4-Flash-AVG-16e`

- This is the **only acceptable draft model choice** in the current documented local inventory
- It shares the DSv4 tokenizer/vocab family with the target path
- It is rejected as the primary ship path because target+draft VRAM does not fit comfortably on a single 32 GiB V100

**Explicit rejections**

- `DSv4-Flash-MIN-*` as draft: random expert weights, so acceptance should be near-zero
- Qwen as draft: tokenizer/vocab incompatibility
- `IQ2-64e` as draft: larger than the single-GPU target we can iterate on
- New lookahead implementation: not already present in `common/speculative.*`

### 2. Tokenizer and vocab gate

The repo already enforces the real compatibility rule in `common/speculative.cpp`:

- vocab type must match
- BOS/EOS behavior must match
- token text must match across the checked vocab range

For this sprint, treat that as a hard product rule:

- **Only DeepSeek-family draft/target pairs are allowed**
- Architecture name match is not enough by itself; the actual speculative compatibility check must pass at runtime
- No benchmark result counts unless the server is running with the real compatibility path, not a hand-waved assumption

### 3. CLI and server wiring

The repo already has most of the plumbing:

- `--model-draft` / `-md` loads the draft model in `tools/server/server-context.cpp`
- `--override-tensor-draft` / `-otd` applies draft-only tensor placement overrides
- `--cpu-moe-draft` and `--n-cpu-moe-draft` exist
- `--device-draft`, `--gpu-layers-draft`, and `--ctx-size-draft` already exist for explicit draft placement
- `--spec-type` selects draftless speculation

This sprint should make the surface less error-prone for this fork:

1. Keep upstream flags untouched.
2. Add **`--speculative-type`** as an alias for `--spec-type`.
3. Add **`-od`** as a short alias for `--override-tensor-draft` while retaining upstream `-otd`.
4. Document the accepted public values as the hyphenated CLI forms:
   - `none`
   - `ngram-cache`
   - `ngram-simple`
   - `ngram-map-k`
   - `ngram-map-k4v`
   - `ngram-mod`
5. Keep the internal underscore enum names private implementation detail.

One important behavioral detail must be called out in docs and bench scripts:

- If a draftless implementation and a draft model are both configured in the same run, the draftless implementation wins first.
- Therefore `ngram-cache` and `--model-draft` must be benchmarked in **separate server runs**.

### 4. VRAM plan with two models

The sprint should make the memory story explicit instead of discovering it late:

| Configuration | Approx model weight footprint | Practical verdict |
|---|---|---|
| `AVG-16e` target alone | ~18 GiB | Good single-V100 ship path |
| `AVG-16e` target + `AVG-16e` draft | ~36 GiB before KV, scratch, allocator overhead | Not a real single-V100 perf path |
| `IQ2-64e` target + `AVG-16e` draft | ~46 GiB before KV/scratch | Not viable on one V100 |
| `256e` on 8x V100 + `AVG-16e` draft | Very likely memory-blocked because SPRINT-025 already leaves only a few GiB/GPU headroom | Stretch-only, prove or defer |

That leads to a clear sprint rule:

- **REPORT-20 headline numbers come from the single-model self-spec path**
- Two-model draft work is for wiring validation and a future larger-model path, not for the main success criterion

### 5. Exactness and measurement plan

The server already returns the right metrics in completion timings:

- `predicted_per_second`
- `draft_n`
- `draft_n_accepted`

So the sprint does **not** need to scrape logs for its main benchmark loop.

The correctness gate is:

- request parameters: `temperature = 0.0`, `top_k = 1`, `seed = 0`
- compare target-only vs target+spec on the same prompt suite
- exact output match is required

The acceptance gate is:

- **median acceptance >= 0.50** for the recommended ship configuration
- **0.30-0.49** means "interesting but not recommended by default"
- **< 0.30** is a fail for the claimed workload

### 6. Minimal-core-change policy

The plan should avoid changing `common/speculative.*` unless the current behavior makes benchmarking or tuning misleading. One such fix is justified:

- `ngram-cache` currently hardcodes `n_draft = 8` in `create_state_ngram_cache(...)`
- That must be changed to honor the configured speculative draft length, otherwise `--draft-max` is fake for the chosen ship path

No other speculative-core changes are part of the base plan.

---

## Implementation

### P0 — Lock scope, feasibility, and command surface (1 day)

**Goal:** remove ambiguity before writing code or running long benches.

1. Confirm the actual public flag surface in `common/arg.cpp` and `tools/server/README.md`.
2. Verify the precedence rule: do not mix `--model-draft` and `ngram-*` in the same benchmark command.
3. Reproduce a single-GPU `AVG-16e` server baseline on the V100 turbomind path.
4. Write down the two-model memory math for:
   - `AVG-16e` target + `AVG-16e` draft
   - `IQ2-64e` target + `AVG-16e` draft
   - `256e` target + `AVG-16e` draft
5. Reject lookahead as out of scope unless an existing implementation is found in-tree.

**P0 Gate**

- ✅ Primary ship path is fixed: `AVG-16e` + self-spec on one V100
- ✅ Only accepted draft model is fixed: `AVG-16e`
- ✅ Two-model path is explicitly marked feasible or stretch-only

### P1 — CLI aliases, docs, and draft-specific override ergonomics (1 day)

**Goal:** make the DSv4 speculative surface obvious and scriptable.

1. Add `--speculative-type` as an alias for `--spec-type`.
2. Add `-od` as a short alias for `--override-tensor-draft`, preserving `-otd`.
3. Clarify in help text that `--model-draft` is a second model load, while `--speculative-type` selects draftless speculation.
4. Update `tools/server/README.md` and `docs/speculative.md` with:
   - alias names
   - precedence rule
   - example DSv4 commands
5. Add one unit test or parser-level smoke to confirm the aliases parse to the expected fields.

**P1 Gate**

- ✅ `--speculative-type` and `-od` parse
- ✅ Upstream names still parse
- ✅ Docs no longer mix internal underscore names with public hyphenated CLI values

### P2 — Exactness harness and benchmark plumbing (1.5 days)

**Goal:** make speculative results measurable and falsifiable.

1. Add a small server-bench harness for speculative runs, for example:
   - `tools/server/bench/speculative_bench.py`
   - `tools/server/bench/prompts/speculative-026.json`
2. Prompt suite: 10 prompts, biased toward the workloads where self-spec can win:
   - 4 code-edit / rewrite prompts
   - 3 repetitive instruction / summarization prompts
   - 3 general chat prompts
3. Run three modes from the same harness:
   - target-only baseline
   - `ngram-cache`
   - optional draft-model smoke mode
4. Compare:
   - exact output text
   - `predicted_per_second`
   - `draft_n`
   - `draft_n_accepted`
5. Extend `tools/server/tests/unit/test_speculative.py` with a DSv4-style exactness assertion pattern, still using `temp=0`.

**P2 Gate**

- ✅ Harness produces one table row per prompt per mode
- ✅ Exact-output comparison works
- ✅ Timings JSON is the source of truth for TPS and acceptance

### P3 — Primary ship path: `ngram-cache` on `AVG-16e` (2 days)

**Goal:** land the actual sprint deliverable.

1. Patch `common/speculative.cpp` so `ngram-cache` honors configured draft length instead of hardcoding `8`.
2. Keep the initial tuning space deliberately small:
   - `--draft-max` in `{8, 12, 16}`
   - `--draft-min` in `{4, 8}`
3. Use the existing single-V100 turbomind DSv4 server command, then benchmark speculative vs baseline via the harness.
4. Require the `temp=0` same-output gate before considering performance numbers valid.
5. Pick one recommended command line for REPORT-20.

**P3 Gate**

- ✅ Same-output gate passes on the full prompt suite
- ✅ Median acceptance >= 0.50 for the recommended config, or the sprint drops to experimental-only
- ✅ Decode TPS uplift >= 1.3x over target-only on the chosen workload mix

**P3 Stop Rules**

- If median acceptance is `< 0.30`, stop calling `ngram-cache` the ship path
- If output diverges at `temp=0`, stop and fix correctness before any more tuning

### P4 — Two-model draft path: wiring validation, not hero benchmarking (1.5 days)

**Goal:** validate the explicit draft-model surface without pretending the memory problem does not exist.

1. Use `DSv4-Flash-AVG-16e` as the draft model in every draft-path experiment.
2. Exercise the real server path:
   - `--model-draft`
   - `-od` / `--override-tensor-draft`
   - `--device-draft` and `--gpu-layers-draft`
   - `--cpu-moe-draft` or `--n-cpu-moe-draft` if a smoke run needs relief
   - default `--draft-p-min 0.75` unless there is a proven reason to retune it
3. Only claim performance results if both models are in a layout that is actually relevant for deployment.
4. If the only runnable draft configuration requires obviously non-shipworthy CPU spill or destroys latency, record it as a blocker, not a success.
5. If SPRINT-025 has landed far enough to try `256e`, attempt exactly one layout:
   - target on the multi-GPU path
   - draft = `AVG-16e`
   - abort on memory pressure instead of spending the sprint inventing a new placement scheme

**P4 Gate**

- ✅ Draft path either runs end-to-end with exact output, or the VRAM blocker is documented precisely
- ✅ No fake benchmark numbers from a knowingly non-viable deployment layout

### P5 — REPORT-20 and ship recommendation (1 day)

**Goal:** turn the sprint into a decision, not just a pile of measurements.

1. Write `docs/sprints/SPRINT-026-REPORT-20.md`.
2. Required sections:
   - baseline command and TPS
   - recommended speculative command and TPS
   - acceptance table
   - exactness result
   - memory table
   - two-model draft-path verdict
   - recommended default / experimental / rejected configurations
3. End with one explicit verdict:
   - **SHIP** if acceptance >= 0.50 and TPS uplift >= 1.3x
   - **EXPERIMENTAL** if acceptance is 0.30-0.49 but results are workload-specific
   - **STOP** if acceptance < 0.30 or exactness fails

**P5 Gate**

- ✅ REPORT-20 is reproducible
- ✅ One recommended command line exists, or the sprint explicitly says not to enable speculative mode by default

### P6 — Fallback path if `ngram-cache` misses the floor (1 day, conditional)

**Goal:** salvage the sprint without changing the architecture.

1. Reuse the same harness and prompt suite.
2. Evaluate `ngram-mod` as the only fallback candidate.
3. Do **not** reopen draft-model research or invent lookahead here.
4. If `ngram-mod` clears the same-output and acceptance gates where `ngram-cache` did not, REPORT-20 can ship `ngram-mod` as the recommended path and record `ngram-cache` as informative-only.

**P6 Gate**

- ✅ Either a self-spec path ships, or the sprint closes with a clean "not worth enabling on this workload" conclusion

---

## Files Summary

### Modified

| Path | Change |
|---|---|
| `common/arg.cpp` | Add `--speculative-type` and `-od` aliases; clarify help text |
| `common/speculative.cpp` | Make `ngram-cache` honor configured draft length |
| `tools/server/tests/unit/test_speculative.py` | Add alias and exact-output coverage |
| `tools/server/README.md` | Document aliases, precedence, DSv4 examples |
| `docs/speculative.md` | Same documentation updates, with public CLI names only |

### Added

| Path | Change |
|---|---|
| `tools/server/bench/speculative_bench.py` | Reproducible speculative benchmark harness |
| `tools/server/bench/prompts/speculative-026.json` | Fixed prompt suite for REPORT-20 |
| `docs/sprints/SPRINT-026-REPORT-20.md` | Measurement and ship decision |

---

## Definition of Done

1. `AVG-16e` target-only and target+spec runs are reproducible on the single-V100 turbomind path.
2. `--speculative-type` and `-od` work, while `--spec-type` and `-otd` remain supported.
3. The chosen self-spec path passes the `temp=0` same-output gate on the fixed prompt suite.
4. The chosen self-spec path reaches **median acceptance >= 0.50**, or REPORT-20 explicitly downgrades it from recommended to experimental.
5. Any result below **0.30 acceptance** is treated as a fail for the claimed workload.
6. `ngram-cache` obeys configured draft length rather than silently hardcoding `8`.
7. Two-model draft-path feasibility is documented with actual memory numbers and an explicit yes/no verdict.
8. REPORT-20 contains baseline TPS, speculative TPS, acceptance data, exactness result, memory table, and a reproducible command block.

---

## Risks

1. **Workload dependence.** Self-spec can look great on code rewrites and mediocre on general chat.
2. **Two-model VRAM failure.** The explicit draft path may be structurally blocked on current hardware.
3. **False tuning confidence.** If `ngram-cache` ignores `--draft-max`, benchmark conclusions are misleading until fixed.
4. **Benchmark contamination.** Mixing `--model-draft` with draftless speculation in one command would produce the wrong conclusion because of precedence rules.
5. **Overfitting to greedy decode.** `temp=0` is the right correctness gate, but REPORT-20 should still state that non-greedy behavior may differ.

---

## Security

1. Speculative decoding must not change the exact output in the `temp=0` gate; correctness is the first safety control.
2. Do not enable any cross-request shared speculative cache by default in this sprint.
3. If a persistent lookup-cache file is used experimentally, keep it local to the bench workspace and document that it is derived from prompt traffic.
4. No new network surface is introduced; all changes stay inside argument parsing, server wiring, and benchmark tooling.

---

## Dependencies

1. SPRINT-023 baseline numbers and command surface.
2. Existing upstream speculative infrastructure in `common/speculative.*` and `tools/server/server-context.cpp`.
3. A working single-V100 DSv4 turbomind server path for `AVG-16e`.
4. SPRINT-025 only if the stretch `256e` draft experiment is attempted; it is not a hard dependency for the sprint to ship.

---

## Open Questions

1. Should the fork add `--speculative-type` and `-od`, or keep upstream names only and fix the sprint docs instead?
2. Is in-memory `ngram-cache` sufficient for REPORT-20, or do we want persistence semantics for `--lookup-cache-dynamic` in the same sprint?
3. If `ngram-cache` lands in the 0.30-0.49 band, do we ship it as an explicitly workload-specific flag, or require P6 to clear the bar with `ngram-mod` first?
4. If the only viable `--model-draft` run uses CPU-offloaded draft MoE weights, is that still worth keeping as a smoke test, or should the draft path be documented and deferred without running it?

# SPRINT-026 — Critique of CODEX and GEMINI drafts

**Date:** 2026-05-15
**Reviewer:** Claude
**Scope:** SPRINT-026-CODEX-DRAFT.md and SPRINT-026-GEMINI-DRAFT.md, measured against SPRINT-026-INTENT.md and verified against the codebase.

---

## TL;DR

- **Codex** is the safer, more accurate plan. It correctly identifies the real upstream flag names, the real-weight draft constraint, the VRAM blocker, and has a workable fallback. Weak spots: the "self-speculative" framing is mislabeled, P3 quietly bets the report on `ngram-cache` clearing 50% acceptance, and the effort estimate is optimistic.
- **Gemini** has multiple concrete technical errors that would cause the sprint to fail at runtime: wrong CLI flag names, wrong CLI values (underscores instead of hyphens), wrong draft model choice (MIN-8e is random weights), and a primary ship path that depends on an unshipped predecessor (SPRINT-025). The VRAM math is shallow and the 256e + draft layout will not close on 8x V100 32 GiB without numbers it does not show.
- The intent doc itself has at least one error that both drafts inherited or partially corrected: it lists `--speculative-type` and `--n-draft` and a mix of underscore/hyphen values that do not match what is in `common/arg.cpp`.

If only one ships, ship the Codex plan with the corrections in §8.

---

## 1. Ground-truth checks against the codebase

Before critiquing, here is what the codebase actually says — both drafts must square with this.

### 1.1 Real upstream flag surface (`common/arg.cpp`)

| Concept | Real flag(s) | Codex claim | Gemini claim |
|---|---|---|---|
| Draft model | `-md`, `--model-draft` | `--model-draft` ✅ | `--model-draft` / `-md` ✅ |
| Draftless spec selector | `--spec-type` | `--spec-type` (proposes adding `--speculative-type` alias) ✅ | `--speculative-type` ❌ (does not exist upstream) |
| Spec-type values | `none\|ngram-cache\|ngram-simple\|ngram-map-k\|ngram-map-k4v\|ngram-mod` (hyphenated) | hyphenated ✅ | `ngram_cache` ❌ (underscore, will not parse) |
| Override tensor for draft | `-otd`, `--override-tensor-draft` | `-otd` (proposes adding `-od` alias) ✅ | `-od` ❌ (does not exist upstream) |
| Draft length | `--draft`, `--draft-n`, `--draft-max` (aliases) | `--draft-max` ✅ | `--n-draft` ❌ (does not exist; closest is `--draft-n`) |
| Draft min | `--draft-min`, `--draft-n-min` | `--draft-min` ✅ | not used |

Gemini's example command in P3.2:
```
llama-server -m /models/256e.gguf -md /models/AVG-16e.gguf \
  -sm layer -ngl 999 -ot 'exps=CUDA_TURBOMIND' \
  -od 'CUDA_TURBOMIND0' --n-draft 4
```
Two of those flags (`-od`, `--n-draft`) do not exist. This command will fail at argument parsing. Also `-ot 'exps=CUDA_TURBOMIND'` (no GPU index) is suspect — SPRINT-023/024 used `CUDA_TURBOMIND0` family.

The intent doc itself perpetuates some of this: it says `--n-draft / --draft-max` and lists `ngram_simple`, `ngram_cache`, `ngram_mod` etc. (underscored). Anyone copying from intent will inherit the bug. Codex correctly resolves to the real upstream names; Gemini does not.

### 1.2 `ngram-cache` actually hardcodes `n_draft = 8`

Codex's P3 step 1 says: *"Patch `common/speculative.cpp` so `ngram-cache` honors configured draft length instead of hardcoding 8."*

**Verified.** `common/speculative.cpp:757`:
```cpp
static common_speculative_state_ngram_cache create_state_ngram_cache(
    ...
    uint16_t n_draft = 8; // TODO get from config?
```

This is a real bug that turns `--draft-max` into a no-op for the ngram-cache path. Codex caught it; Gemini did not. This is the only `common/speculative.cpp` change Codex proposes and it is justified.

### 1.3 `server-context.cpp` does what intent claims

Verified in `tools/server/server-context.cpp`:
- L661-693: loads draft model when `params_base.speculative.has_dft()`, applies `cparams_dft`.
- L770-796: `common_speculative_is_compat(ctx)` gate, `common_speculative_init` per slot.
- L338-341: emits `draft_n` and `draft_n_accepted` in completion timings — Codex correctly uses these as the source of truth; Gemini says "from server logs", which is the longer/scrape-y path.
- L1197-1198: `backend_sampling &= !(slot.spec && task.params.speculative.n_max > 0)`. **Neither draft notices** that enabling speculative *forces sampling off the CUDA backend* and back to CPU sampling. For DSv4 + TURBOMIND on V100, this is non-trivial overhead in the M=1 regime and could eat part of the speculative win. This deserves a line in the risks table.
- L729-731: speculative auto-disabled with multimodal — not relevant for DSv4, but neither draft documents it.

### 1.4 Real-weight constraint

From memory: `dsv4_flash_min_models_are_garbage.md` — MIN-* models have random expert weights. Acceptance rate from a random-weight draft will be near-zero by construction. Codex respects this; Gemini does not (P2.1 uses `MIN-8e` as a draft).

### 1.5 VRAM ground truth on 256e

SPRINT-025 is planning-only. The 256e GGUF size is ~156 GiB and 8x V100 = 256 GiB total, so distributed = ~19.5 GiB/GPU before KV cache, scratch, allocator overhead. KV cache for DSv4-Flash 256e at the contexts SPRINT-025 plans is several GiB per GPU. **Neither draft documents the post-KV per-GPU headroom on 8x V100**, and that headroom is the whole question for the two-model path.

---

## 2. Codex draft — assessment

### 2.1 Strengths

- **Correct flag surface.** Recognizes that `--spec-type` is the upstream flag and that `--speculative-type` would need to be added as an alias. Treats hyphenated values as the public surface and underscores as private implementation detail.
- **Correct real-weight constraint.** Explicitly rejects MIN-* as draft, rejects Qwen on tokenizer, rejects IQ2-64e as too large.
- **Correct VRAM verdict.** AVG-16e + AVG-16e ≈ 36 GiB before KV ≫ 32 GiB on one V100, so the two-model single-V100 path is correctly marked not-a-ship-path. IQ2-64e + AVG-16e ≈ 46 GiB is correctly marked not-viable.
- **Caught the real `ngram-cache` bug** (hardcoded `n_draft=8`) — without this fix `--draft-max` sweeps in P3 would be fake. Good catch.
- **Decoupled from SPRINT-025.** Primary path is `AVG-16e` + `ngram-cache` on one V100; the 256e + draft experiment is stretch and abortable.
- **Exactness gate is correct.** Asks for `temperature=0`, `top_k=1`, `seed=0` — Gemini only specifies `temp=0`, which is necessary but insufficient (see §4.4).
- **Right benchmark surface.** Uses completion-timings JSON (`predicted_per_second`, `draft_n`, `draft_n_accepted`) instead of scraping logs.
- **Has a real fallback (P6).** If `ngram-cache` misses the floor, try `ngram-mod` with the same harness and prompt suite. No reopening of draft-model research.
- **Honest "STOP" verdict.** Allows the sprint to close with "do not enable by default" if acceptance < 0.30.

### 2.2 Weaknesses

- **"Self-speculative" is mislabeled.** The overview calls the primary path "self-speculative decoding on `DSv4-Flash-AVG-16e` using `ngram-cache`." Self-speculation usually means a model drafts for itself via its own shallower computation (e.g., medusa heads). `ngram-cache` is **draft-free**, not self-spec. Calling it self-spec invites a reader to expect a second model context, conflating it with the stretch path that uses a real draft model. Rename to "draftless speculation via ngram-cache" or just "ngram-cache" everywhere.
- **Effort estimate is optimistic.** ~8 days (P0…P6). My memory rule (`feedback_effort_estimation_undocumented_hardware.md`) says multiply gut estimates by 3 when signals like "draft model + TURBOMIND interaction not previously exercised" are present. Realistic: 12–14 days. The P2 1.5d harness + P3 2d ngram-cache landing alone usually slides.
- **Whether `ngram-cache` can hit 0.50 acceptance is unaddressed.** Lookup decoding rates are workload-dependent. Published numbers on lookup decoding land 0.20–0.45 for general chat and 0.40–0.60 for code rewrite. Codex commits the headline number to ngram-cache hitting 0.50 without analysis; if it lands at 0.35 the sprint downgrades to experimental, but P3 also says "Pick one recommended command line for REPORT-20" — that line might not exist. Should explicitly call out that the prompt mix is biased toward code-edit and that this is required to hit the floor; if the prompt mix is general-chat-weighted, drop the floor or change strategy.
- **`--draft-min` sweep values are guessed.** Plan tunes `--draft-min ∈ {4, 8}`; with `--draft-max ∈ {8, 12, 16}`, `--draft-min=8` is degenerate against `--draft-max=8`. The combinatorics need pruning before this becomes a P3 task.
- **P4 (two-model path) is half-defined.** "Use AVG-16e as the draft in every draft-path experiment" but AVG-16e + AVG-16e self-draft on a single V100 is rejected in §3.4. There is no concrete single-GPU two-model layout. So P4 becomes either (a) a multi-GPU experiment that requires SPRINT-025, (b) a CPU-spill experiment that is explicitly ruled out, or (c) skipped. Codex hedges with "if the only runnable draft configuration requires obviously non-shipworthy CPU spill ... record it as a blocker" — fine, but then P4 is effectively a doc deliverable, not a runnable test. Mark it that way.
- **Does not flag the `backend_sampling` side-effect** (server-context.cpp:1197-1198) that disables CUDA backend sampling whenever speculative is on. For an M=1 launch-bound workload, CPU sampling adds visible per-token overhead that will reduce the measured TPS lift.
- **Does not call out KV-cache rewind on rejection** as a TURBOMIND-specific risk. Per-expert dispatch state might not unwind cleanly under spec decode's reject-and-resample path. This is a concrete sm70 risk worth listing.
- **Open Question #2** asks whether `--lookup-cache-dynamic` persistence should land in this sprint. The default plan does not persist anything. Persistence has security implications (Codex notes it in §Security #3, good) but the question should be answered in the plan, not punted.

### 2.3 Gaps in risk analysis

- No risk for **CUDA backend sampling forced off** under spec mode (server-context.cpp:1197).
- No risk for **TURBOMIND KV-cache rewind correctness** under rejection.
- No risk for **MoE routing nondeterminism between batch=1 (decode) and batch>1 (verify)** — if expert dispatch picks differently because of batch-dependent top-k tiebreaks, the verify pass produces different logits than a re-decode would, breaking the same-output gate even at `temp=0, top_k=1`. This is a real concern for DSv4-Flash MoE and worth a P0 sanity check.
- No risk for **prompt-suite gaming**. With 10 prompts hand-picked toward code edits, hitting 0.50 may say more about the suite than the workload. Note that the report's verdict is suite-conditional.

### 2.4 DoD completeness

The Codex DoD covers: reproducible runs, alias flags work, same-output gate, acceptance floor (≥0.50 or downgrade), the `ngram-cache` draft-length fix, the two-model documented verdict, and the report sections. Missing:

- "Same-output gate passed across N≥3 different `--draft-max` values" — otherwise an acceptance number from one config is paired with an exactness check from a different config.
- "Server boots and serves with `--spec-type ngram-cache` AND with `--model-draft <m>` in two separate runs" — the precedence rule (§3.3 of the plan) is documented but no DoD item enforces it was actually tested both ways.
- "VRAM peak measured with `nvidia-smi --query-gpu=memory.used` and recorded per GPU, not estimated."

---

## 3. Gemini draft — assessment

### 3.1 Strengths

- **Names the two strategies clearly** (n-gram lookup vs small draft model) and frames the M=1 launch-bound bottleneck correctly.
- **Notes the exact same-tokenizer constraint** and gives valid/invalid pairs.
- **Risk table format** is clean: likelihood × impact × mitigation.
- **VRAM section has the right shape** (target shard + draft + KV/scratch > 32 GiB) even if the numbers are shallow.
- **Acknowledges SPRINT-025 dependency as a risk** (Risk #4) — even though the consequence is fatal to its primary plan.

### 3.2 Weaknesses — technical errors that break the plan

1. **CLI flag errors.** As enumerated in §1.1:
   - `--speculative-type` does not exist upstream; the flag is `--spec-type`.
   - `--n-draft` does not exist; the canonical flag is `--draft-max` (alias `--draft-n`, `--draft`).
   - `-od` does not exist; the flag is `-otd` / `--override-tensor-draft`.
   - `ngram_cache` (underscore) is the internal enum name; the CLI value is `ngram-cache` (hyphen).
   Every example command in Gemini's plan would fail at argument parsing. Without these fixes, P0.2, P1.2, and P3.2 are all non-runnable as written.

2. **MIN-8e used as draft model.** P2.1: *"Load a (small, non-real) `MIN-8e` draft and `AVG-16e` target."* — MIN-8e has random expert weights (verified in memory record [[dsv4_flash_min_models_are_garbage]]) so its draft predictions cannot match the target. P2.2 effectively says "we expect this to fail; we'll document it." That makes P2 a 2-day phase whose conclusion is known before it starts. Drop it or replace with AVG-16e self-draft load smoke + a single real measurement.

3. **Primary ship path depends on unshipped SPRINT-025.** The DoD requires "Measured decode TPS for 256e + 16e-draft ≥ 1.3× baseline" — i.e., the headline gate is a multi-GPU 256e configuration that the predecessor sprint has not produced. Risk #4 acknowledges this but only proposes "delay P3-P4." If 025 does not ship, more than half the implementation phases collapse. The intent doc explicitly says single-GPU first; Gemini inverts that.

4. **VRAM math on 256e is hand-wavey.**
   - "~156 GiB / 8 GPUs = ~19.5 GiB/GPU" — this is *weights only*. KV cache (per-layer, per-batch-position) on DSv4-Flash at typical context lengths adds 2–4 GiB/GPU. Scratch buffers, allocator overhead, NCCL buffers, and CUBLAS workspace add another 1–2 GiB. Real per-GPU after-weights headroom on a 32 GiB V100 is closer to 6–8 GiB, not the implied 12.5 GiB.
   - "Carve out 18 GiB on GPU 0 for the AVG-16e draft" — requires shifting ~18 GiB of weights off GPU 0 onto GPUs 1–7, i.e. +2.6 GiB/GPU on the other seven. With ~6–8 GiB headroom each, that fits *only if* SPRINT-025 leaves ≥3 GiB headroom on each non-zero GPU, which is not yet a measured fact.
   - The plan does not show how `-sm layer` weights actually achieve a non-uniform split (it controls split mode; layer weights need `-ts` or per-layer assignment which `-sm layer` alone does not give).

5. **Self-speculation contradiction.** P1.2: *"Self-speculation (draft=target) is only for loading verification; it will fit on 2x V100 but not 1x."* — the test as written uses one GPU (the example shows `llama-server -m AVG-16e -md AVG-16e -od 'CUDA_TURBOMIND'`). If it does not fit on 1x, the test cannot execute. P1.2 has no 2x V100 invocation.

6. **`-ot 'exps=CUDA_TURBOMIND'` missing GPU index.** SPRINT-023/024 patterns use `CUDA_TURBOMIND0` (or the family alias). Bare `CUDA_TURBOMIND` may match nothing depending on how the backend name registry resolves regex.

7. **"Mathematically exact" overclaims.** §3.4: *"Under temp=0 (greedy), speculative decoding is mathematically exact."* True only if (a) `top_k=1` (or equivalent deterministic tie-break), (b) the verify pass produces bit-identical logits to a serial decode for the same prefix, and (c) sampling is deterministic. cuBLAS GEMM order and reduction tree can differ between batch=1 decode and batch=K+1 verify, producing tiny FP16 logit differences that change argmax. **Bit-identical at `temp=0` is not free**, especially on MoE where top-k expert selection at batch>1 can route differently than batch=1. Codex's `top_k=1, seed=0, temp=0` triple is closer to right; Gemini's `temp=0` alone is not enough.

### 3.3 Gaps in risk analysis

- Same as Codex: no risk for CUDA backend sampling disabled under spec mode.
- No risk for **MoE expert routing differing between decode and verify batch shapes** — for a same-output gate this is the most likely failure mode and Gemini does not list it.
- No risk for **KV-cache rewind under TURBOMIND**.
- Risk #2 ("Low acceptance rate") mitigation is "Revert to N-gram" — but if both n-gram and draft-model are the two strategies, "revert to n-gram" is not a fallback, it is the other primary. Circular.
- No risk for **`-sm layer` actually delivering the required non-uniform split**.
- No risk for the fact that **the entire primary path is gated on a sprint that has not started**.

### 3.4 Missing edge cases

- KV-cache rewind on rejection (also missed by Codex).
- MoE routing batch-shape determinism (also missed by Codex; more salient for Gemini since its primary is 256e MoE).
- Acceptance-rate behavior for `ngram_cache` on first ~64 tokens of a conversation (cold cache) — Gemini's P0.4 measures lift but does not separate cold-start vs warm.
- "What happens if `-md` is supplied but the vocabs do not actually match" — server should refuse; needs explicit smoke.
- Whether `common_speculative_is_compat(ctx)` (server-context.cpp:770) returns true for TURBOMIND contexts — neither draft verifies this, and a `false` return silently disables spec without crashing.

### 3.5 DoD completeness

Gemini's DoD requires uplift on the 256e + 16e-draft path (item #3). Problems:
- That path requires SPRINT-025 multi-GPU to have shipped. As of this date it has not.
- No DoD item for the *single-GPU* path; if 025 slips, there is no fallback success criterion in the DoD.
- Item #5 "Memory accounting" requires "rebalance parameters" — but no DoD item requires those parameters actually produce a working layout, only that they are "documented."
- No DoD item for the `ngram-cache` config-driven draft length (the real upstream bug Codex caught).
- No DoD item for "the `temp=0` exactness gate is checked across N draft-max values" — same gap as Codex.

---

## 4. Cross-cutting technical errors and ambiguities

### 4.1 Same-vocab requirement

- Both drafts correctly state the requirement and correctly exclude Qwen.
- Both correctly accept that DSv4-Flash variants share vocab.
- **Codex** additionally requires the runtime compatibility check (`common_speculative_is_compat`) actually pass, not just "architecture name match." That is the right level of strictness.
- **Gemini** treats architecture match as sufficient — and uses MIN-8e (random weights, same vocab) which would pass compat but produce no acceptance. The same-vocab gate is *necessary, not sufficient*; Gemini conflates the two.

### 4.2 Draft model choice (real-weight constraint)

- **Codex**: AVG-16e is the only acceptable real-weight draft. Correct.
- **Gemini**: Uses MIN-8e in P2.1, then AVG-16e in P3. Mixing random-weight and real-weight in the same plan poisons the acceptance-rate analysis. P2's "expected failure" phase is wasted sprint days.
- Neither draft considers training/sourcing a smaller real-weight draft (e.g., a 4e or 2e variant). The intent doc explicitly puts that out of scope, which is reasonable.

### 4.3 VRAM with two models

- **Codex**: AVG-16e + AVG-16e single-V100 = ~36 GiB > 32 GiB. Marks not-a-ship-path. IQ2-64e + AVG-16e = ~46 GiB. Marks not-viable. 256e + AVG-16e on 8x V100 = "very likely memory-blocked." Conservative; safe; correct directionally.
- **Gemini**: Same math at the surface, but proposes "rebalance with `-sm layer` weights" without showing that the resulting per-GPU footprint fits within the SPRINT-025-baseline KV+scratch headroom. The mitigation is asserted, not computed.
- **Both** omit KV-cache, scratch, NCCL buffer accounting from the VRAM table. For the answer to even be plausible on 32 GiB cards, those need rows.

### 4.4 `--spec-type` / `--speculative-type` values

- Upstream: `--spec-type` with hyphenated values (`none`, `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`).
- **Codex**: correct; adds `--speculative-type` as alias proposal.
- **Gemini**: uses non-existent `--speculative-type` and underscore values; commands will not parse.
- **Intent doc**: lists `--speculative-type` and a mix of underscore (`ngram_simple`, `ngram_cache`, `ngram_mod`) and `--n-draft / --draft-max`. The intent doc should be corrected to track upstream, otherwise the next sprint draft will keep inheriting it.

### 4.5 `server-context.cpp` wiring

- Both drafts identify the right hooks (loading draft, slot init, completion timings).
- **Neither** notices `backend_sampling &= !(slot.spec && task.params.speculative.n_max > 0)` (L1197-1198) which disables CUDA backend sampling whenever spec is on. On M=1 launch-bound DSv4 + TURBOMIND, that means every accepted token round-trips a logits tensor through CPU sampling. Worth a measured-overhead line in P3/P4.
- **Neither** verifies that `common_speculative_is_compat(ctx)` returns true for a TURBOMIND-backed context. A `false` here silently disables spec.

### 4.6 `temp=0` same-output gate semantics

- **Codex**: requires `temperature=0, top_k=1, seed=0`. Right.
- **Gemini**: requires `temp=0` only, calls spec decode "mathematically exact." Insufficient. Exact only with deterministic tie-break (top_k=1 or equivalent) and bit-identical logits between the decode and verify paths. With FP16 MoE on V100, the latter is *not* free.
- **Both** should add a "tolerance window" gate: if the same-output gate fails at the bit level but matches at the top-3 candidate level for the divergence tokens, it is a numerical-determinism artifact, not a spec-decode correctness bug. A pure bit-equality gate with no tolerance turns measurement noise into a sprint stop. Recommend: "first N=200 generated tokens identical" + "if divergence, top-3 candidates at the divergence point overlap by ≥2."

---

## 5. What the drafts agree on (and is correct)

- DSv4-Flash family shares vocab; Qwen is out.
- `common/speculative.*` and `tools/server/server-context.cpp` already have the spec-decode plumbing; sprint wires from outside.
- M=1 launch-bound is the right place to attack.
- `temp=0` same-output is the correctness gate.
- REPORT-20 is the deliverable.

---

## 6. What the drafts agree on (and is wrong or shaky)

- "Median acceptance ≥ 0.50" as the floor for `ngram-cache` on mixed chat is **probably too high**. Published lookup-decoding numbers on general-chat workloads land 0.20–0.40. Codex hedges via "experimental" tier at 0.30–0.49; Gemini does not. Recommend: lower the headline floor to 0.40 for ngram-cache (keep 0.50 for draft-model path), or weight the prompt suite explicitly toward code-edit and document that constraint in REPORT-20.
- Both treat the SPRINT-025 multi-GPU path as a viable stretch. **Without measured headroom from SPRINT-025**, the 256e + draft layout is speculation in both plans.

---

## 7. Side-by-side: which plan to ship

| Axis | Codex | Gemini |
|---|---|---|
| Correct upstream flag names | ✅ | ❌ multiple |
| Correct flag values (hyphen vs underscore) | ✅ | ❌ |
| Real-weight draft constraint | ✅ | ❌ (uses MIN-8e) |
| Caught `ngram-cache` hardcoded 8 | ✅ | ❌ |
| Independent of SPRINT-025 | ✅ | ❌ (primary depends on it) |
| Honest VRAM verdict | ✅ | partial |
| Same-output gate (top_k=1) | ✅ | ❌ |
| Has a real fallback (P6) | ✅ | ❌ |
| Days estimate realism | optimistic | realistic but on wrong path |
| Risk table covers backend-sampling-off | ❌ | ❌ |
| Risk table covers MoE routing batch-shape | ❌ | ❌ |
| Risk table covers KV-cache rewind | ❌ | ❌ |

Codex is the plan to ship. Apply the corrections in §8 before execution.

---

## 8. Recommended edits to ship the Codex plan

Apply these in order before P0:

1. **Rename "self-speculative"** everywhere it refers to `ngram-cache`. It is draftless speculation, not self-spec. Self-spec means draft=target as a real second context, which §3.4 (correctly) rules out.
2. **Add to P0 a determinism sanity check**: with target-only at `temp=0, top_k=1, seed=0`, run the same 10-prompt suite twice and confirm bit-identical output. If target-only is not deterministic, the same-output gate is meaningless and the sprint blocks on numerical determinism first.
3. **Add to P0 a MoE routing batch-shape check**: target-only single-token decode vs target-only `n_batch=K+1` verify-shape decode of the same prompt; confirm logits/argmax match. This is the most likely correctness failure mode under TURBOMIND.
4. **Add to risks**: (a) CUDA backend sampling disabled under spec (`server-context.cpp:1197`), with an estimated per-token overhead measurement task in P3; (b) KV-cache rewind under TURBOMIND on rejection; (c) MoE routing nondeterminism between decode and verify batch shapes.
5. **Tighten P3 sweep**: drop `--draft-min=8 × --draft-max=8` (degenerate). Keep `{(4,8), (4,12), (8,12), (4,16), (8,16)}`.
6. **Lower the floor for `ngram-cache` to 0.40 median** OR explicitly weight the prompt suite toward code-edit (≥6 of 10). State which choice in P3, do not leave both ambiguous.
7. **Mark P4 as documentation-only** unless SPRINT-025 ships during 026. The "single-V100 two-model path" is already rejected in §3.4, so P4 has no runnable single-GPU configuration. Rename P4 to "Two-model wiring smoke + memory-blocker writeup."
8. **Inherit corrections to INTENT doc**: update `--n-draft` → `--draft-max`, `--speculative-type` → `--spec-type` (or "to be added as alias"), underscore values → hyphen values, and add a footnote that intent is reconciled against `common/arg.cpp` HEAD on this branch.
9. **Effort revision**: stretch the P0–P5 budget from 6 days to 9–10 days, given the new P0 determinism + MoE batch-shape checks and historical sprint slip.
10. **DoD addition**: "Same-output gate passed for at least 3 `--draft-max` values, not just the recommended one" + "VRAM peak measured per GPU via `nvidia-smi`, not estimated."

---

## 9. Items neither plan handles that a sprint reviewer should flag

- **No plan for accepting a no-lift outcome.** If the M=1 launch overhead dominates so heavily that even spec decode cannot recover (e.g. CPU sampling round-trip dominates), the right exit is "spec decode is a no-op on this hardware until SPRINT-024 launch amortization lands." Neither draft says that explicitly. Codex's `STOP` verdict gets closest.
- **No prompt-suite reproducibility commitment.** Both drafts say "10 fixed prompts" but neither commits them to a checked-in JSON file with hash. Codex's `tools/server/bench/prompts/speculative-026.json` is closer but does not require a hash in REPORT-20.
- **No baseline-drift control.** SPRINT-023 measured 16.6 t/s; if the AVG-16e baseline in P0 drifts to 14 t/s or 18 t/s, the lift ratio is over a different denominator. The sprint should require the P0 baseline to match SPRINT-023's number within ±10% before declaring a lift.
- **No statement on the `n-gram cache` warm-up policy.** Cold-cache acceptance is very different from warm-cache. The same prompt repeated wins; the same prompt first-time may not. REPORT-20 should report both.
- **No note on chat-template parity.** If target and draft use different chat templates (even with same vocab), the input token streams will not align prefix-for-prefix and acceptance will be artificially low. Should be a P0 check.

---

## 10. Bottom line

- **Codex** has the right structural choices and one real codebase-fixing find (`n_draft=8` hardcoded). It needs the corrections in §8 (rename "self-spec", add P0 determinism/MoE-batch checks, fix the risk table, tighten the sweep). With those, it is shippable.
- **Gemini** has the right conceptual map but multiple concrete CLI-flag errors, a wrong draft model choice in P2, a primary path that depends on an unshipped sprint, and an exactness gate that is too loose. Not shippable as written without substantial rework.
- **Intent doc** itself needs a small reconciliation pass against `common/arg.cpp` HEAD; otherwise the same flag-name errors will keep re-entering every draft.

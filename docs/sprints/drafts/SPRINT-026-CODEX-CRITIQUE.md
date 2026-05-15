# SPRINT-026 Codex Critique

Overall: the CLAUDE draft is materially stronger. It follows the intent's single-GPU-first shape, recognizes that most speculative plumbing already exists, and has a much better verification story. The GEMINI draft has several hard technical errors that make its primary path non-runnable or out of scope for the stated sprint.

Both drafts also inherit a few command-surface mistakes from the intent. Those need to be corrected before either plan is executed.

## Shared technical corrections

- The server flag is `--spec-type`, not `--speculative-type` (`common/arg.cpp:3517-3534`, `docs/speculative.md:115-140`, `tools/server/README.md:242`). Both drafts use the wrong flag name.
- `--spec-type` only accepts draftless modes: `none`, `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` (`common/arg.cpp:3517-3534`). There is no CLI value `draft`, and the CLI values are hyphenated, not underscored.
- A draft model is enabled by `--model-draft`; it is not selected via `--spec-type draft`. If a draft model and a draftless mode are both configured, the draftless implementation is tried first (`docs/speculative.md:9,106`, `common/speculative.cpp:852-896,995-1017`).
- The short draft override flag is `-otd`, not `-od` (`common/arg.cpp:2298-2303`, `tools/server/README.md:185`). The intent and GEMINI draft both use `-od`.
- `--first-cpu-moe-draft` does not exist in this tree. The available draft-side MoE flags are `--cpu-moe-draft` and `--n-cpu-moe-draft` (`common/arg.cpp:2326-2339`).
- `server-context.cpp` already loads the draft model, applies draft-side `devices`, `n_gpu_layers`, `tensor_buft_overrides`, stores `model_dft`, and calls `common_speculative_init` per slot (`tools/server/server-context.cpp:661-693,770-794`). Any sprint plan that treats this as missing core wiring is starting from the wrong premise.
- The compatibility check is stricter than "same architecture" but looser than "identical tokenizer required". Current code checks vocab type, BOS/EOS behavior, token text alignment, and allows close vocab sizes; incompatible pairs can be translated with `--spec-replace` (`common/speculative.cpp:50-103,198-205`). For this sprint, DSv4-family-only is still the sane practical constraint, but both drafts overstate the API requirement.
- The `temp=0` exactness gate should compare the accepted target output stream under identical request parameters. Seed is irrelevant once sampling is greedy. The thing that must match is generated token sequence / text, not timings, acceptance counters, or log lines (`tools/server/server-context.cpp:2908-2931`).

## CLAUDE draft

### Strengths

- Best alignment with the intent. It keeps the primary sprint on a single V100 and leaves `256e` as stretch-only (`CLAUDE` 79-94, 282-286), which matches the intent's "single-GPU first" constraint (`INTENT` 23, 47, 59).
- Correctly notices that most of the useful work is already upstream: existing speculative primitives, existing draft-model load path, and existing acceptance counters (`CLAUDE` 13-20).
- Strong verification structure. P0/P1/P2/P3/P4 is the right order: baseline and fit first, functional smoke second, exactness gate before performance claims, then tuning and report (`CLAUDE` 180-270).
- Better DoD than GEMINI. It requires exactness, acceptance/TPS measurement, n-gram comparison, and a report with reproducible commands (`CLAUDE` 319-330).
- Better risk posture. It explicitly calls out CPU-draft slowness, OOM risk, and the multi-instance TURBOMIND singleton hazard (`CLAUDE` 343-350).

### Weaknesses and technical errors

- The command surface is wrong in multiple places. `--speculative-type draft`, `--speculative-type ngram_cache`, and `ngram_cache` with underscores are all invalid (`CLAUDE` 33, 108, 120, 164, 186, 201, 250, 267, 322, 324). Current CLI is `--spec-type ngram-cache` etc., and draft mode is selected by `--model-draft`, not by `--spec-type draft` (`common/arg.cpp:3517-3534`).
- The "same-vocab invariant (hard constraint)" is overstated (`CLAUDE` 63-77). Current code does not require strict identical vocab objects; it allows compatible vocabs and even translation with `--spec-replace` (`common/speculative.cpp:50-103,198-205`). For the sprint, "stay in DSv4 family" is good advice, but the document should not describe that as the literal code contract.
- The primary model choice is shaky. `IQ2-64e` target + `AVG-16e` "draft on CPU" (`CLAUDE` 85-92) is not actually specified as full-CPU draft placement. `--cpu-moe-draft` only pushes MoE expert tensors to CPU; it does not by itself force the entire draft model, dense layers, or draft KV off GPU (`common/arg.cpp:2326-2339`, `server-context.cpp:669-684`). If this path is intended, it needs explicit draft-device settings such as `--device-draft none` and probably `-ngld 0`.
- Related: the VRAM table assumes the draft is fully host-resident (`CLAUDE` 142-155), but the proposed command line does not guarantee that. That makes the memory budget internally inconsistent.
- The exactness section is directionally right but slightly misframed. "Fixed seed" is not part of the real greedy guarantee (`CLAUDE` 161-168); identical sampler/grammar/stop settings are. Also "byte-identical" (`CLAUDE` 53) is more brittle than "identical generated token IDs / text".
- `P0.2` overfits to equality checks that the runtime does not require. Asserting exact `n_vocab` equality and a 100-token round-trip (`CLAUDE` 77, 185) is stricter than the actual compatibility gate and may reject usable pairs the runtime would accept.
- `P5` says the report will recommend which `--speculative-type` ships (`CLAUDE` 267), but that is the wrong surface again. The recommendation should be phrased as "ship with `--model-draft`" or "ship with `--spec-type ngram-cache`".

### Gaps in risk analysis

- Missing explicit risk that draftless and draft-model speculation can be mixed, and draftless has precedence. A misconfigured command can silently benchmark `ngram-*` instead of the draft model (`docs/speculative.md:9,106`, `common/speculative.cpp:852-896,995-1017`).
- Missing explicit risk that the proposed "CPU draft" can accidentally remain partially GPU-offloaded unless `--device-draft` / `-ngld` are set. This is the main feasibility risk in the chosen primary path.
- Missing explicit risk that `MIN-*` fixtures are unusable not just for quality but also for meaningful acceptance tuning. The document says this in architecture (`CLAUDE` 72-76) but it should be a first-class risk because fallback pressure can easily push the sprint there.
- Missing explicit risk that the exactness harness depends on the endpoint returning token IDs in a stable form. The plan assumes that exists but does not name the API field or fallback parsing path.

### Missing edge cases

- No explicit "wrong command but still boots" check. Given the repeated invalid `--speculative-type` spelling, the sprint should require one `--help`/startup validation pass for the chosen commands before any measurement.
- No explicit fallback if `IQ2-64e` fits target-only but not with the chosen context/KV settings once the draft is configured correctly.
- No explicit test for "draft configured but n-gram mode still taking precedence" when both are present.
- No explicit distinction between "identical output stream" and "identical response envelope". The former is the exactness requirement; the latter is not.

### DoD completeness

- Better than GEMINI, but not clean enough yet.
- DoD item 2 is not runnable as written because the command uses an invalid flag/value combination (`CLAUDE` 322).
- DoD should explicitly require a valid finalized command surface using current upstream flags: `--model-draft`, `-otd`, `--spec-type ngram-cache`, hyphenated names, and no `draft` value.
- DoD should explicitly require proof of draft placement for the primary path if the plan still claims "draft on CPU". Without that, the memory gate is ambiguous.
- DoD should explicitly say the exactness gate compares generated token IDs or generated text only.

## GEMINI draft

### Strengths

- Concise and readable. The high-level goal is easy to understand.
- It does preserve the key exactness claim at `temp=0` (`GEMINI` 57-61).
- It at least includes both n-gram and draft-model paths conceptually (`GEMINI` 13-17).

### Weaknesses and hard technical errors

- It violates the intent's scope. The intent says single-GPU first and treats `256e + draft` as stretch-only (`INTENT` 23, 47, 59). GEMINI makes `256e + AVG-16e draft on 8x V100` the main success path (`GEMINI` 15-17, 50-55, 93-104, 148-153, 178-180).
- The command surface is wrong throughout:
  - `--speculative-type` instead of `--spec-type` (`GEMINI` 40, 72, 153).
  - `ngram_cache` instead of `ngram-cache` (`GEMINI` 72, 153).
  - `-od` instead of `-otd` (`GEMINI` 53, 81, 102, 133).
  - `--n-draft 4` instead of existing aliases `--draft`, `--draft-n`, or `--draft-max` (`GEMINI` 102, `common/arg.cpp:3440-3445`).
  - `-ot 'exps=CUDA_TURBOMIND'` / `-od 'CUDA_TURBOMIND0'` is malformed override syntax; the right-hand side is the buffer type, the left-hand side is the tensor regex (`common/arg.cpp:2292-2303`).
- The draft-model choice is bad. `MIN-8e` as a draft candidate (`GEMINI` 89-91) directly conflicts with the intent's own note that `MIN-*` variants have random expert weights and are useless as drafts (`INTENT` 17-21). Running an acceptance sweep on a knowingly bad draft is wasted sprint time.
- The 256e VRAM plan is not credible. GEMINI itself notes `19.5 + 18 + KV/Scratch > 32 GiB` on GPU 0 (`GEMINI` 50-55), then treats manual rebalance as an ordinary mitigation (`GEMINI` 55, 97-103, 162-165). That is not a routine tuning pass; it is an out-of-scope hardware/layout problem for the primary sprint path.
- The file summary proposes changes to `server-context.cpp` and `common/arg.cpp` for plumbing that already exists (`GEMINI` 132-134). Draft model loading, draft-specific device settings, draft-specific tensor overrides, per-slot init, and draft counters are already wired (`tools/server/server-context.cpp:661-693,770-794,2101-2146,2908-2931`).
- The same-vocab requirement is overstated here too (`GEMINI` 42-46). For sprint planning this is a useful heuristic, but as a technical statement about the code it is inaccurate.

### Gaps in risk analysis

- No risk entry for the existing command-surface mismatch. This draft's primary commands will fail before any real speculative work starts.
- No risk entry for "MIN draft is useless by construction." That should be a high-likelihood, high-impact risk if P2 still uses `MIN-8e`.
- No risk entry for the already-wired nature of `server-context.cpp`. That matters because the sprint could waste time modifying stable plumbing instead of measuring the existing path.
- No risk entry for draftless precedence when both draft and draftless implementations are configured.
- No risk entry for the fact that manual rebalance on the heaviest 256e shard may still fail even if total node VRAM is ample.

### Missing edge cases

- No single-GPU shippable fallback that remains within the intent if `256e + draft` never fits.
- No explicit proof step that the chosen commands are valid against current `llama-server --help`.
- No explicit check that the draft model is using a separate context but the existing target-side sampling still owns exactness.
- No explicit edge-case handling for "n-gram path ships, draft-model path deferred", even though that is the most plausible outcome from the stated risks.

### DoD completeness

- Incomplete and partially invalid.
- DoD item 1 is underspecified because "boots with both target and draft models" does not say whether the command line is using valid upstream flags.
- DoD item 3 anchors success to `256e + 16e-draft` (`GEMINI` 150), which conflicts with the intent's single-GPU-first scope and makes the sprint unnecessarily dependent on the hardest layout.
- DoD never requires proving that the already-existing server plumbing was sufficient without invasive core changes.
- DoD never requires a valid single-GPU measured path, which is the intent's main branch.

## Recommendation

- Use the CLAUDE draft as the base.
- Before merging anything from it, fix all command-surface issues:
  - `--spec-type`, not `--speculative-type`
  - `ngram-cache`, not `ngram_cache`
  - `-otd`, not `-od`
  - no `draft` value for `--spec-type`
- Rewrite the primary feasibility story around a real, explicit placement decision. If the plan wants a CPU draft, it must say how the full draft model is kept off GPU, not just the MoE experts.
- Keep the exactness gate, but phrase it as "identical generated token IDs / text at `temp=0` under identical request params".
- Drop GEMINI's `MIN-8e` acceptance sweep and `256e`-first success gate. Those are the two biggest scope and correctness errors in that draft.
- If a merged sprint needs one practical ship target today, the best shape is:
  - single-GPU target first
  - n-gram path fully measured
  - draft-model path only if the placement story is explicit and memory-feasible
  - `256e + draft` stretch-only

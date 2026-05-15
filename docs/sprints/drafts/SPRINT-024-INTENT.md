# SPRINT-024 — Intent

## Seed prompt

Land the SPRINT-024 perf work that SPRINT-023 set up. The bottleneck is now
known: kernel-launch overhead at M=1, not memory bandwidth or per-kernel
compute. The C ABI has a `ggml_turbomind_mul_mat_grouped` entry point that
collapses all top-k expert linears in one launch per layer. Wire it through
the ggml-cuda dispatch, measure, and either ship (if the gate is met) or
diagnose-and-extend.

Secondary: keep activations FP16 across the FFN boundary, and run the new
path against a non-`MIN-*` model so quality is verifiable.

## Orientation summary

1. **SPRINT-023 closed at 3-3.6× decode speedup vs CPU MoE baseline**
   (16.2-16.6 t/s on V100 across MIN-8e through MIN-32e). Throughput is
   essentially flat in model size — clear signature of launch-bound regime
   at M=1.
2. **`libggml-turbomind.so` already exports `ggml_turbomind_mul_mat_grouped`**
   (api.cc lines ~513-619). Takes the input activation pointer, per-expert
   token_indices, expert_offsets, num_experts, and a device array of
   per-expert (weight, scale) `StridedPtr` structs. We prototyped the
   StridedPtr layout during P2.3 debugging.
3. **`ggml_cuda_mul_mat_id` currently uses per-expert slicing**
   (`ggml-cuda.cu` ~lines 2670-2785): sorts tokens by expert host-side, then
   calls `ggml_cuda_mul_mat` once per (expert, sorted-tokens) pair. Each
   call lands in `ggml_cuda_mul_mat_turbomind` — 6 active experts × 43
   layers × decode rate = ~4 100 launches/sec.
4. **The activation cast pair (FP32→FP16→FP32)** runs on every dispatch in
   `ggml_cuda_mul_mat_turbomind`. Moving the FP16 boundary upstream is a
   small follow-on win.
5. **No VISION.md** — planning from scratch given prior sprint context.
   Ledger script absent; we'll skip that step.
6. **MIN-* models produce gibberish on any path** (random weights, per user).
   Need a non-MIN variant or a quantized-256e that fits 32 GiB to verify
   end-to-end quality.

## Active followups from SPRINT-023

| # | What | Severity | Target | Files |
|---|---|---|---|---|
| F-01 | Grouped MoE per-layer dispatch via `mul_mat_grouped` | Important | **SPRINT-024 P1** | `ggml-cuda-turbomind.{cu,cuh}`, `ggml-cuda.cu` |
| F-02 | Keep activations FP16 across FFN boundary | Nice-to-have | SPRINT-024 P3 | `ggml-cuda.cu` (FFN routing) |
| F-03 | Real-model quality verification | Important | **SPRINT-024 P2** | measurement-only |
| F-04 | Multi-slot decode (parallel slots / speculative) | Strategic | ≥SPRINT-025 | dispatch surface |
| F-05 | Use ggml CUDA pool for buffer alloc | Nice-to-have | when needed | `ggml-cuda-turbomind.cu` |
| F-06 | Per-expert scales alignment | Nice-to-have | when needed | `ggml-cuda-turbomind.cu` |
| F-07 | Update absolute-tolerance test gates | Nice-to-have | next correctness sprint | tests |

## Active deferreds from SPRINT-023 (still applicable)

- **#1 — Decode TPS perf gate**: SPRINT-024's first job. SPRINT-023 P5 measured
  16.6 t/s at the baseline launch-per-expert rate. With grouped MoE the
  theoretical ceiling is ~58 launches/token instead of 464. SPRINT-024 should
  set a concrete numeric gate.
- **#2 — Grouped MoE dispatch**: F-01 above; the C ABI exists, no new kernel
  design needed.
- **#3 — F8_E4M3_B128 dense layers via turbomind**: small follow-on once the
  MoE path produces real numbers. Trivial extension (regex tweak in `-ot`),
  but only worth it if dense compute is a measurable fraction post-grouped.
- **#7 — PCIe expert streaming Plan B**: only triggers if grouped MoE doesn't
  move the needle decisively. Almost certainly not needed.

## Relevant codebase areas

- `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}` — add the grouped helper
  `ggml_cuda_mul_mat_grouped_turbomind`. Resolves per-expert pointer arrays
  on device, calls `ggml_turbomind_mul_mat_grouped`, casts.
- `ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_cuda_mul_mat_id`:
  - Predicate: if `src0` is `CUDA_TURBOMIND` AND the type is in our supported
    set, dispatch to the new grouped helper directly. Skip the existing
    per-expert slicing path.
  - Build the (token_indices, expert_offsets) the same way the existing
    fallback does, but on-device.
- `ggml/vendor/turbomind/api.cc` `ggml_turbomind_mul_mat_grouped`:
  - We already fixed Bdesc/Vdesc orders + packed-ld derivation in SPRINT-023
    P2. The grouped path uses `Bdesc.ld = 0` for per-expert pointer-array
    dispatch — the kernel reads StridedPtr per gemm_id. The StridedPtr.stride
    field needs to be the **packed** ld (K*32 for HMMA_884 OPERAND_B Pack_M=1),
    not the unpacked K.
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` — extend `extra` struct if
  needed so the dispatch helper can rebuild per-expert StridedPtr arrays
  cheaply (it already has `n_experts` and `scales_per_expert`).

## Constraints

- **V100 sm70**: no Hopper kernel paths. Stick with `Config_E4M3<kColMajor, 0>`
  and `Config_MXF4<kColMajor, 0>` registry entries from SPRINT-023.
- **No upstream PRs to ggml/llama.cpp**: AGENTS.md says llama.cpp doesn't
  accept AI-generated PRs. Work lands on the user's private fork
  `rapatel0/deepseek-v100-build` (branch `sprint-022-dsv4-integration`).
- **Commit after every phase, push to origin**: durable since SPRINT-023.
- **Stable C ABI**: `ggml-turbomind-api.h` is the contract between the .so
  and ggml-cuda. Don't change without bumping `ggml_turbomind_api_version`.
- **FP16 precision tolerance gate**: P2.3's relative-error + 2-ULP-floor
  pattern is the gate template. SPRINT-015 P2 absolute thresholds are
  unusable at the output magnitudes inference produces.

## Success criteria

1. **Functional**: A grouped-MoE dispatch path exists end-to-end. Loading
   DSv4-Flash-MIN-32e with `-ot exps=CUDA_TURBOMIND0` exercises the grouped
   path (verifiable via debug log or counter) and produces stable output.
2. **Correctness**: Output matches the per-expert path within FP16 ULP at
   the same prompt/seed/temperature. The MIN-* models are random-weight
   fixtures so the gate is "results identical to the legacy per-expert
   dispatch on the same model" — not absolute output quality.
3. **Performance gate**: Decode TPS on MIN-16e or MIN-32e ≥ **24 t/s**
   (≥ 1.45× over SPRINT-023's 16.6 t/s baseline). Stretch target **30 t/s**.
   Rationale: launch count drops 6× (n_active_experts), but per-launch cost
   is finite and FP16 cast pair stays; conservative model is ~1.5×.
4. **Quality verification on a non-MIN model**: One of these:
   - DSv4-Flash-AVG-16e (18 GiB, real expert weights per user) — preferred.
   - DeepSeek-V4-Flash-IQ2-64e (28 GiB) — fallback.
   - Compare deterministic 32-token completions vs CPU MoE baseline on a
     small prompt set. Gate: identical leading tokens (≥75% match) under
     greedy decode.
5. **No regression**: Tests in `ggml/vendor/turbomind/test_correctness.cpp`
   still pass.

## Verification strategy

- **Functional**: Add a one-liner debug print on first dispatch from the
  grouped path. Manual smoke load to confirm.
- **Correctness regression**: Extend `test_correctness.cpp` with a
  per-expert vs grouped comparison: pack the same weight as N experts,
  run both dispatch paths with the same activations, compare D within
  FP16 ULP.
- **Perf**: `llama-bench -p 128 -n 32 -r 3` on MIN-8e, MIN-16e, MIN-32e.
  Compare against SPRINT-023 P5 numbers in the same table format.
- **Quality**: `llama-server` + a fixed seed prompt; greedy decode 32 tokens
  on both grouped TURBOMIND and CPU MoE legs; diff token IDs.

## Uncertainty assessment

| Factor | Level | Why |
|---|---|---|
| Correctness | **Medium** | The grouped C ABI exists and was wired up against pack at P2.3; we know the StridedPtr layout works at num_experts=1. Risk: per-expert stride/scale offset arithmetic on the device-side pointer array. |
| Scope | **Low** | Bounded — F-01 is the main lever, with F-02/F-03 as scoped sub-phases. |
| Architecture | **Low** | Extends the SPRINT-023 dispatch surface; no new abstractions needed. Buffer type, extra struct, dispatch predicate all in place. |

## Open questions (for the interview)

1. **Hard perf gate** — Is "≥24 t/s decode on MIN-16e" the right gate? Or
   prefer a relative bar ("≥1.5× over SPRINT-023 baseline")? Or even softer
   ("ship if no regression vs P5 baseline; do follow-on if disappointing")?
2. **Quality verification depth** — Is "matches CPU MoE leading 75% of
   tokens on a small prompt set" sufficient? Or do we want a perplexity
   sweep on a real eval set?
3. **F-02 (FP16 boundary)** — Include in SPRINT-024 or defer to SPRINT-025?
   Bytes-through-DRAM is the win; estimate 5-10% additional. Adds dispatch
   complexity in `ggml_cuda_mul_mat_id`.
4. **F-03 quality model** — AVG-16e (preferred) vs IQ2-64e — pick one for
   the gate, or do both?

## Vision context

No `docs/sprints/VISION.md`. Prior context: SPRINT-022 made DSv4-Flash
operational at 4.28/4.73 t/s; SPRINT-023 added the turbomind path infra at
~16 t/s; this sprint cashes the launch-amortization check that SPRINT-023
left on the table.

## What this sprint is NOT

- New kernel design. Turbomind's `Config_E4M3<kColMajor, 0>` + `Config_MXF4<kColMajor, 0>`
  on sm70 are the kernels; we're just amortizing their launches.
- Multi-GPU TP. Single V100.
- Speculative decoding. F-04 stays at SPRINT-025+.
- New buffer types or ABI changes. The CUDA_TURBOMIND surface and C ABI
  stay frozen.
- A perplexity / quality sweep against the real 256e model. That needs a
  multi-GPU box and is out of scope.

# Codex Critique: `SPRINT-020-GEMINI-DRAFT.md`

## Overall Read
The Gemini draft has the right architectural instinct: it recognizes that `v12_ms3` is likely near a local ceiling and that `research/lmdeploy/src/turbomind/kernels/gemm/` is the only in-repo candidate likely to produce a materially different `sm_70` outcome. It is also better than a pure essay; it names phases, files, and concrete TF goals.

The main problem is execution rigor. As a sprint plan, it is still too loose at the exact points where SPRINT-020 needs hard gates: apples-to-apples measurement, fallback behavior if Turbomind underperforms, runtime ownership, and definition of what actually lands if the sprint closes with a ceiling proof instead of a breakthrough.

## Strengths
- **Good primary bet in P1/P2:** Choosing the Turbomind path is coherent with the repo state. The named source area, `research/lmdeploy/src/turbomind/kernels/gemm/`, is real and already contains `mainloop_sm70.h`, `iterator_sm70.h`, `scheduler_sm70.cuh`, `registry.cu`, and the commented `gemm_bench` target in `CMakeLists.txt`.
- **P0 correctly includes SPRINT-019 debt:** The call to clear `v12s` sanitizer debt before shipping new dispatch behavior is the right ordering. Targeting `tools/tc-grid/src/main.cu` for `--n-list` / `--k-list` also matches the current harness bottleneck.
- **The draft keeps DSv4 MoE shapes visible:** It does not reduce the sprint to only `M=2048, N=K=7168`. Calling out `7168x18944`, `18944x7168`, and `2048x7168` is directionally correct.
- **There is a real headline decision in the document:** The draft does not hide behind vague “investigate further” language. It states explicit targets: `>= 50 TF` at `M=2048`, `>= 25 TF` at `M=64`, and a Turbomind ceiling proof if the architecture still tops out near `<= 41 TF`.

## Weaknesses
- **Thresholds are inconsistent across phases.** Section 1 says success is `>= 50 TF` or a ceiling proof around `<= 41 TF`. P1 says proceed to P2 at `>= 45 TF`. The DoD later says “Turbomind ceiling documented at `< 45 TF`.” Those are three different gates for the same decision. The sprint needs one numeric contract, not `50 / 45 / 41` depending on section.
- **P2 overcommits to `launch_int8.cu` too early.** “Add `LAUNCH_TURBOMIND` macros to `tools/tc-grid/src/launch_int8.cu` (Version 70)” assumes a wholesale dispatcher integration before proving data-layout compatibility. That is the highest-risk part of the plan, and the draft skips the intermediate bridge step. The Codex draft is stronger here: it splits native `gemm_bench` bring-up from a dedicated bridge file, `tools/tc-grid/src/launch_turbomind_int8.cu`.
- **P4 names the wrong ownership level for runtime integration.** “Move the champion kernels into the actual model serving code” is too vague, and the file list makes it worse by naming `common/reasoning-budget.cpp` as a placeholder wiring target. That file exists, but it is unrelated to GEMM runtime ownership. The real runtime surfaces are in Turbomind-side files like `research/lmdeploy/src/turbomind/models/llama/LlamaLinear.cu` and `research/lmdeploy/src/turbomind/models/llama/moe_ffn_layer.cc`.
- **The M=64 target is not backed by a concrete plan.** The draft sets `>= 25 TF` at `M=64`, but P1-P4 are almost entirely framed around the Turbomind port path. There is no explicit small-M fallback gate such as “retain `v12s_ks8` at `21.55 TF` if Turbomind does not beat it.” That makes the `25 TF` target read aspirational rather than planned.
- **P3 dispatch work is underspecified.** “Implement the per-(M, N, K) dispatch table in `launch_int8.cu`” does not say whether the table is hand-maintained, benchmark-derived, or imported from Turbomind artifacts. That matters because `research/lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.{h,cu}` already exists. The Gemini draft risks inventing a second dispatcher instead of reusing the one in the candidate runtime.

## Gaps In Risk Analysis
- **No explicit apples-to-apples comparison risk.** P1 uses native `gemm_bench`, while P2/P3 use `tc-grid`, but the risk section never names the possibility that Turbomind can look good in `gemm_bench` and lose once measured with `tc-grid`’s quantization, reference path, and tolerances. That is the central methodological risk of the whole sprint.
- **No explicit layout / quantization compatibility risk.** The draft assumes `tools/tc-grid` tensors can be consumed by Turbomind kernels after a wrapper in `tools/tc-grid/kernels/turbomind_wrapper.cuh`. That may be false, or only true with a repack step that invalidates the GEMM comparison.
- **No explicit runtime-ownership risk.** P3 puts dispatch rules into `tools/tc-grid/src/launch_int8.cu`, while P4 says production wiring goes into “the DSv4-flash inference backend.” The risk section never addresses the possibility that lab dispatch and runtime dispatch diverge immediately.
- **No explicit same-wall risk.** The overview correctly says `v12_ms3` is SMEM-bandwidth-bound (`mio_throttle`). The risks do not name the possibility that Turbomind on V100 is bound by the same Volta SMEM/L1 limits and therefore cannot realistically reach `>= 50 TF` despite a different mainloop.
- **No explicit build-surface risk for re-enabling `gemm_bench`.** `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt` currently has the `gemm_bench` block commented under `BUILD_TEST`. The draft notes build complexity in general, but not the concrete risk that reviving this target pulls in `nvbench`, `core`, `reference.cu`, and test-only dependencies that are not part of the normal path.

## Missing Edge Cases
- **`M=1` and `M<64` decode cases are absent.** The draft uses `M=64` as the small-M proxy, but if P3 is about production dispatch, then `M=1`, `M=2`, `M=4`, and `M=8` are the real edge cases. A dispatch table that only gates on `M=64` and `M=2048` is incomplete.
- **No explicit fallback for asymmetric wins without square-shape wins.** P1’s decision gate is square-shape-centric: `>= 45 TF` continues, `<= 41 TF` pivots. But the sprint intent allows the asymmetric DSv4 shapes to be first-class evidence. The draft does not say what happens if Turbomind is only, for example, `39-41 TF` at `7168x7168` but materially faster at `18944x7168`.
- **No shape-validity edge cases for the native bench.** `research/lmdeploy/src/turbomind/kernels/gemm/test/models.h` is a model-shape catalog, not a generic `tc-grid` shape surface. The Gemini draft says “add DSv4 MoE shapes,” but it does not call out the need to verify those shapes are actually legal through the existing `gemm_bench` axes and `group_size = 128` assumptions.
- **No failure mode for sanitizer findings in P0.** “Fix any discovered race conditions” is not enough. If `v12s` fails `racecheck` for `KSPLIT={2,3,5,8,16}`, the sprint needs an explicit branch: block deployment integration, limit valid `KSPLIT` values, or revert `v12s` from production candidacy.
- **No edge case for benchmark artifacts becoming non-reusable.** If P3 produces per-shape winners manually in `launch_int8.cu`, but P4 later needs imported runtime selections, the measured results may not map cleanly to real runtime descriptors. The draft should call out this “lab label vs runtime key” mismatch.

## Definition Of Done Completeness
- **The DoD does not define what lands if Turbomind loses.** “`>= 50 TF` reached OR Turbomind ceiling documented at `< 45 TF`” is not enough. If the sprint closes on a ceiling proof, the DoD should still require the concrete P0 follow-ups to land: `v12s` sanitizer cleanup, `--n-list` / `--k-list`, and a production dispatch outcome for the existing `v12_ms3` / `v12s` split.
- **The DoD does not require artifact quality.** There is no requirement for median-of-5 CSVs, named `ncu` exports, or a documented metric pack. A sprint centered on a ceiling proof should require archived evidence under `tools/tc-grid/docs/` and/or `tools/tc-grid/docs/ncu/`.
- **The DoD does not define the runtime owner.** “Per-(M, shape) dispatch wired into DSv4-flash” is too broad. The sprint should state whether the owner is `tools/tc-grid/src/launch_int8.cu`, Turbomind `registry.cu` plus `dispatch_cache.{h,cu}`, or some separate DSv4-serving layer. Without that, P3 and P4 can both be “done” while implementing two different dispatch systems.
- **The DoD omits build-gating requirements.** If `gemm_bench` is re-enabled in `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`, the DoD should require that the path is guarded and does not silently alter default builds on non-benchmark configurations.
- **The DoD does not restate the correctness thresholds.** The draft mentions `rel <= 1e-2 ∧ p99 <= 1.0 ∧ maxabs <= 5.0`, which is good, but the DoD should also say which phases must satisfy those gates: native Turbomind bench only, `tc-grid` bridge, and runtime integration. Right now the scope of the gate is ambiguous.

## Comparison To `SPRINT-020-CODEX-DRAFT.md`
- The Codex draft is stronger on phase separation. It distinguishes:
  - native Turbomind bench bring-up in `research/lmdeploy/src/turbomind/kernels/gemm/test/gemm_bench.cu`
  - `tc-grid` bridge work in `tools/tc-grid/src/launch_turbomind_int8.cu`
  - runtime follow-through in `LlamaLinear.cu` / `moe_ffn_layer.cc`
  Gemini compresses those into a faster narrative, but the compression hides risk.
- The Codex draft is also stronger on fallback structure. Its P5 explicitly defines what to land if Turbomind does not clear the threshold. Gemini has the right instinct to pivot, but not a complete close-out plan.
- Gemini is slightly stronger rhetorically on the “architectural break” framing, but for execution the Codex draft is the better base document because it is clearer about ownership and intermediate gates.

## Bottom Line
The Gemini draft is a solid strategic memo and a plausible starting point, but it is not yet a tight sprint execution plan. The biggest fixes are:

- unify the TF decision thresholds around one contract
- split native benchmark, bridge, and runtime integration into separate gates
- replace placeholder runtime targets like `common/reasoning-budget.cpp` with the real Turbomind call sites
- define the fallback close if Turbomind loses
- make the DoD require concrete artifacts and an explicit runtime dispatch owner

With those changes, the draft could become a viable SPRINT-020 plan. As written, it still leaves too much ambiguity at the exact handoff points that decide whether the sprint produces a benchmark, a bridge, or a shippable runtime path.

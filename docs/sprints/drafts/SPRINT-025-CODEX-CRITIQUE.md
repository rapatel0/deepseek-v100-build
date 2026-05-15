# SPRINT-025 Codex Critique

Overall: the CLAUDE draft is materially stronger. It is closer to the intent, it correctly focuses on `-sm layer` as the shippable path, and it notices the real multi-GPU hazard in the current `CUDA_TURBOMIND` loader. The GEMINI draft has several hard technical errors that would make the sprint fail if executed as written.

## CLAUDE draft

### Strengths

- Strongest alignment with the intent: it keeps the sprint centered on "load the 156 GiB model on 8x V100, verify decode, measure TPS" rather than turning the sprint into a generic TP feature sprint.
- Correctly identifies the current single-process `TmLib` singleton as the main multi-GPU correctness risk in `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu:46-101`.
- Good phasing. P0/P1/P2/P3 is a sensible order: hardware and NCCL first, small-model multi-GPU smoke next, then the 256e load.
- Better risk posture than the GEMINI draft. It at least acknowledges that NCCL can be built but never exercised on the `-sm layer` hot path, and that the turbomind library may have a deeper singleton than the wrapper.
- Better DoD framing: it treats this sprint as a first successful landing and baseline, not as a full optimization sprint.

### Weaknesses and technical errors

- `LLAMA_SPLIT_MODE_TENSOR` is described as a "legacy synonym" of `LAYER` (`CLAUDE` lines 86-91). That is incorrect. Current code gives `TENSOR` its own meta-device path in `src/llama.cpp:945-990`, requires flash attention in `src/llama-context.cpp:2957-2964`, and rejects quantized KV with `TENSOR` in `src/llama-context.cpp:2966-2968`. This matters because the draft uses `TENSOR` as a conceptual shorthand when it is actually separate plumbing.
- The draft never calls out the hard `deepseek4` block on row split. Current code throws on `LLAMA_SPLIT_MODE_ROW` for this architecture in `src/llama-model.cpp:770-771`. That means row-split comparison is not just "deferred"; it is impossible without first changing model plumbing.
- The `-ot` examples are wrong in multiple places: `-ot 'exps=CUDA_TURBOMIND[0-7]'` (`CLAUDE` line 33) and `-ot 'exps=CUDA_TURBOMIND0,exps=CUDA_TURBOMIND1'` (`CLAUDE` line 233). In `common/arg.cpp:243-285` and `2292-2294`, the left side is the tensor-name pattern and the right side must be an exact buffer-type name. `CUDA_TURBOMIND[0-7]` is not a valid buffer type, and repeating the same left-hand pattern twice does not route half the experts to one buffer and half to another.
- The NCCL troubleshooting path says to inspect `cmake/Modules/FindNCCL.cmake` (`CLAUDE` line 210). In this repo the relevant file is `ggml/cmake/FindNCCL.cmake`, and `ggml/src/ggml-cuda/CMakeLists.txt:184-192` already wires `find_package(NCCL)` plus `NCCL::NCCL`.
- `llama-server --help` is a weak NCCL init check (`CLAUDE` lines 213, 219). The important validation is an actual CUDA backend initialization path or a harness that calls `ggml_backend_cuda_init()` / `ggml_backend_cuda_allreduce_tensor()`. Help output is not a reliable proxy for `ncclCommInitAll`.
- The VRAM section overstates the benefit of `q8_0` KV. `CLAUDE` claims a 4x reduction (`lines 159, 167`), but `q8_0` is 8-bit quantized storage, not a quarter of FP16 by definition. In-tree definitions show `F16` is 2 bytes per element (`ggml/src/ggml.c:646-650`) while `Q8_0` stores 32 int8s plus one half scale (`ggml/src/ggml-common.h:251-256`), so the reduction is much closer to 2x than 4x. The draft should not budget 8K context from that assumption.

### Gaps in risk analysis

- Missing explicit risk that `-ot` tensor overrides disable scheduler pipeline parallelism. Current scheduler logic only enables layer-mode pipeline parallelism when `!model.has_tensor_overrides()` in `src/llama-context.cpp:316-321`. Since this sprint relies on `CUDA_TURBOMIND` overrides, the performance and utilization expectations for `-sm layer` change materially.
- Missing risk that the turbomind library itself may be process-global even if the wrapper becomes per-device. The draft hints at this, but it should be promoted from caveat to sprint-level risk because it can invalidate P2 entirely.
- Missing risk around load imbalance under `-sm layer`. The draft discusses `-ts`, but does not tie it to the actual fact that layer assignment and VRAM headroom are per-device constraints, not just total-cluster constraints.
- Missing risk that the NCCL build path may succeed while the model path never calls allreduce in the shipped configuration, leaving the allreduce plumbing effectively unvalidated unless the synthetic harness is mandatory.

### Missing edge cases

- No explicit check for one-GPU regression in weight upload plus multi-GPU regression in expert dispatch after the loader refactor beyond output equality. A log-based "which device packed which expert tensor" check would catch silent misplacement earlier.
- No explicit plan for the "layer split loads, but one GPU gets the heaviest layers and OOMs" case. The draft mentions `-ts` only as a future tuning lever, but the load test itself needs a fallback procedure.
- No explicit validation that the actual DeepSeek4 expert tensor names are matched by the proposed regex. The draft uses shorthand `exps`, but the sprint should verify the real names before making `-ot` part of DoD.

### DoD completeness

- Better than GEMINI, but still incomplete.
- DoD should explicitly require "shipped path uses `LLAMA_SPLIT_MODE_LAYER` only" unless the row-split architecture block is removed first.
- DoD should require proof that `CUDA_TURBOMIND` dispatch happened on more than one GPU using actual runtime counters or logs, not just memory allocation.
- DoD should require measured per-GPU memory at both model load and decode peak. Load-time buffer sizes alone are not enough for the VRAM question the sprint is trying to answer.

## GEMINI draft

### Strengths

- High-level objective matches the intent: land the full 256e model on 8x V100, wire NCCL, and collect a first baseline.
- The phase structure is easy to read and the document stays concise.
- It at least distinguishes `LAYER` and `ROW` conceptually, which is better than ignoring split mode entirely.

### Weaknesses and hard technical errors

- The biggest issue: it plans to verify and ship `LLAMA_SPLIT_MODE_ROW` repeatedly (`GEMINI` lines 22, 37-45, 74-89, 116-118, 177-180, 219-221), but current code throws for DeepSeek4 in `src/llama-model.cpp:770-771`. As written, P1, P3.3, DoD item 3, Risk 2, and Open Question 3 are all built on a mode that is not currently implemented for this architecture.
- It misses the real multi-GPU `CUDA_TURBOMIND` problem entirely. The current wrapper is still a single global `TmLib` with `shutdown()` / `init(device)` rebinding in `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu:46-101`. The draft assumes per-device buft registration is enough (`GEMINI` lines 47-51, 94-101), but the bug is in loader state, not just buffer-type enumeration.
- The `-ot` routing syntax is wrong (`GEMINI` lines 49, 97). `common/arg.cpp:243-285` resolves the right-hand side by exact buffer-type name, so `CUDA_TURBOMIND[0-3]` and `CUDA_TURBOMIND[0-7]` are invalid. This is not a minor syntax detail; the plan's P2 gate depends on it.
- The file summary is technically wrong. `ggml/src/ggml-cuda/CMakeLists.txt` already has `find_package(NCCL)` and `NCCL::NCCL` (`ggml/src/ggml-cuda/CMakeLists.txt:184-192`), and the repo already contains `ggml/cmake/FindNCCL.cmake`. This is existing wiring, not new sprint work.
- `src/llama-model.cpp` does not need changes just to "ensure multi-GPU `CUDA_TURBOMIND` registration" (`GEMINI` line 162). Extra bufts are already enumerated via the backend-reg proc-address mechanism, and `ggml_backend_cuda_turbomind_buffer_type(int device)` already names `CUDA_TURBOMIND0..N`.
- The dependency section says SPRINT-024 results are required (`GEMINI` line 213), which conflicts with the intent. The intent explicitly says SPRINT-024 is not blocking because SPRINT-023 already established the per-device buft surface and SPRINT-025 can run with either per-expert or grouped MoE dispatch.
- The VRAM math is too hand-wavy and at times plainly wrong. "100 GiB slack is plenty" (`GEMINI` line 193) reasons from total node memory instead of per-device headroom. The real problem is whether the heaviest-shard GPU fits after weights, packed scales, KV, temp buffers, and context overhead. Total-cluster slack does not answer that.
- The context claim in Open Question 2 is unsupported: "32k in FP16, or 64k in q8_0" (`GEMINI` line 220) is not grounded in the current per-GPU budget and ignores that q8_0 KV is not a 4x win. This should not appear in a sprint plan without measured bytes.

### Gaps in risk analysis

- No risk entry for the existing single-loader multi-device bug, which is the most obvious correctness blocker in the current code.
- No risk entry for the hard `deepseek4` row-split restriction, even though the draft depends on row split in multiple phases.
- No risk entry for invalid `-ot` syntax / tensor-name matching, despite P2 and DoD depending on expert placement.
- No risk entry for pipeline parallel being disabled when tensor overrides are present (`src/llama-context.cpp:316-321`). That affects expected utilization and could easily confuse measurement results.
- No risk entry for allreduce never being exercised by the shipped `-sm layer` path even if NCCL links successfully.

### Missing edge cases

- No small-model multi-GPU smoke before attempting the 156 GiB load. That makes debugging much slower if `CUDA_TURBOMIND` placement is wrong.
- No explicit single-GPU regression check after any multi-GPU changes.
- No fallback plan if one GPU OOMs because of uneven layer distribution.
- No explicit requirement to verify actual expert tensor names and actual per-GPU placement in logs.

### DoD completeness

- Incomplete and partially invalid.
- DoD item 3 requires a row-split path that the code currently rejects for DeepSeek4.
- DoD never requires fixing or validating the per-device turbomind loader state.
- DoD never requires proving that multiple GPUs actually executed turbomind dispatches.
- DoD should include a concrete VRAM accounting artifact, not just a generic "memory breakdown."

## Recommendation

- Use the CLAUDE draft as the base.
- Correct the split-mode section so it reflects current code reality: `ROW` is blocked for DeepSeek4 today, and `TENSOR` is separate plumbing, not a synonym for `LAYER`.
- Rewrite all `-ot` examples using exact buffer-type names on the right-hand side and verified DeepSeek tensor-name regexes on the left.
- Keep the multi-GPU `CUDA_TURBOMIND` loader fix as a first-class sprint item.
- Tighten VRAM budgeting around per-GPU peak bytes, not total-node slack, and remove unsupported context-length claims.
- Make an allreduce harness mandatory if NCCL is in scope, because the shippable `-sm layer` path will not validate row-split/allreduce behavior by itself.

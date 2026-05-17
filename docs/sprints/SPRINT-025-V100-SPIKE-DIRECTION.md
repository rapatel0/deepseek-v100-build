---
title: SPRINT-025 V100 Runtime Spike Direction
date: 2026-05-17
status: completed spike direction
scope: DeepSeek V4 Flash performance strategy after P9 TurboMind expansion failure
---

# V100 Runtime Spike Direction

## Decision

The next spike should stay in the llama.cpp fork, not pivot to DS4 yet.

This is not a sunk-cost decision. It is based on the assets each path has today:

- llama.cpp already loads the target `DSv4-Flash-256e-fixed.gguf`, shards it
  across 8x V100, and has a coherent exps-only TurboMind path.
- DS4 has a cleaner DeepSeek-only graph, but it cannot load the current
  FP4/FP8-style GGUF and its CUDA backend is single-device.

The useful DS4 information is architectural: fixed DeepSeek graph boundaries,
KV discipline, HC/norm fusion points, and MoE/shared-expert fusion targets. The
fastest V100 route is to bring those ideas into the existing llama.cpp fork
where the model format, sharding, and deployment path already work.

## Current Failure Signal

P9 tested pure launch-flag expansion:

- exps + shexp + output: broken, empty content
- exps + shexp only: broken, empty content
- exps only: coherent

The important correction is that TurboMind is not currently a general drop-in
matmul backend in this fork. It is a working stacked routed-expert backend.
Shared experts use the single-tensor path, not the grouped MoE path, and routing
any shared-expert projection through that path is enough to break coherence.

Follow-up isolation on `DSv4-Flash-MIN-8e-fixed.gguf` showed:

- exps only: non-empty baseline output;
- exps + all shexp: empty content;
- exps + only `ffn_gate_shexp`: empty content;
- exps + only `ffn_up_shexp`: empty content;
- exps + only `ffn_down_shexp`: empty content.

The original `M=1` tile-floor hypothesis was incomplete. Padding the single
path to `M_run = ceil(M, 8)` made synthetic single-GEMM tests pass at DS4
shared-expert dimensions, but the mini model still emitted empty content when
any shared-expert projection used TurboMind.

An f32-output single-GEMM experiment also passed synthetic tests, but did not
restore model coherence. That path was killed rather than retained.

Deeper root-cause probes on 2026-05-17 narrowed the failure:

- Actual GGUF bytes for `blk.0.ffn_{gate,up,down}_shexp.weight` pack and run
  correctly in isolation. The raw tensors have normal F8 scale ranges
  (`e_min=115`, `e_max=116`, no zero/255 scales, no F8 NaN payloads).
- The single wrapper had a real shape bug: it used `src1->ne[1]` as `M`, while
  ggml's normal matmul treats all rows after `K` as batched rows. The wrapper
  now uses `ggml_nrows(src1)`. This fixes the wrapper contract, but does not
  restore model coherence by itself.
- A direct ggml A/B comparison using the same real shared-expert weights shows
  TurboMind single GEMM differs from llama.cpp's native CUDA F8 path by about
  0.5% relative error on random activations:
  - gate, `M=1,N=2048,K=4096,scale=1`: `rel=0.005399`, `p99=0.015585`;
  - down, `M=1,N=4096,K=2048,scale=1`: `rel=0.005089`, `p99=0.010785`.

Conclusion: this is not primarily a GGUF byte-layout or scale-decode bug.
The failing path has a different numerical contract: TurboMind consumes FP16
activations and returns FP16 before conversion to FP32, while llama.cpp's native
F8 CUDA path uses its quantized-activation kernels. That drift is tolerable in
the routed expert path, where top-k routing/scaling dampens it, but not in the
always-on shared expert path.

## Implemented Fix

The llama.cpp fork now treats TurboMind as a packed stacked-expert backend by
default. Single dense tensors are left as an explicit opt-in experiment.

Implementation:

- `CUDA_TURBOMIND` `set_tensor` packs only tensors with `ne[2] > 1` by
  default.
- Non-MoE tensors (`ne[2] <= 1`) are uploaded in normal GGUF layout even if
  `-ot` places them on a `CUDA_TURBOMIND<N>` buffer.
- `GGML_TM_ENABLE_SINGLE=1` re-enables single-tensor packing for controlled
  experiments. Current `shexp` opt-in still reproduces empty content.
- CUDA matmul dispatch enters the single TurboMind path only when packed
  TurboMind metadata exists on the original tensor.
- Packed TurboMind tensors can no longer silently fall through to native CUDA,
  because native CUDA would read packed bytes as GGUF bytes.
- The single path now flattens `M = ggml_nrows(src1)` and pads `M` to an 8-row
  floor before calling sm70 HMMA.884.

This makes an over-broad `-ot` regex operationally safe: `shexp` and similar
single tensors may sit in a `CUDA_TURBOMIND` buffer, but they run through the
normal CUDA quantized kernels instead of TurboMind's single packed path.

Validation:

- TurboMind synthetic correctness: pass for F8/MXFP4 at `M=8`, plus DS4
  shared-expert shapes at `M=1` and `M=4`.
- Raw real-weight correctness: pass for
  `blk.0.ffn_gate_shexp.weight` from `DSv4-Flash-MIN-8e-fixed.gguf`.
- Mini model broad regex `exps|shexp`: coherent non-empty output, 20.05 tok/s.
- Mini model broad regex with `GGML_TM_ENABLE_SINGLE=1`: empty content,
  21.08 tok/s. This is the intentional opt-in reproduction.
- Full `DSv4-Flash-256e-fixed.gguf` broad regex `exps|shexp`: coherent
  Fibonacci completion, 12.32 tok/s for the 64-token decode probe.

Performance interpretation:

- This fix restores correctness and removes the footgun.
- It does not provide a shared-expert speedup; `shexp` currently falls back to
  default CUDA.
- Operational performance is therefore still essentially the exps-only
  TurboMind envelope, not the hoped-for P9 +10-20%.

## Spike A: Fix TurboMind Single-Tensor Decode Shapes

Goal: make `blk.N.ffn_{gate,up,down}_shexp.weight` safe when it is accidentally
matched by `-ot`.

Steps:

1. Add single-path correctness tests for F8/MXFP4 at `M=1` and shared-expert
   dimensions:
   - gate/up: `M=1, K=4096, N=2048`
   - down: `M=1, K=2048, N=4096`
2. Pad single-path `M` to the sm70 tile floor before calling
   `ggml_turbomind_mul_mat`.
3. Re-run TurboMind correctness tests in the V100 pod.
4. Launch full 8-GPU server with `exps|shexp`.
5. Gate on coherence first, then speed.

Result:

- Synthetic single-path tests pass after padding.
- Real shared-expert tensors pass isolated pack/matmul correctness.
- The single wrapper's batched-row shape bug is fixed.
- Full single-path shared-expert routing remains incoherent because its
  numerical path does not match llama.cpp's native F8 CUDA path closely enough.
- The accepted fix is a default guard/fallback, plus an opt-in experiment flag,
  not shared-expert TurboMind routing.

## Spike B: Import DS4 Fusion Boundaries Into llama.cpp

Only after Spike A is either fixed or killed.

Candidates, in priority order:

1. shared expert path: gate/up/SwiGLU/down plus routed add;
2. HC split + weighted sum + RMS norm;
3. shared-down + HC expansion;
4. q/kv RMS row handling;
5. KV finalizer/raw-cache store.

Rule: every fused path needs a bit-compare or token-level A/B gate before it
touches the server path.

## Spike C: DS4 V100 Fork Feasibility

Run only as a bounded option-buying experiment.

Required proof points:

- load or convert the current FP4/FP8/MXFP4 target format;
- create a minimal 8-GPU layer-sharded execution skeleton;
- reuse TurboMind or equivalent sm70 kernels;
- beat or materially simplify the llama.cpp path.

Kill after one week if any of those proof points remain vague. DS4 is
strategically interesting, but it should not displace the working model/shard
infrastructure until it earns that role.

## Near-Term Recommendation

Treat Spike A as closed for now. The operational configuration should keep
TurboMind acceleration on stacked routed experts only. A broad `exps|shexp`
regex is now safe, but it should not be expected to speed up shared experts.

The next performance work should avoid the sunk-cost trap around the current
single-tensor TurboMind path. Either build DS4-inspired fusions inside
llama.cpp with bit-compare gates, or run a bounded DS4 fork spike whose first
proof point is loading/converting the actual FP4/FP8/MXFP4 target model.

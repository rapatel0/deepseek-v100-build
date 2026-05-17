---
title: SPRINT-025 DS4 Runtime Strategy Evaluation
date: 2026-05-16
status: investigation report
scope: research/ds4 viability for V100 DeepSeek V4 Flash inference
---

# DS4 Runtime Strategy Evaluation

## Executive Summary

`research/ds4` is a real DeepSeek-V4-Flash-only inference runtime, not a thin
wrapper around llama.cpp. It already takes the narrow-runtime bet the user is
considering: fixed DeepSeek constants, a DS4-specific GGUF layout, persistent
GPU tensors, compressed KV handling, tool/API rendering, and many fused CUDA /
Metal kernels.

Verdict: **viable as a medium-term research branch, not viable as an immediate
replacement for the current 8x V100 llama.cpp/TurboMind path.**

The two blocking gaps are:

1. **Model format mismatch.** DS4 only accepts its own q2/q4 GGUFs. It does not
   load the current `/models/DSv4-Flash-256e-fixed.gguf` path used by the V100
   llama.cpp experiments, which is based on DeepSeek's FP4/FP8-style weights.
2. **CUDA is single-device today.** The DS4 CUDA backend initializes device 0,
   keeps global cuBLAS/model-cache state, and has no layer sharding, peer copy,
   NCCL, or per-device graph state. A q2 DS4 GGUF is about 81 GB and q4 is about
   153 GB, so neither can be a single-GPU resident model on a 32 GB V100.

The right short-term move is to continue the llama.cpp/TurboMind recovery and
use DS4 as a reference for graph boundaries, fusion ideas, and long-context KV
design. The right medium-term spike is a DS4 layer-sharded CUDA fork, because
DS4's fixed graph may make layer sharding much easier than preserving llama.cpp
generality.

## What DS4 Is

DS4 is intentionally model-specific. Its README says it is "not a generic GGUF
runner" and only targets DeepSeek V4 Flash, with Metal as the primary backend
and CUDA as a supported Linux backend. The loader is built around fixed model
constants: 43 layers, 4096 embedding dim, 256 experts, top-6 routing, 2048
routed FFN width, 128-token raw SWA, ratio-4 / ratio-128 compressed KV layers,
and mHC state.

Important code references:

- `research/ds4/README.md:1` through `research/ds4/README.md:15`: narrow DS4
  runtime statement and backend targets.
- `research/ds4/README.md:89` through `research/ds4/README.md:123`: supported
  GGUF family and q2/q4 download path.
- `research/ds4/ds4.c:86` through `research/ds4/ds4.c:109`: fixed DS4 model
  constants.
- `research/ds4/ds4.c:409` through `research/ds4/ds4.c:415`: layer-dependent
  compression schedule.
- `research/ds4/ds4_gpu.h:11` through `research/ds4/ds4_gpu.h:15`: tensor-
  resident GPU API, with activations/KV/scratch staying device-owned across the
  graph command sequence.

This confirms the premise: avoiding llama.cpp generality does expose a simpler
optimization surface.

## Model Compatibility

DS4 does not accept arbitrary GGUFs. It validates tensor names, dimensions, and
types against one fixed layout. The runtime expects:

- token embeddings in F16;
- dense attention/shared/output paths mostly Q8_0/F16/F32;
- routed gate/up experts as `IQ2_XXS` or `Q4_K`;
- routed down experts as `Q2_K` or `Q4_K`.

The supported GGUF type table in `ds4.c` contains normal GGML quants such as
F16, Q8_0, Q2_K, Q4_K, and IQ2_XXS, but no MXFP4 or F8_E4M3_B128 path. The
CUDA MoE launcher enforces the same split: q2 means gate/up `IQ2_XXS` and down
`Q2_K`; q4 means all routed experts `Q4_K`.

References:

- `research/ds4/ds4.c:856` through `research/ds4/ds4.c:895`: supported tensor
  type IDs.
- `research/ds4/ds4.c:2223` through `research/ds4/ds4.c:2257`: routed expert
  type checks.
- `research/ds4/ds4.c:2283` through `research/ds4/ds4.c:2354`: full layout
  validation.
- `research/ds4/ds4_cuda.cu:9813` through `research/ds4/ds4_cuda.cu:9815`: CUDA
  routed-MoE type gate.
- `research/ds4/download_model.sh:33` through `research/ds4/download_model.sh:39`:
  q2-imatrix is about 81 GB; q4-imatrix is about 153 GB.
- `research/ds4/gguf-tools/README.md:59` through `research/ds4/gguf-tools/README.md:67`:
  full q2/q4 GGUF generation expectations.

Implication: DS4 cannot be tested by just pointing it at the current
`DSv4-Flash-256e-fixed.gguf`. Either use the published antirez q2/q4 GGUFs, or
port DeepSeek FP4/FP8/MXFP4 support into DS4.

## CUDA / V100 Fit

The CUDA backend is single-device. `ds4_gpu_init()` hard-codes `dev = 0`, creates
one global cuBLAS handle, and the source search shows no `cudaGetDeviceCount`,
peer access, NCCL, device map, layer-to-device assignment, or peer copies.

References:

- `research/ds4/ds4_cuda.cu:1205` through `research/ds4/ds4_cuda.cu:1222`:
  CUDA init selects device 0 and creates a single cuBLAS handle.
- `research/ds4/ds4_cuda.cu:1448` through `research/ds4/ds4_cuda.cu:1491`:
  model copy or host registration for the single CUDA context.
- `research/ds4/ds4_cuda.cu:1327` through `research/ds4/ds4_cuda.cu:1351`:
  managed-memory policy is for oversized KV/context buffers, not multi-GPU
  model sharding.

A 32 GB V100 cannot hold the 81 GB q2 GGUF or 153 GB q4 GGUF on one GPU. DS4 can
register the mmap for device access and optionally cache weight ranges, but that
is not a credible performance plan for this workload: decode would stream huge
expert weight ranges over host memory / PCIe instead of keeping the active layer
weights resident on the owning GPU.

V100-specific risks:

- V100 has no native FP8/TF32 acceleration. Current llama.cpp/TurboMind work is
  specifically exploiting an sm70 TurboMind path for the model's FP4/FP8-style
  expert weights.
- DS4's q2 path uses custom IQ2_XXS/Q2_K kernels. It may be memory efficient,
  but it is not automatically faster than the current TurboMind expert path.
- The published speed table reports Metal and DGX Spark/GB10 numbers, not V100
  numbers.

## Fusion And Optimization Surface

DS4 already has many of the fusions we would hope to get from a model-specific
runtime:

- fused q/kv RMS normalization rows;
- fused FP8 KV round-trip plus raw-cache store for decode;
- fused HC split + weighted sum + RMS norm;
- fused Q8 matmul + HC expand;
- fused shared gate/up SwiGLU;
- fused shared-down Q8 matmul + routed add + HC expand;
- routed MoE decode and batch paths that understand exactly 256 experts and
  top-6 routing.

References:

- `research/ds4/ds4_gpu.h:226` through `research/ds4/ds4_gpu.h:238`: q/kv RMS
  row fusion API.
- `research/ds4/ds4_gpu.h:269` through `research/ds4/ds4_gpu.h:282`: decode KV
  finalizer fusion.
- `research/ds4/ds4_gpu.h:775` through `research/ds4/ds4_gpu.h:788`: shared down
  plus HC expansion fusion API.
- `research/ds4/ds4.c:9124` through `research/ds4/ds4.c:9868`: decode layer
  graph calling the fused kernels.
- `research/ds4/ds4.c:12363` through `research/ds4/ds4.c:12520`: batched prefill
  FFN path and routed MoE batch call.
- `research/ds4/ds4_cuda.cu:10443` through `research/ds4/ds4_cuda.cu:10493`:
  fused HC split + norm implementation.
- `research/ds4/ds4_cuda.cu:10573` through `research/ds4/ds4_cuda.cu:10603`:
  fused shared-down + HC expansion implementation.

This is good news and bad news. Good: the codebase has the right shape for DS4-
specific optimization. Bad: a lot of low-hanging fusion is already present, so
switching runtimes alone is unlikely to unlock a large immediate speedup. The
remaining gains probably require either:

- multi-GPU layer sharding plus per-device weight caching;
- porting the current TurboMind MXFP4/F8 expert path into DS4;
- deeper layer-level fusion that reduces inter-kernel round trips across the
  HC, attention, router, and MoE boundary.

## Serving / Agent Features

DS4 includes a serious local server: OpenAI chat, OpenAI Responses, Anthropic
messages, SSE streaming, DSML tool call handling, exact tool-call replay, and
disk KV snapshots. This is valuable if we want a self-contained DeepSeek agent
runtime.

The server is intentionally serialized through one graph worker. That is simpler
and may avoid some llama.cpp slot/KV complexity, but it also means no request
batching or true concurrent inference today.

References:

- `research/ds4/README.md:288` through `research/ds4/README.md:306`: server
  model and single graph worker.
- `research/ds4/ds4_server.c:4` through `research/ds4/ds4_server.c:11`: worker
  ownership of session/KV state.

## Why DS4 Is Not The Immediate Replacement

The current V100 operational path is:

- 8x V100-SXM2-32GB on `gpu-01`;
- llama.cpp with `-sm layer`;
- current `DSv4-Flash-256e-fixed.gguf`;
- TurboMind expert tensors distributed across devices.

DS4 today is:

- one CUDA device;
- its own q2/q4 GGUF family;
- no arbitrary DeepSeek GGUF loading;
- no 8-GPU layer split;
- no V100 performance evidence.

Given those facts, a DS4 switch would replace a nearly operational path with a
new porting project. It may still be the cleaner long-term runtime, but it is
not the fastest route to "get this operational" on the current cluster.

## Recommended Strategy

### Keep llama.cpp/TurboMind As The Production Track

Use the existing SPRINT-025 path to finish the 8x V100 server. The local sprint
reports currently say the TurboMind MXFP4 nibble-lane bug and the DeepSeek4
slot/KV reset bug were fixed in the llama.cpp path. That path already has:

- model compatibility;
- 8-GPU layer sharding;
- working V100 deployment machinery;
- DSv4-shape TurboMind tests;
- full-model coherent decode evidence.

DS4 should not interrupt that track.

### Use DS4 As A Reference Immediately

Cannibalize DS4 ideas into llama.cpp where they are low-risk:

- q/kv RMS row fusion;
- KV finalizer fusion;
- HC split + weighted sum + norm fusion;
- shared-down + routed add + HC expansion fusion;
- routed MoE sorted-pair/direct-sum ideas;
- serialized exact-KV server semantics for slot/prefix bugs.

This gives practical value without needing a full runtime switch.

### Run A Bounded DS4 CUDA Spike Later

Suggested kill-gated spike:

1. Build DS4 on a CUDA host with `make cuda CUDA_ARCH=sm_70`.
2. Try loading the current `DSv4-Flash-256e-fixed.gguf` only to capture the exact
   type/layout failure. Expect failure.
3. Download or stage the published q2-imatrix GGUF. Do not expect it to fit
   single-GPU resident; run a tiny correctness smoke only if host-mapped access
   is tolerable.
4. If the runtime is still attractive, implement minimal 8-GPU layer sharding:
   per-device CUDA state, per-layer device ownership, per-device weight-range
   caches, per-layer KV allocations on the owning device, and peer/host copies
   for the ~64 KB HC state at layer boundaries.
5. Compare q2 DS4 sharded generation against the current llama.cpp/TurboMind
   baseline. If q2 quality or speed is not acceptable, stop unless we are ready
   to port MXFP4/F8 support.

Suggested stop-loss:

- Stop after one week if DS4 cannot produce coherent q2 output on V100 with
  layer sharding.
- Stop immediately if it cannot exceed or materially simplify the llama.cpp
  operational path.

## What A Serious DS4 V100 Port Requires

Minimal layer-sharded CUDA design:

- Replace global CUDA state with per-device state:
  - cuBLAS handle per GPU;
  - model range cache per GPU;
  - quality/math mode per GPU;
  - current-device selection wrappers.
- Add `layer_device[43]` or a contiguous layer partition.
- Allocate each layer's raw/comp KV caches on the layer's owning GPU.
- Cache only that layer's weight ranges on that GPU.
- Copy HC state between GPUs at layer boundaries. This is small relative to
  weights: `DS4_N_HC * DS4_N_EMBD * sizeof(float)` is 4 * 4096 * 4 = 64 KiB.
- Decide where embeddings and LM head live. A simple first version can put
  embeddings on layer-0's GPU and output on the final GPU, then optimize later.
- Add CLI/env controls and a memory report that shows per-device weight/KV/
  scratch usage.
- Reuse DS4's existing official-vector and long-context tests for correctness.

This is plausible because DS4's graph is fixed and layer boundaries are clear.
It is not a small patch because the CUDA backend currently assumes one global
device.

## Bottom Line

DS4 is strategically interesting. It validates the idea that a DeepSeek-only
runtime can have much tighter graph boundaries and more targeted fusion than
llama.cpp. But in its current state it is not the operational answer for the
8x V100 cluster: it cannot load the current model, cannot shard across GPUs, and
has no V100 performance track record.

Recommendation: **do not pivot away from llama.cpp/TurboMind for SPRINT-025.**
Treat DS4 as a source of fusion patterns now, and schedule a separate DS4 CUDA
layer-sharding spike only if we want to own a DeepSeek-specific runtime after
the current server is stable.

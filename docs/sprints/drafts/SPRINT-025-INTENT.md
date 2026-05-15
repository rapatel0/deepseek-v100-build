# SPRINT-025 — Intent

## Seed prompt

Land the full DSv4-Flash-256e (156 GiB GGUF) on a multi-V100 k8s pod with tensor parallel sharding. Existing llama.cpp has the TP plumbing (`LLAMA_SPLIT_MODE_*`, `tensor_split[]`, `ggml_backend_cuda_split_buffer_type`, conditional `GGML_USE_NCCL` allreduce); SPRINT-023's CUDA_TURBOMIND buffer type is already per-device. The sprint provisions hardware, builds with NCCL, verifies the 256e model loads + decodes coherently, and measures end-to-end TPS.

Per SPRINT-024-DEFERRED #2 — the only previous gap was hardware. Hardware survey shows the on-cluster node `gpu-01` has **8× V100-SXM2-32GB = 256 GiB VRAM**, comfortably fits 156 GiB model + KV cache + activations.

## Orientation summary

1. **Hardware**: `gpu-01` reports `nvidia.com/gpu.count: 8`, all Tesla V100-SXM2-32GB. Same node as the existing `llamacpp-build` pod (which currently requests 1 GPU). SPRINT-025 P0 provisions a 4-GPU and/or 8-GPU pod variant.
2. **Existing TP surface in upstream llama.cpp**:
   - `LLAMA_SPLIT_MODE_NONE / LAYER / ROW / TENSOR` enum
   - `tensor_split[128]` float array (per-GPU fractions)
   - `ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split)` — already exists for row-split
   - `ggml_backend_cuda_allreduce_tensor` — declared in `ggml/include/ggml-cuda.h:31`; implementation in `ggml/src/ggml-cuda/ggml-cuda.cu:1249-1297` is gated on `#ifdef GGML_USE_NCCL` and calls `ncclAllReduce`
   - `ncclCommInitAll` at line 466 (conditional)
3. **SPRINT-023 CUDA_TURBOMIND buft is per-device**: `ggml_backend_cuda_turbomind_buffer_type(int device)` returns `CUDA_TURBOMIND<i>` per device. The buft singletons + extra-bufts proc-address mechanism already enumerates one buft per discoverable CUDA device. No structural change needed to make TURBOMIND work on each GPU.
4. **SPRINT-024 status**: planning bundle landed (commit `97f6c3f0d`), execution not started. SPRINT-025 does NOT block on SPRINT-024 shipping — multi-GPU layer-split TP works regardless of grouped vs per-expert dispatch (both pre-SPRINT-024 and post-SPRINT-024 paths run on the per-device CUDA_TURBOMIND buft). The sprint design accommodates either path.
5. **No VISION.md**; ledger script absent; SPRINT-024-DEFERRED.md is the authoritative deferred queue.

## Relevant codebase areas

- `ggml/src/ggml-cuda/ggml-cuda.cu` lines 461-470 (NCCL init), 991-1310 (split buffer + allreduce), 5544-5562 (proc_address)
- `ggml/include/ggml-cuda.h` (allreduce / split_buffer declarations)
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}` (already per-device; verify per-buft init is GPU-local)
- `ggml/src/ggml-cuda/CMakeLists.txt` — already has `GGML_CUDA_NCCL` option (line 184); just needs `find_package(NCCL)`
- `common/arg.cpp` — `-sm`, `-ts`, `-mg` flags
- `common/common.{cpp,h}` — `tensor_split[]`, `split_mode`
- `src/llama-model.cpp` — `make_cpu_buft_list` and per-device buft enumeration

## Constraints

- **V100 sm70**: no Hopper or sm80+ shortcuts. Stick with the SPRINT-023 sm70 packed kernels.
- **Single node, no inter-node networking**: 8 GPUs on `gpu-01`. PCIe + NVLink (V100 SXM2 has NVLink-2). No NCCL inter-node config to worry about.
- **No upstream PRs to ggml/llama.cpp** (per AGENTS.md). All work lands on `rapatel0/deepseek-v100-build`, branch `sprint-022-dsv4-integration`.
- **Commit after every phase, push to origin** (sticky from SPRINT-023).
- **156 GiB model fits but is tight**: with KV cache + workspace + activations, watch VRAM. Plan for `--cache-type-k q8_0 --cache-type-v q8_0` if FP16 KV overflows.
- **CUDA_TURBOMIND buft assumption**: SPRINT-023 set up the per-device buft correctly; SPRINT-024 (if/when it ships) extends it for grouped MoE. SPRINT-025 *doesn't* re-implement the buft — it verifies the existing one works across N GPUs.
- **Build environment**: CUDA 12.2, cmake 3.22, gcc 11.4, nvidia/cuda:12.2.2-devel-ubuntu22.04 image. NCCL needs to be installed in the build image.

## Active SPRINT-024 deferreds becoming relevant

| # | What | Relevance |
|---|---|---|
| 1 | Multi-slot decode / speculative | NOT this sprint — that's SPRINT-026+ |
| **2** | **Full DSv4-Flash-256e on multi-GPU** | **This sprint** |
| 3 | Hot-expert profile pipeline | NOT this sprint — only matters if 256e doesn't fit (it does) |
| 6 | PCIe expert streaming | Not needed — 256 GiB VRAM is enough |
| 11 | CUDA-graph capture for grouped path | Side question — verify it still works under TP |

## Success criteria

1. **Functional**: A 4-GPU and an 8-GPU pod load `DSv4-Flash-256e-fixed.gguf` with `-sm layer` and run inference end-to-end without crash.
2. **NCCL operational**: Build incorporates `GGML_USE_NCCL`; `ncclAllReduce` paths execute when TP is enabled and at least one tensor needs cross-GPU reduction.
3. **CUDA_TURBOMIND on all GPUs**: `-ot 'exps=CUDA_TURBOMIND[0-7]'` (or whatever -ot regex syntax allows) places expert weights on the TURBOMIND buft on each GPU. Verifiable via memory breakdown showing TURBOMIND buft allocation per GPU.
4. **Output coherence**: Greedy decode 32 tokens, `temp=0`, fixed prompt set of 10 prompts. Output should be coherent English (subjective — no requirement of matching a CPU baseline, since the 256e on CPU is not feasible).
5. **TPS measured**:
   - Bench `llama-bench -p 128 -n 32 -r 3` on the 4-GPU and 8-GPU configs.
   - Report decode TPS and prefill TPS as headline numbers.
   - No hard gate — this sprint is the first time the 256e runs at all on this stack. The number is the report.
6. **VRAM accounting**: Memory breakdown per GPU shows model fits with ≥ 10% headroom for KV+compute.

## Verification strategy

- **Build**: clean build with `-DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON`, NCCL libraries available; verify `ldd bin/llama-server | grep nccl` succeeds.
- **Smoke load**: `llama-server` with `-ngl 999 -sm layer -mg 0` on a 4-GPU pod first. If it loads, repeat on 8-GPU.
- **TURBOMIND smoke**: Same load with `-ot 'exps=CUDA_TURBOMIND0,exps=CUDA_TURBOMIND1,...'` (or a single regex if the parsing supports it).
- **Decode**: 32-token greedy completion via the HTTP API; eyeball for English fluency.
- **Bench**: `llama-bench -m /models/DSv4-Flash-256e-fixed.gguf -ngl 999 -sm layer -p 128 -n 32 -r 3` and `-r 3` variants with TURBOMIND `-ot`.
- **Layer-split per-GPU activity**: `nvidia-smi dmon -s p -d 1` during decode shows roughly even utilization across the 4 (or 8) GPUs.

## Uncertainty assessment

| Factor | Level | Why |
|---|---|---|
| Correctness | **Medium-High** | First time the 256e runs at all on V100. Layer-split path is well-trodden upstream but TURBOMIND × multi-GPU has not been exercised. Possible issues: per-device dlopen of libggml-turbomind.so, NCCL link availability, per-device CUDA_TURBOMIND state. |
| Scope | **Medium** | Bounded by "load + measure". No new dispatch design. But VRAM provisioning and pod construction can balloon if NCCL doesn't work cleanly. |
| Architecture | **Low-Medium** | Reuses existing TP scaffolding. Risk is the integration seam: TURBOMIND buft × split buft × NCCL allreduce. |

## Open questions (for the interview)

1. **Split mode** — `LLAMA_SPLIT_MODE_LAYER` (per-layer per-GPU, no allreduce needed) vs `LLAMA_SPLIT_MODE_ROW` (split rows of each linear, needs allreduce per layer)?
   - **Layer**: simplest, lowest comm overhead, but per-token latency = sum of per-GPU layer times (no pipelining built in).
   - **Row**: full TP, needs `ncclAllReduce` for output projections; lower per-token latency at higher comm cost on PCIe-without-full-NVLink.
2. **GPU count** — start with 4 V100 (128 GiB, fits with `q8_0` KV) or jump to 8 V100 (256 GiB, comfort)?
3. **Should SPRINT-024 ship first as a prerequisite, or run SPRINT-025 in parallel?** If SPRINT-025 lands first, per-GPU per-expert turbomind dispatch will be the regime; if SPRINT-024 first, per-GPU grouped MoE dispatch.
4. **Hard perf gate or no?** SPRINT-024 ended up with a soft "ship if no regression + lift". SPRINT-025 has no comparable baseline — this is the FIRST run of 256e on this stack. Suggest: no hard gate, capture numbers as REPORT-19.

## What this sprint is NOT

- New kernel design.
- Multi-slot decode / speculative (SPRINT-026+).
- Expert parallelism (256 experts sharded ACROSS GPUs as opposed to layers sharded across GPUs).
- Inter-node networking. Single-node only.
- A perplexity sweep on a real eval set. Coherence-only verification.

## Vision context

No `docs/sprints/VISION.md`. The thread of work from SPRINT-022 → 023 → 024 is: get DSv4-Flash operational → get the V100 path fast → consolidate the launch budget. SPRINT-025 is the orthogonal axis: scale the model size by adding GPUs, not just optimizing throughput at fixed model size.

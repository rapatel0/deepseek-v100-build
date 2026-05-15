# SPRINT-025 — Multi-GPU Tensor Parallel & DSv4-Flash-256e Landing (V100 sm70)

**Status:** DRAFT 2026-05-15
**Predecessor:** SPRINT-024 (Grouped MoE dispatch / single-GPU perf landing)
**Successor:** SPRINT-026 (Multi-slot / Speculative decoding)

---

## 1. Overview

SPRINT-025 scales the DeepSeek-V4-Flash deployment from a single V100 to a multi-GPU Tensor Parallel (TP) configuration. The primary goal is to land the full **DSv4-Flash-256e (156 GiB GGUF)** on the `gpu-01` node using 8× V100-SXM2-32GB (256 GiB total VRAM).

This sprint leverages the existing llama.cpp TP infrastructure (`LLAMA_SPLIT_MODE_*`, `tensor_split[]`, NCCL allreduce) and SPRINT-023's per-device `CUDA_TURBOMIND` buffer type. It provisions the K8s hardware environment, wires NCCL into the build, verifies multi-GPU expert dispatch, and establishes the first performance baseline for the 256e model on this stack.

---

## 2. Use Cases

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | K8s pod manifests (4-GPU/8-GPU) + NCCL-enabled build image available. |
| P1 | NCCL operational in `llama-cli`; Row-split TP verified on smaller models. |
| P2 | Multi-GPU `CUDA_TURBOMIND` verified; experts sharded across N GPUs. |
| P3 | DSv4-Flash-256e (156 GiB) loaded + coherent decode on 8-GPU pod. |
| P4 | VRAM optimization: Q8_0 KV cache integration for tight 4-GPU/8-GPU configs. |
| P5 | REPORT-19: First end-to-end TPS numbers for the full 256e model. |

---

## 3. Architecture

### 3.1 Multi-GPU K8s Provisioning
The `gpu-01` node supports up to 8 GPUs. SPRINT-025 provisions two pod variants:
- **8-GPU (Default)**: 256 GiB VRAM. Comfortably fits the 156 GiB model + FP16 KV cache + activations.
- **4-GPU (Testing)**: 128 GiB VRAM. Used for plumbing verification and "partial offload" or high-quantization tests (e.g., if a smaller 256e GGUF exists or for 16e/32e variants). Note: The 156 GiB 256e model requires 8 GPUs for full offload.

### 3.2 Tensor Parallelism (TP) Modes
The sprint supports two primary splitting modes:
1. **`LLAMA_SPLIT_MODE_LAYER`**: Layers are distributed across GPUs. Simple, no cross-GPU communication (NCCL) required for weights. Low overhead but unbalanced if layer sizes vary significantly.
2. **`LLAMA_SPLIT_MODE_ROW`**: Individual tensors (linear layers) are split across GPUs. Requires `ncclAllReduce` for output projections. This is the "True TP" mode targeted for maximum throughput on NVLink.

### 3.3 NCCL Integration
The build system is extended to find and link NCCL. `ggml-cuda` already contains implementation for `ggml_backend_cuda_allreduce_tensor` gated on `GGML_USE_NCCL`. This sprint verifies the initialization sequence:
- `ncclCommInitAll` called for the requested GPU set.
- Allreduce paths active for `LLAMA_SPLIT_MODE_ROW` or `TENSOR`.

### 3.4 Per-Device CUDA_TURBOMIND
SPRINT-023 implemented `ggml_backend_cuda_turbomind_buffer_type(int device)`. SPRINT-025 ensures:
- The `-ot` regex parser correctly routes experts to `CUDA_TURBOMIND<i>` where `i` is the GPU index.
- Expert packing/upload happens per-device.
- Grouped MoE (from SPRINT-024) functions correctly within the per-device context.

---

## 4. Implementation

### P0 — Provisioning & Build Infrastructure (2 days)

**Goal:** Establish the multi-GPU execution environment and NCCL-enabled build.

1. **P0.1 — Docker Image Update**
   - Update `llamacpp-build` Dockerfile to include `libnccl2` and `libnccl-dev`.
   - Verify `find_package(NCCL)` works in `ggml/src/ggml-cuda/CMakeLists.txt`.
2. **P0.2 — K8s Manifests**
   - Create `ds-v4-256e-8gpu.yaml` requesting 8 GPUs on `gpu-01`.
   - Create `ds-v4-256e-4gpu.yaml` requesting 4 GPUs.
   - Configure shared memory (`/dev/shm`) and model volume mounts.
3. **P0.3 — NCCL Build Verification**
   - Build with `-DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON`.
   - Verify `ldd bin/llama-cli | grep nccl` shows successful linkage.

**P0 Gate:** 8-GPU pod is `Running`; `llama-cli` links NCCL; `--help` shows `-sm` and `-mg` flags.

### P1 — Multi-GPU NCCL Plumbing & Row-Split (3 days)

**Goal:** Verify basic TP sharding and NCCL communication.

1. **P1.1 — Layer-Split Smoke Test**
   - Load a smaller model (e.g., DSv4-Flash-AVG-16e) on 2 or 4 GPUs with `-sm layer`.
   - Verify `nvidia-smi` shows memory and utilization across all requested GPUs.
2. **P1.2 — Row-Split Verification**
   - Load same model with `-sm row`.
   - Trace `ncclCommInitAll` and `ncclAllReduce` calls (using `NCCL_DEBUG=INFO`).
   - Verify output coherence vs single-GPU baseline.
3. **P1.3 — Tensor Split Array (`-ts`)**
   - Test uneven splits (e.g., 2:1 on 2 GPUs) to ensure `tensor_split[]` weights are honored.

**P1 Gate:** `LLAMA_SPLIT_MODE_ROW` produces coherent output on 4 GPUs with NCCL active.

### P2 — Multi-GPU CUDA_TURBOMIND Wiring (3 days)

**Goal:** Ensure MoE experts are correctly sharded and dispatched on multi-GPU.

1. **P2.1 — Per-Device Buft Registration**
   - Confirm `src/llama-model.cpp` correctly enumerates `CUDA_TURBOMIND<i>` for all available GPUs.
2. **P2.2 — Sharded Expert Routing**
   - Use `-ot 'exps.*=CUDA_TURBOMIND[0-3]'` (on 4-GPU) or `-ot 'exps.*=CUDA_TURBOMIND[0-7]'` (on 8-GPU).
   - Verify `set_tensor` (packing) triggers for each GPU.
3. **P2.3 — Dispatch Verification**
   - Verify MoE dispatch (single-expert or grouped) routes to the correct device-local Turbomind instance.
   - Smoke test: load 16e model, offload experts across 4 GPUs, verify no inter-GPU memory leaks or invalid device accesses.

**P2 Gate:** Experts distributed across 4+ GPUs on TURBOMIND buffers; coherent 32-token decode.

### P3 — DSv4-Flash-256e Landing (4 days)

**Goal:** Full 156 GiB model operational on 8 GPUs.

1. **P3.1 — Full Load (8-GPU)**
   - Launch 8-GPU pod. Load `DSv4-Flash-256e-fixed.gguf` (156 GiB).
   - Use `-ngl 999 -sm layer` (initial simple path).
   - Monitor VRAM: expect ~20 GiB per GPU for weights + ~2-5 GiB for overhead/KV.
2. **P3.2 — Output Coherence**
   - Greedy decode 32 tokens on fixed prompts.
   - Verify English fluency (subjective English baseline).
3. **P3.3 — Row-Split DSv4-256e**
   - Transition to `-sm row`.
   - Verify NCCL scaling: Does row-split reduce per-token latency vs layer-split on 8 GPUs?

**P3 Gate:** 156 GiB model loads on 8 GPUs; greedy decode produces coherent text.

### P4 — VRAM Optimization & q8_0 KV (2 days)

**Goal:** Secure headroom for long-context or tight configurations.

1. **P4.1 — Cache Type Integration**
   - Test `--cache-type-k q8_0 --cache-type-v q8_0`.
   - Measure VRAM delta vs default FP16 KV.
2. **P4.2 — 4-GPU Stress Test**
   - Attempt partial offload or highly-quantized load of 256e on 4 GPUs.
   - Document the "Maximum Offloadable Layers" for 128 GiB VRAM budget.

**P4 Gate:** Q8_0 KV operational; VRAM breakdown documented for 4-GPU and 8-GPU configs.

### P5 — Measurement & REPORT-19 (2 days)

**Goal:** Quantitative performance characterization.

1. **P5.1 — TPS Sweep**
   - `llama-bench -p 128 -n 32 -r 3` on 4-GPU (partial) and 8-GPU (full).
   - Compare `-sm layer` vs `-sm row`.
   - Compare TURBOMIND vs CPU experts (on multi-GPU).
2. **P5.2 — Scaling Efficiency**
   - Measure TPS on 2, 4, 6, 8 GPUs (using subset of model if needed) to calculate NCCL/TP scaling factor.
3. **P5.3 — REPORT-19**
   - Write close-out narrative.
   - Headline numbers: 256e Decode TPS and Prefill TPS.
   - VRAM accounting per GPU.

**P5 Gate:** REPORT-19 published; memory updated.

---

## 5. Files Summary

### Modified

| Path | Change |
|---|---|
| `ggml/src/ggml-cuda/CMakeLists.txt` | Add `find_package(NCCL)`, link NCCL if `GGML_CUDA_NCCL=ON`. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Verify `ncclCommInitAll` / `ggml_backend_cuda_allreduce_tensor` integration. |
| `src/llama-model.cpp` | Ensure multi-GPU `CUDA_TURBOMIND` registration. |
| `.devops/cuda.Dockerfile` | Add NCCL dependencies. |

### New

| Path | Purpose |
|---|---|
| `docs/sprints/drafts/ds-v4-256e-8gpu.yaml` | K8s manifest for 8-GPU deployment. |
| `docs/sprints/drafts/ds-v4-256e-4gpu.yaml` | K8s manifest for 4-GPU deployment. |
| `docs/sprints/REPORT-19.md` | Sprint results and TPS baseline. |

---

## 6. Definition of Done

1. ✅ NCCL-enabled build producing `llama-cli` linked against `libnccl.so`.
2. ✅ Functional 8-GPU K8s pod on `gpu-01`.
3. ✅ `LLAMA_SPLIT_MODE_ROW` verified working with NCCL allreduce.
4. ✅ DSv4-Flash-256e (156 GiB) loaded and running on 8 V100s.
5. ✅ `CUDA_TURBOMIND` experts sharded across all GPUs.
6. ✅ Coherent greedy decode (32 tokens) on 256e model.
7. ✅ Memory breakdown showing ≥ 10% VRAM headroom on 8-GPU config.
8. ✅ TPS measured and reported in REPORT-19.
9. ✅ No regression on single-GPU 16e/32e paths.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | 156 GiB model + KV + Workspace > 256 GiB VRAM | Low | High | P4 (Q8_0 KV) provides ~50% KV reduction; 100 GiB slack is plenty. |
| 2 | NCCL allreduce overhead destroys gain on PCIe/NVLink-2 | Medium | Medium | P1 measures Row vs Layer; fall back to Layer-split if Row is slower. |
| 3 | Per-device TURBOMIND init race conditions | Low | Medium | Sequential init or mutex protection in `ggml_turbomind_init`. |
| 4 | K8s pod scheduling delays or resource contention on `gpu-01` | Medium | Low | Use nodeSelector and explicit resource requests/limits. |
| 5 | Output divergence on multi-GPU vs single-GPU | Low | Medium | P1 correctness gate; verify allreduce precision. |

---

## 8. Security

- **NCCL Port Exposure**: NCCL uses TCP for coordination. Ensure K8s pod network is isolated to the node or cluster.
- **Shared Memory**: Multi-GPU communication via NVLink/PCIe is local to the node. No new network surface.
- **Container Privileges**: Pods may need `IPC_LOCK` for NCCL pinned memory. Audit manifest for minimum required capabilities.

---

## 9. Dependencies

1. **Hardware**: Node `gpu-01` with 8× V100.
2. **Model**: `DSv4-Flash-256e-fixed.gguf` (156 GiB).
3. **Software**: CUDA 12.2, NCCL 2.x, SPRINT-024 (Grouped MoE) results.

---

## 10. Open Questions

1. **NVLink Availability**: Does `gpu-01` have a full NVLink mesh? SXM2 usually does. If not, Row-split over PCIe will be bandwidth-constrained.
2. **Context Length**: What is the target context window for the 256e model? 8-GPU 256 GiB VRAM can support ~32k context in FP16, or ~64k in Q8_0.
3. **Layer vs Row Performance**: Will the MoE layers (very large) benefit more from Row-split (intra-layer parallelism) or Layer-split (simpler)? P5 will answer.

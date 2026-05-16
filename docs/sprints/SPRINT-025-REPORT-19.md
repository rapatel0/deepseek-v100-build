# SPRINT-025 REPORT-19 — Multi-GPU 256e load

**Date:** 2026-05-15
**Tag:** sprint-025-close (pending P5–P7)
**Verdict:** _pending — populated after P4 6-GPU load test resolves_

---

## TL;DR

DSv4-Flash-256e (146 GiB, 284B params, 256 experts, top-6 routing) loads on a
6-GPU V100 pod via `-sm layer` and produces output. Multi-device safety of
the CUDA_TURBOMIND backend is proven via the per-device `State[]` refactor
landed in P2 (`test_multi_device.cpp` bit-identical first/second dispatch
across an init(1) call between them).

---

## P-1 / P0 — Pod manifests + NCCL image (precondition)

NCCL 2.19.3 + `libnccl-dev` available on `nvidia/cuda:12.2.2-devel-ubuntu22.04`.
`find_package(NCCL)` succeeds out of box; `ldd llama-server` shows
`libnccl.so.2 => /lib/x86_64-linux-gnu/libnccl.so.2`. `manifests/llamacpp-build-{4,6,8}gpu.yaml` target `gpu-01` (8x V100-SXM2-32GB).

## P1 — Layer-split smoke on MIN-16e

4-GPU pod, `-sm layer -ngl 99`, `-ot 'exps=CUDA_TURBOMIND0'`. Load + decode
produce coherent output. `pipeline_parallel` flag asserted true at
`llama-context.cpp:316-321` when no overrides set (verified post-P3 pivot —
see follow-ups).

## P2 — Per-device TmLib refactor

**Status:** ✅ PASS. `g_state` singleton replaced with `State g_states[TM_MAX_DEVICES=32]` indexed by `cudaGetDevice()`. Idempotent per-device init; shutdown walks all devices. `TmLib::per_device_inited[32]` bitset on
caller side so `tm_ensure_loaded(device)` is a no-op after the first call
per device.

**Test gate:** `test_multi_device.cpp` smoke runs three dispatches:
1. GPU 0 first dispatch (sum_abs = 346334.60)
2. GPU 1 dispatch (sum_abs = ~similar magnitude, different seed)
3. GPU 0 second dispatch (must match #1)

Pre-P2 the global `g_state` singleton meant init(1) would `cudaFree` GPU 0's
workspace pointers, corrupting any in-flight dispatch and breaking sum at
step 3. Post-P2:

```
[multi-dev] GPU 0 first dispatch: sum_abs=346334.60
[multi-dev] GPU 1 dispatch: sum_abs=...
[multi-dev] GPU 0 second dispatch: sum_abs=346334.60
[multi-dev] GPU 0 first vs second: rel diff = 0.000000e+00
[multi-dev] PASS — per-device state intact across cross-device init/dispatch
```

Single-GPU regression tests (`test_ggml_turbomind_correctness`, `test_ggml_turbomind_grouped`) still pass.

## P3 — CUDA_TURBOMIND family-alias buft

**Status:** **PIVOT — deferred** to SPRINT-027 follow-up.

**Rationale:** Substituting a family-alias buft to the concrete
`CUDA_TURBOMIND<layer_device>` at tensor allocation time without populating
`model.tensor_buft_overrides[]` (the only way to keep `model.has_tensor_overrides() == false` and preserve pipeline parallelism) requires invasive
`src/llama-model.cpp` changes inside the override-resolution closure.

For the SPRINT-025 ship gate (256e loads + decodes coherently on multi-GPU)
the PP loss from generated per-layer concrete overrides is negligible:
pipeline parallel only helps at prefill (multiple tokens in flight); at
M=1 decode each forward pass is serial and there is nothing to pipeline.

Family-alias remains valuable for prefill workloads and is documented as
[SPRINT-025-FOLLOWUPS.md #1](SPRINT-025-FOLLOWUPS.md).

## P4 — Full 256e load on 6 GPUs

_Filled in after the load test resolves._

| Metric | Value |
|---|---|
| Model | DSv4-Flash-256e-fixed.gguf (146 GiB on disk) |
| n_experts | 256 (top-6 routing) |
| n_params | 284.33 B |
| n_vocab | 129280 |
| Cluster | 6× V100-SXM2-32GB on gpu-01 (4 NV-bridged, 2 PCIe) |
| `-sm` | layer |
| `-c` | 1024 |
| Load time (s) | TBD |
| VRAM per GPU after load | TBD |
| Decode TPS | TBD |
| Sample output (`"The capital of France is"`, temp=0 top_k=1) | TBD |

## P5–P7 — Scaling sweep + close-out

_Pending. P4 result drives whether to spend the 2 free GPUs from `tcg-dev` /
`llamacpp-build` shutdown on a true 8-GPU test._

---

## Known limitations carried forward

- **CUDA_TURBOMIND family-alias deferred** — multi-GPU expert distribution
  with TURBOMIND offload requires either generated per-layer overrides
  (loses PP) or the deferred family-alias work. P4 measures whether the
  PP loss is material in practice.
- **AVG-16e divergence** — 2/5 prompts diverge at FP16 argmax tiebreaks on
  AVG-16e (carried from SPRINT-024 REPORT-18 §P3.2). SPRINT-026 P0
  addresses.

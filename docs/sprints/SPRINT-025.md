# SPRINT-025 — Full DSv4-Flash-256e multi-GPU landing (V100 sm70)

**Status:** PLANNED 2026-05-15
**Predecessor:** SPRINT-024 (grouped MoE dispatch — planning bundle landed, execution deferred)
**Successor:** SPRINT-026 (multi-slot decode / speculative decoding, or row-TP follow-on if P6 punted)

---

## 1. Overview

SPRINT-023 + SPRINT-024 staged the V100 turbomind path on a single GPU; SPRINT-022 P5 measured 16.6 t/s decode on MIN-Ne fixtures. The next axis is **model size**: the real production target DSv4-Flash-256e is 156 GiB on disk and won't fit on one V100 32 GB. SPRINT-025 lands it on **8× V100-SXM2-32GB** on cluster node `gpu-01` with `LLAMA_SPLIT_MODE_LAYER` and per-device CUDA_TURBOMIND.

The sprint is **decision-complete on three load-bearing technical reality checks** the multi-agent planning surfaced:

1. **`LLAMA_SPLIT_MODE_ROW` is currently blocked for `deepseek4`** — `src/llama-model.cpp:770-771` throws unconditionally. Row-TP is P6 conditional with a 1-day kill criterion, not part of the ship path.
2. **`CUDA_TURBOMIND` is not multi-device-safe today** — `tm_ensure_loaded` in `ggml-cuda-turbomind.cu` and `g_state` in `ggml/vendor/turbomind/api.cc` are single-device singletons that tear down on device hop. **P2 refactors to per-device state with no C ABI change.**
3. **`-ot regex=BUFT` pins to one device-specific buft name**, not a layer's assigned device — `CUDA_TURBOMIND0` ≠ "the TURBOMIND buft for whichever GPU this layer lives on". **P3 introduces a `CUDA_TURBOMIND` family-alias buft** that resolves to the layer's device at tensor-allocation time, preserving pipeline parallelism (`model.has_tensor_overrides()` stays false).

Other architectural decisions per user interview:
- Per-device TM refactor → **Option A** (per-device state inside `libggml-turbomind.so`, no ABI change).
- Family-alias buft → **chosen over generated per-layer overrides** to keep pipeline parallel enabled.
- SPRINT-024 sequencing → **SPRINT-025 lands first**; multi-GPU on the per-expert dispatch path is the milestone. SPRINT-024 grouped path ships later as orthogonal perf work.

### Perf framing

This is the **first end-to-end run of DSv4-Flash-256e on this stack**. No baseline → no hard perf gate. Ships if 256e loads, decodes coherently on 8 GPUs, REPORT-19 captures TPS + VRAM + scaling sweep numbers.

---

## 2. Use Cases

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | 4-GPU + 8-GPU pod manifests; NCCL libs in build image; topology captured. |
| P1 | NCCL-enabled build; smoke harness proves allreduce path is live. |
| P2 | Per-device CUDA_TURBOMIND lifecycle; multi-GPU pack + dispatch validated. |
| P3 | CUDA_TURBOMIND family alias; 4-GPU layer-split MIN-16e smoke loads + decodes. |
| P4 | 8-GPU 256e load + 32-token greedy decode; q8_0 vs FP16 KV decision. |
| P5 | REPORT-19 with TPS + scaling sweep + memory + reproducible command line. |
| P6 | Row-TP feasibility verdict (measured or precise defer blocker). |
| P7 | Tag, followups for SPRINT-026. |

---

## 3. Architecture

### 3.1 Pod topology

`gpu-01` reports `nvidia.com/gpu.count: 8` (all Tesla V100-SXM2-32GB, NVLink-2 mesh, single node).

- **4-GPU pod** (`llamacpp-build-4gpu`) — bring-up harness only. **Not a full-model target** (128 GiB < 156 GiB before KV/scratch).
- **8-GPU pod** (`llamacpp-build-8gpu`) — 256e ship target. 256 GiB gross VRAM.

Both pods request GPUs through `resources.limits.nvidia.com/gpu`, mount the same `workspace` emptyDir + read-only `/models` PVC, and pin `nodeSelector: kubernetes.io/hostname: gpu-01`.

**Visibility invariant** (P0 gate): `nvidia-smi -L` count, `cudaGetDeviceCount`, and `ggml_cuda_init()` device count must agree with the pod's GPU request. NCCL's `ncclCommInitAll` enumerates all visible devices (`ggml-cuda.cu:461-467`); pod GPU masking must come from k8s `CUDA_VISIBLE_DEVICES`, not from runtime flags.

### 3.2 Split mode: LAYER ships, ROW is P6 conditional

**Ship path: `LLAMA_SPLIT_MODE_LAYER`.**

- Already implemented for deepseek4 in upstream llama.cpp.
- No allreduce needed (each layer runs on one GPU; activations flow GPU→GPU only between layers).
- Pipeline-parallel-eligible IF tensor overrides are absent (`model.has_tensor_overrides()` flag in `src/llama-context.cpp:316-321` gates `cparams.pipeline_parallel`).

**`LLAMA_SPLIT_MODE_ROW` is P6 conditional only.** `src/llama-model.cpp:770-771` currently throws `LLAMA_SPLIT_MODE_ROW not implemented for architecture 'deepseek4'`. Lifting the guard is not free — DSv4's MoE plumbing needs the buffer-type system to know how to split expert tensors across GPUs, which the row split-buffer doesn't natively do for the CUDA_TURBOMIND buft. P6 investigates with an explicit 1-day kill criterion.

**`LLAMA_SPLIT_MODE_TENSOR` is out of scope.** It's separate plumbing (requires FA, rejects quantized KV; `llama.cpp:945-990`, `llama-context.cpp:2957-2968`); not a synonym for LAYER.

### 3.3 Per-device TmLib refactor (P2 — Option A)

**Current bug.** In `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu:46-101`, `tm_ensure_loaded(device)` holds a single global `TmLib t` with one `init_device` field. The "different device" branch calls `t.shutdown()` then `t.init(device)`. Concurrently in `ggml/vendor/turbomind/api.cc:121-167`, `ggml_turbomind_init(int cuda_device)` keeps one `g_state` and frees `d_barriers / d_partials / d_flags` on device change. Any cross-device dispatch in one process therefore goes through a teardown / re-init cycle, freeing the previous device's workspace pointers.

**P2 fix (Option A).** Per-device state inside `libggml-turbomind.so`, no C ABI change.

```cpp
// ggml/vendor/turbomind/api.cc
namespace {
constexpr int TM_MAX_DEVICES = 32;
struct State { /* same fields as before */ };
State g_states[TM_MAX_DEVICES];   // keyed by cuda_device
}

extern "C" GGML_TM_EXPORT int ggml_turbomind_init(int cuda_device) {
    GGML_TM_CHECK(cuda_device >= 0 && cuda_device < TM_MAX_DEVICES);
    auto & s = g_states[cuda_device];
    std::lock_guard<std::mutex> lk(s.mtx);
    if (s.initialized) return 0;  // idempotent
    // … allocate barriers/partials/flags etc. for this device only.
    s.initialized = true;
    s.device      = cuda_device;
    return 0;
}
```

Same change to `ggml_turbomind_shutdown`, `_packed_bytes`, `_pack_weight_expert`, `_mul_mat`, `_mul_mat_grouped` — all dispatch by `cuda_device` through `g_states[device]`. No global "current device" assumption on the hot path.

Caller side (`ggml-cuda-turbomind.cu`): replace `TmLib::init_device` + `shutdown / init` re-init with `if (!t.per_device_inited[device]) { t.init(device); t.per_device_inited[device] = true; }`.

`dlopen` of `libggml-turbomind.so` stays once-per-process. Per-device state lives behind the single handle.

### 3.4 CUDA_TURBOMIND family-alias buft (P3)

**Problem.** `-ot 'exps=CUDA_TURBOMIND0'` pins matching tensors to GPU 0. In layer-split mode, layers 1-4 might live on GPUs 0-1 and layers 5-43 on GPUs 2-7; we want expert weights to live on the SAME GPU as their layer. The exact-name `-ot` mechanism can't express this.

**Two solutions, the cleaner one chosen per interview.**

- Rejected: generate per-layer overrides at model load (`blk.0.ffn_*_exps.weight=CUDA_TURBOMIND0`, `blk.1.ffn_*_exps.weight=CUDA_TURBOMIND0`, …). Cost: `model.has_tensor_overrides() = true` disables pipeline parallel.
- **Selected: device-local family alias.** A new sentinel buft `CUDA_TURBOMIND` (no device suffix) that's NOT one of the per-device singletons. When a tensor is assigned to this sentinel during model load, the loader resolves it to `CUDA_TURBOMIND<i>` for the layer's actual device — without populating `model.tensor_buft_overrides`, so `has_tensor_overrides()` stays false and pipeline parallel stays on.

**Implementation surface.**

- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` — add a new singleton `ggml_backend_cuda_turbomind_buffer_type_family()` that returns a buft with name `CUDA_TURBOMIND` (no suffix). The buft's `alloc_buffer` is a NO-OP / error; it's not directly allocatable.
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` — export the new function.
- Extras list: extend `ggml_backend_cuda_turbomind_get_extra_bufts` to include the family buft.
- `src/llama-model.cpp` — in the tensor-placement path, detect family-buft assignments and substitute with the concrete `CUDA_TURBOMIND<layer_device>`. Hook into the existing `tensor_buft_overrides` resolution but apply the substitution before recording.
- `common/arg.cpp` `parse_tensor_buffer_overrides` (already enumerates extras since SPRINT-023 P3) — no change.

The user invocation becomes: `-ot 'exps=CUDA_TURBOMIND'` (no suffix). This matches every expert tensor and binds each to its layer's GPU automatically.

### 3.5 NCCL build wiring

Already wired in upstream llama.cpp:
- `ggml/src/ggml-cuda/CMakeLists.txt:184-190` — `find_package(NCCL)` + conditional `GGML_USE_NCCL` define + `NCCL::NCCL` link.
- `ggml/cmake/FindNCCL.cmake` — present.
- `ggml/src/ggml-cuda/ggml-cuda.cu:461-467` — `ncclCommInitAll` for all visible devices.
- `ggml-cuda.cu:1249-1297` — `ggml_backend_cuda_allreduce_tensor` calls `ncclAllReduce` under `GGML_USE_NCCL`.

**Real P1 work** (not new CMake authoring):
1. Add `libnccl2 + libnccl-dev` (matching CUDA 12.2 / NCCL 2.18+) to the build image used by `llamacpp-build`.
2. `rm -rf build/` and rebuild with `-DGGML_CUDA_NCCL=ON` from clean cache.
3. Verify: `ldd build/bin/llama-server | grep libnccl` returns hits.
4. Smoke harness: tiny binary that calls `ggml_backend_cuda_init()` + `ggml_backend_cuda_allreduce_tensor()` on 2 GPUs; grep `NCCL_DEBUG=INFO` log for `NCCL INFO Bootstrap`. `ldd` alone is insufficient (would miss `dlopen`-only paths, though we don't expect any).

### 3.6 VRAM accounting (per-GPU, not cluster slack)

156 GiB model + KV + scratch + workspace + allocator fragmentation must fit per GPU. The heaviest-shard GPU is the binding constraint.

| Line item | Per-GPU |
|---|---|
| Model weights (avg, 8-way layer split) | ~19.5 GiB |
| Heaviest-shard GPU (embedding + lm_head, plus layers) | ~22 GiB worst case |
| Allocator + fragmentation + ggml-cuda pool overhead | 2-3 GiB |
| Turbomind scratch (`d_barriers`, `d_partials` = 256 MiB) | 0.5-1 GiB |
| Decode workspace + transient activations | 1-2 GiB |
| **Subtotal before KV** | **24-28 GiB** |
| Available KV per GPU on a 32 GiB card | 4-8 GiB |

**KV budget at 4-8 GiB per GPU × 8 = 32-64 GiB cluster-wide.** Context length must respect this. q8_0 KV vs FP16 KV is a real ~2× factor (Q8_0 = 1 byte/elem + 1 fp16 scale per 32 elems ≈ 1.0625 bytes/value; FP16 = 2 bytes/value — call it 1.9× reduction). P4 measures both.

**Per-GPU thresholds (stop-the-line):**
- After model load, before decode: no non-main GPU above 28 GiB.
- During 32-token decode: no GPU above 30 GiB.
- If either exceeded → drop context length, switch to q8_0 KV, rebalance with `-ts` before claiming success.

### 3.7 Pipeline parallel preservation

The family-alias mechanism is specifically chosen to keep `model.has_tensor_overrides() == false` (the loader applies the substitution but doesn't populate `model.tensor_buft_overrides` with anything matching tensors). With pipeline parallel on, decode latency is `max-per-GPU-layer-time` plus inter-GPU sync, not the sum of all layer times.

Verification in P3/P5: log `pipeline_parallel` flag at startup; ensure it's `1` (8 in the 8-GPU pod).

---

## 4. Implementation

### P0 — Pod variants + NCCL build image + topology (1 day)

**Goal:** make hardware + image plan explicit, eliminate ambiguity.

1. **P0.1** — Author `llamacpp-build-4gpu.yaml` and `llamacpp-build-8gpu.yaml` k8s manifests (mirror existing `llamacpp-build` spec; change `nvidia.com/gpu` request, name).
2. **P0.2** — Build a new container image (or update existing) that includes `libnccl2 libnccl-dev` matching CUDA 12.2 (NCCL 2.18+). Push to the cluster registry.
3. **P0.3** — Stand up the 4-GPU pod; capture:
   - `nvidia-smi -L` (device count + UUIDs)
   - `nvidia-smi topo -m` (NVLink mesh)
   - `nvidia-smi --query-gpu=memory.total,memory.free --format=csv`
   - `dpkg -l | grep nccl`
4. **P0.4** — Repeat on 8-GPU pod.
5. **P0.5** — Document the rule: 4-GPU is harness only, 8-GPU is the 256e target.

**P0 Gate**:
- ✅ 4-GPU pod runs, sees 4 GPUs (no more, no less)
- ✅ 8-GPU pod runs, sees 8 GPUs
- ✅ NCCL libs present in both pods
- ✅ Topology captured

### P1 — NCCL-enabled build + link verification (1 day)

**Goal:** make NCCL real, not a CMake option that fell back.

1. **P1.1** — On the 4-GPU pod: `rm -rf build/` to avoid CMake cache poisoning.
2. **P1.2** — Build: `-DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON` plus existing sm70 / CUDA 12.2 settings.
3. **P1.3** — Verify CMake: `find_package(NCCL)` succeeded; `NCCL_FOUND=TRUE` in `CMakeCache.txt`.
4. **P1.4** — Verify link: `ldd build/bin/llama-server | grep libnccl` returns ≥ 1 hit.
5. **P1.5** — Smoke harness `tests/test-nccl-allreduce.cpp` (new): allocate one f32 tensor on each of 2 GPUs, call `ggml_backend_cuda_allreduce_tensor`, verify result is `2 × input`. Run with `NCCL_DEBUG=INFO` and grep stderr for `NCCL INFO Bootstrap`.
6. **P1.6** — Capture `NCCL_DEBUG=INFO` log as REPORT-19 artifact.

**P1 Gate**:
- ✅ `ldd` shows libnccl linkage
- ✅ Smoke harness completes; allreduce result correct
- ✅ `NCCL_DEBUG=INFO` log shows Bootstrap on the right device fleet

### P2 — Per-device CUDA_TURBOMIND lifecycle refactor (2 days)

**Goal:** remove the single-device singleton; no global "current device" on the hot path.

1. **P2.1** — `ggml/vendor/turbomind/api.cc`: replace `State g_state` with `State g_states[TM_MAX_DEVICES]`. Add `cuda_device` index argument resolution to every entry point. Per-device mutex.
2. **P2.2** — Make `ggml_turbomind_init(cuda_device)` idempotent per device (`if (s.initialized) return 0`).
3. **P2.3** — `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu`: replace `TmLib::init_device` + tear-down-on-hop with `per_device_inited[]` bitset; remove `t.shutdown()` from the device-change path.
4. **P2.4** — Unit smoke `tests/test-turbomind-multi-device.cpp` (new): pack a small FP8 tensor on GPU 0, pack another on GPU 1 (both in the same process), dispatch both via `ggml_turbomind_mul_mat`, assert no shutdown happened and both outputs are within FP16 ULP of host-ref.
5. **P2.5** — `test_correctness.cpp` extension: run the existing single-device test on each of {GPU 0, GPU 1} in sequence in the same process; assert correctness on both.
6. **P2.6** — Bench `llama-bench` on MIN-16e single-GPU after the refactor to confirm no regression vs SPRINT-023 P5 baseline (16.6 t/s).

**P2 Gate**:
- ✅ Multi-device pack + dispatch in one process — no `shutdown` / `init` churn
- ✅ Single-GPU `test_correctness` still passes
- ✅ Single-GPU `llama-bench` MIN-16e ≥ 0.95× SPRINT-023 baseline
- ✅ `nm libggml-turbomind.so` shows no new C ABI symbols (ABI stability)

### P3 — CUDA_TURBOMIND family-alias buft (2 days)

**Goal:** `-ot 'exps=CUDA_TURBOMIND'` (no suffix) routes expert tensors to the layer's assigned device, preserving pipeline parallel.

1. **P3.1** — `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}`:
   - Add `ggml_backend_cuda_turbomind_buffer_type_family()` returning a sentinel buft named `CUDA_TURBOMIND` (no suffix).
   - The sentinel buft's `alloc_buffer` returns `nullptr` with an error message ("CUDA_TURBOMIND is a family alias; the loader must substitute"); it's not directly allocatable.
   - `ggml_backend_cuda_turbomind_get_extra_bufts(device)` includes the family buft in the returned list (alongside `CUDA_TURBOMIND<i>`).
   - `ggml_backend_buft_is_cuda_turbomind_family(buft)` predicate.
2. **P3.2** — `src/llama-model.cpp`: in the tensor placement / override resolution path, after layer-to-device assignment, detect family-buft matches and substitute the concrete `CUDA_TURBOMIND<layer_device>` for that tensor. Implementation choice: extend `make_gpu_buft_list` or the override-resolution closure to call a new helper `cuda_turbomind_resolve_family_alias(buft, layer_device)`.
3. **P3.3** — Crucially: the substitution must NOT populate `model.tensor_buft_overrides[]` (which would trip `has_tensor_overrides()`). The family-alias resolution happens "behind" the overrides system, applied to the tensor's `buft` field directly.
4. **P3.4** — Update `common/arg.cpp` `parse_tensor_buffer_overrides` if needed to accept `CUDA_TURBOMIND` (no suffix) as a valid buft name. Should JustWork™ because P3.1 adds it to the extras list.
5. **P3.5** — 4-GPU pod smoke load on DSv4-Flash-MIN-16e with `-sm layer -ot 'exps=CUDA_TURBOMIND'`. Verify in startup logs:
   - Each GPU shows non-zero `CUDA_TURBOMIND<i>` allocation.
   - `pipeline_parallel = 4` (or whatever the GPU count is).
6. **P3.6** — 32-token greedy decode completes without crash.

**P3 Gate**:
- ✅ `-ot 'exps=CUDA_TURBOMIND'` succeeds (no "unknown buffer type" error)
- ✅ MIN-16e loads on 4 GPUs with expert weights distributed
- ✅ Pipeline parallel flag is enabled (verify in startup log)
- ✅ Greedy decode completes

### P4 — Full 8-GPU DSv4-Flash-256e + q8_0 KV decision (2 days)

**Goal:** first successful end-to-end run.

1. **P4.1** — 8-GPU pod, load DSv4-Flash-256e:
   ```
   llama-server -m /models/DSv4-Flash-256e-fixed.gguf \
     -ngl 999 -sm layer -mg 0 -np 1 -c 4096 \
     -ot 'exps=CUDA_TURBOMIND' \
     --cache-type-k f16 --cache-type-v f16 \
     --no-mmap --no-warmup --host 127.0.0.1 --port 12399
   ```
   Capture per-GPU memory after load. If any GPU > 28 GiB, halt and switch to q8_0 KV.
2. **P4.2** — Q8_0 KV variant:
   ```
   --cache-type-k q8_0 --cache-type-v q8_0
   ```
   Compare per-GPU memory delta; expect ~2× KV reduction.
3. **P4.3** — Greedy 32-token decode on 5 fixed prompts; eyeball output for English coherence. (No CPU MoE baseline — running 256e on CPU is not feasible.)
4. **P4.4** — Capture `nvidia-smi dmon -s p -d 1` during decode; all 8 GPUs should show participation.

**P4 Gate**:
- ✅ Full 256e loads on 8 GPUs
- ✅ Per-GPU memory under 30 GiB during decode (with chosen KV dtype)
- ✅ 32-token greedy decode completes; output is coherent English
- ✅ All 8 GPUs participating

### P5 — Measurement + REPORT-19 (1-2 days)

**Goal:** the first headline numbers for 256e on this stack.

1. **P5.1** — `llama-bench -p 128 -n 32 -r 3` on the 8-GPU layer path. Record prefill / decode TPS.
2. **P5.2** — Scaling sweep: same model + bench at 2 / 4 / 6 / 8 GPUs (using smaller DSv4 variants on 2 and 4 GPU configs since 256e won't fit). Document per-GPU efficiency curve.
3. **P5.3** — Per-GPU VRAM at load + decode peak (single-row table per GPU).
4. **P5.4** — `nvidia-smi topo -m` capture into REPORT-19.
5. **P5.5** — Write `docs/sprints/SPRINT-025-REPORT-19.md`:
   - 8-GPU 256e bench (prefill + decode TPS)
   - Scaling sweep table
   - Per-GPU VRAM table
   - Topology capture
   - NCCL_DEBUG bootstrap log (artifact)
   - `# How to reproduce` shell block
   - Pipeline-parallel flag observed (yes / no)
   - LAYER vs ROW preliminary call (P6 will decide for real)

**P5 Gate**:
- ✅ 8-GPU 256e TPS recorded
- ✅ Scaling sweep complete (at least 4 GPU counts)
- ✅ REPORT-19 has reproducible commands

### P6 — Row-split investigation (conditional, kill criterion 1 day)

**Goal:** decision-complete verdict on ROW.

1. **P6.1** — Investigate the `deepseek4` row guard in `src/llama-model.cpp:770-771`. What's the actual missing piece? (Likely: row-split for the indirect MoE addressing path.)
2. **P6.2** — Assess: can the guard be lifted with ≤ 1 day of buffer-type-marriage work (CUDA_TURBOMIND × split buffer)? If no, defer with a precise blocker note in REPORT-19. If yes:
3. **P6.3** — Land the lift, run `-sm row -ot 'exps=CUDA_TURBOMIND'` on MIN-16e. Verify allreduce path exercised (NCCL log).
4. **P6.4** — Compare row-TP vs LAYER on the same model. Document delta.

**P6 Gate** (either fires):
- ✅ ROW measured and accepted, OR
- ✅ ROW deferred with precise blocker named

### P7 — Close-out (0.5 day)

1. If P4 shipped: tag `sprint-025-close`. Memory updates: `dsv4_flash_256e_multi_gpu_landed.md`.
2. Follow-ups doc filed for SPRINT-026 (multi-slot decode and/or row-TP if punted).
3. Single-GPU regression test (`test_correctness.cpp`) confirmed still passing on MIN-Ne (this is DoD item — should already be checked in P2).
4. Push to origin.

---

## 5. Files Summary

### Modified

| Path | Change |
|---|---|
| `ggml/vendor/turbomind/api.cc` | Replace `g_state` singleton with `g_states[TM_MAX_DEVICES]`. Per-device idempotent init. |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}` | Per-device init in `TmLib`; new `ggml_backend_cuda_turbomind_buffer_type_family()` + `_is_family()` predicate. |
| `src/llama-model.cpp` | Family-alias resolution: detect family buft assignment, substitute concrete `CUDA_TURBOMIND<device>` based on layer's assigned device. Do NOT populate `tensor_buft_overrides`. |
| `tests/test-nccl-allreduce.cpp` | New — 2-GPU allreduce smoke. |
| `tests/test-turbomind-multi-device.cpp` | New — 2-GPU pack+dispatch smoke. |

### Possibly modified (P6 conditional)

| Path | Change |
|---|---|
| `src/llama-model.cpp:770-771` | Lift `deepseek4` `LLAMA_SPLIT_MODE_ROW` guard. |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}` | Split-buffer compatibility (if row TP is pursued). |

### New (docs / artifacts)

| Path | Purpose |
|---|---|
| `docs/sprints/SPRINT-025-REPORT-19.md` | Measurement narrative. |
| `docs/sprints/SPRINT-025-P{0..7}-summary.md` | Per-phase summaries. |
| `homelab/manifests/llamacpp-build-4gpu.yaml` (or equivalent) | 4-GPU pod spec. |
| `homelab/manifests/llamacpp-build-8gpu.yaml` | 8-GPU pod spec. |
| `Dockerfile` update (or new image) | Build image with `libnccl2 libnccl-dev`. |

---

## 6. Definition of Done

1. ✅ 4-GPU + 8-GPU pod manifests exist and pods boot on `gpu-01`.
2. ✅ Build image has NCCL libs (`libnccl2 libnccl-dev`).
3. ✅ `llama-server` / `llama-bench` link against `libnccl` (`ldd` confirms).
4. ✅ NCCL allreduce smoke harness completes; `NCCL_DEBUG=INFO` Bootstrap log archived.
5. ✅ Per-device CUDA_TURBOMIND lifecycle — no global "current device" on the hot path; 2-device pack+dispatch smoke passes.
6. ✅ `CUDA_TURBOMIND` family-alias buft works: `-ot 'exps=CUDA_TURBOMIND'` succeeds; pipeline parallel preserved.
7. ✅ Full DSv4-Flash-256e-fixed.gguf loads on 8 V100s with `LLAMA_SPLIT_MODE_LAYER`.
8. ✅ 32-token greedy decode completes; output is coherent English.
9. ✅ Per-GPU VRAM stays within 30 GiB threshold during decode.
10. ✅ All 8 GPUs participate (visible in `nvidia-smi dmon`).
11. ✅ `llama-bench -p 128 -n 32 -r 3` numbers captured for 8-GPU path.
12. ✅ Scaling-efficiency sweep across 2 / 4 / 6 / 8 GPUs documented.
13. ✅ REPORT-19 contains: bench tables, VRAM tables, topology capture, NCCL log, reproducible command line.
14. ✅ LAYER vs ROW decision recorded (either measured ROW or precise defer blocker).
15. ✅ Single-GPU `test_correctness.cpp` still passes (no regression from TmLib refactor).
16. ✅ Single-GPU `llama-bench` MIN-Ne TPS ≥ 0.95× SPRINT-023 P5 baseline.
17. ✅ Tag `sprint-025-close`.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | TmLib refactor (Option A) exposes a deeper turbomind library-level singleton | Medium | High | P2.4 smoke catches it; escalate to Option B opaque-context as Plan B |
| 2 | Family-alias substitution misses a code path that records `tensor_buft_overrides` and silently disables pipeline parallel | Medium | Medium | P3 startup log check; verify `pipeline_parallel` flag = N_gpus |
| 3 | Heaviest-shard GPU (embedding + lm_head + a few MoE layers) OOMs | Medium | High | P4 stop-the-line at 28 GiB after-load + 30 GiB during decode; fallback to `-ts` rebalance or q8_0 KV |
| 4 | NCCL bootstrap fails on intra-node V100 SXM2 NVLink mesh (partial topology) | Low-Medium | Medium | P0.3 `nvidia-smi topo -m` capture; `NCCL_P2P_DISABLE=1` fallback documented |
| 5 | Build/runtime NCCL ABI mismatch (libnccl.so version skew) | Low | Medium | Build + runtime use the same image; pin NCCL version |
| 6 | `ncclCommInitAll` enumerates more devices than pod requested | Low | High | P0 visibility gate requires `nvidia-smi -L` count = pod GPU request |
| 7 | DSv4-Flash-256e GGUF file corruption / unreadable | Low | High | `sha256sum` capture in P4.1; alternate read path |
| 8 | Row guard lift (P6) cascades into more buffer-type marriage than 1 day | High (it does) | Medium | Kill criterion: defer to SPRINT-026 if > 1 day |
| 9 | Single-GPU regression from per-device refactor | Low | Medium | P2.5 + P2.6 explicit no-regression check |
| 10 | CMake cache poisoning from prior single-GPU build | Medium | Low | `rm -rf build/` in P1.1 |
| 11 | Pipeline parallel disabled by some other code path (not tensor_buft_overrides) | Low | Medium | P3.5 explicit log check |
| 12 | Decode hangs because some operation needs allreduce that NCCL didn't init | Low-Medium | High | P1.5 smoke harness must succeed before P4 |

---

## 8. Security

Local k8s cluster, no internet-facing service.

- 4-GPU and 8-GPU pods request `nvidia.com/gpu` resources only; no `securityContext: privileged: true`.
- Build image is shared with existing `llamacpp-build` workflow; no new credential surface.
- NCCL is intra-node only (single node, all on `gpu-01`); no inter-node networking. Pod-to-pod NCCL communication not in scope.
- Model GGUF mounted read-only from existing PVC.
- Workspace emptyDir grants build + report write access only.

---

## 9. Dependencies

1. **SPRINT-023 + SPRINT-024 code** — `CUDA_TURBOMIND` buffer type (per-device), pack pipeline, dispatch helpers, P2.3 correctness gate plumbing.
2. **`gpu-01` with 8 visible V100-SXM2-32GB** — verified at planning time.
3. **`DSv4-Flash-256e-fixed.gguf` (156 GiB)** — verified present at `/models/`.
4. **`libnccl2` + `libnccl-dev`** matching CUDA 12.2 (NCCL 2.18+).
5. **CUDA 12.2 / gcc 11.4 / cmake 3.22** — same as SPRINT-023.
6. **Homelab manifest authoring access** for the 4-GPU and 8-GPU pod specs.

---

## 10. Open Questions

1. **Default context length for the first 8-GPU landing** — 4k or 8k? Driven by per-GPU KV headroom in P4.1 measurement.
2. **Heaviest-shard auto-rebalance** — if the default `-sm layer` distribution OOMs one GPU, do we ship with manual `-ts` tuning or invest in auto-balancing? P4 will surface.
3. **Per-device `cudaMalloc` for turbomind scratch** — already in `ggml_turbomind_init`; with 8 GPUs that's 8 × 256 MiB = 2 GiB lost to scratch. Optimize via the ggml CUDA pool in a follow-on if it shows up in VRAM accounting.
4. **Does `ggml_backend_cuda_allreduce_tensor` get exercised on the LAYER path at all?** Probably not. P5 logs zero allreduce calls under LAYER; that's expected. Only matters if P6 ROW work proceeds.
5. **MAX_DEVICES** — pick a safe upper bound for `g_states[TM_MAX_DEVICES]`. 32 is overkill for V100 fleets; pick 16 if memory matters, 32 for headroom.

---

## 11. Outcome contract

Sprint ships if:
- 8-GPU 256e loads and decodes coherently.
- Per-device TURBOMIND verified executing across all 8 GPUs (memory + NCCL_DEBUG logs).
- No single-GPU regression on MIN-Ne (≥ 0.95× SPRINT-023 baseline).
- REPORT-19 has bench + VRAM + scaling sweep + reproducible commands.

Stop-loss: if P2 Option A refactor exposes a deeper turbomind library-level singleton that needs Option B (versioned opaque-context API), escalate as a follow-up sprint; ship Option A as far as it goes with the documented limitation.

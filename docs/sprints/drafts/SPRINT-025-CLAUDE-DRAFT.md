# SPRINT-025 — Full DSv4-Flash-256e on multi-V100 k8s pod (CLAUDE draft)

**Status:** DRAFT 2026-05-15
**Predecessor:** SPRINT-023 (CUDA_TURBOMIND buft landed, per-device); SPRINT-024 (grouped MoE dispatch — planned, not blocking)
**Successor:** SPRINT-026 (multi-slot decode / speculative; possibly TP-aware grouped MoE)

---

## 1. Overview

Stand up the full `DSv4-Flash-256e-fixed.gguf` (~156 GiB) on the `gpu-01` 8× V100-SXM2-32GB node in the homelab microk8s cluster, sharded with llama.cpp's existing tensor-parallel scaffolding. The work splits cleanly into three concerns:

1. **Hardware + build environment** — provision a k8s pod with the right `nvidia.com/gpu` request, install NCCL into the build image, turn on `GGML_CUDA_NCCL`, link against `NCCL::NCCL`, and verify the resulting `llama-server` actually depends on `libnccl.so`.
2. **Multi-GPU dispatch correctness** — make the SPRINT-023 CUDA_TURBOMIND buffer type work *simultaneously* across N devices. Today the `TmLib` singleton in `ggml-cuda-turbomind.cu:46-101` flips its single device context on every cross-device call (`shutdown()` → `init(new_device)` at lines 72-75). That defeats multi-GPU; SPRINT-025 P2 makes the loader per-device.
3. **Load + run + measure** — load the 156 GiB GGUF with `-sm layer` (default split mode for MoE; no allreduce in the critical decode path), confirm end-to-end greedy decode produces coherent English on a fixed prompt set, and capture `llama-bench` numbers in REPORT-19.

**Scope is explicit:** no new kernels, no expert parallelism, no inter-node networking, no perplexity sweep, no multi-slot decode. The reusable primitive this sprint delivers is **per-device CUDA_TURBOMIND state with a NCCL-linked build**, which all downstream multi-GPU work needs.

**Perf framing (per intent open question 4):** No hard TPS gate. SPRINT-025 is the first run of the full 256e variant on this stack — there is no comparable baseline. The number landed in REPORT-19 *is* the baseline for SPRINT-026.

---

## 2. Use Cases

Each phase delivers something usable even if subsequent phases slip:

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | 8-GPU pod manifest committed; NCCL installed in the dev image; `nvidia-smi` and `nccl-tests` work inside the pod. |
| P1 | `llama-server` builds with `GGML_USE_NCCL` defined and links against `libnccl.so`. Reusable for any future multi-GPU work. |
| P2 | Per-device TmLib refactor — CUDA_TURBOMIND is genuinely per-device, no singleton flip. Unblocks anything that uses TURBOMIND on > 1 GPU. |
| P3 | Real 156 GiB model loads on 8 GPUs with `-sm layer`; smoke decode passes coherence eyeball. |
| P4 | TURBOMIND `-ot 'exps=CUDA_TURBOMIND[0-7]'` works across all GPUs; expert dispatch verified per device. |
| P5 | `llama-bench` numbers captured; REPORT-19 written with a clear narrative of where the time goes. |
| P6 | Tag, memory update, followups filed for SPRINT-026 (multi-slot, row-split, NCCL allreduce coverage). |

---

## 3. Architecture

### 3.1 Hardware topology

`gpu-01` (microk8s node) reports `nvidia.com/gpu.count: 8`, all Tesla V100-SXM2-32GB. SXM2 means NVLink-2 between adjacent GPU pairs (V100 SXM2 → 6 NVLink lanes per GPU, hybrid mesh; not fully connected — pairs share NVLink while cross-pair traffic falls back to PCIe). NCCL auto-detects the topology and routes ring/tree reductions over the fastest path; we do not need to set `NCCL_P2P_LEVEL` or `NCCL_TOPO_FILE` for single-node TP.

### 3.2 Pod size: 4 vs 8 (intent open question 2 — resolved)

The model file is 156 GiB. Per-GPU model footprint with full offload (`-ngl 999 -sm layer`) is **`(156 GiB + workspace + KV) / n_gpu`**:

| Config | Total VRAM | Per-GPU model | Headroom for KV + workspace |
|---|---|---|---|
| 4× V100 32 GiB | 128 GiB | 39 GiB | **−7 GiB** — does NOT fit even with zero KV |
| 8× V100 32 GiB | 256 GiB | 19.5 GiB | 12.5 GiB per GPU — comfortable |

**Conclusion:** 8 GPUs is the only viable config for full offload of the 256e model. 4-GPU mode would require partial offload (`-ncmoe N` to keep N MoE layers on CPU) and adds a comparison axis we don't need this sprint.

**Decision:** SPRINT-025 targets **8× V100** for full offload. 4-GPU partial-offload is deferred to a followup (or to SPRINT-026 if multi-slot decode wants smaller pods). The 8-GPU pod also has the benefit of measuring the natural NVLink/PCIe asymmetry which informs SPRINT-026's row-split decision.

### 3.3 NCCL build wiring

`ggml/src/ggml-cuda/CMakeLists.txt:184-192` already has the option:

```cmake
if (GGML_CUDA_NCCL)
    find_package(NCCL)
    if (NCCL_FOUND)
        add_compile_definitions(GGML_USE_NCCL)
        target_link_libraries(ggml-cuda PRIVATE NCCL::NCCL)
    else()
        message(STATUS "Warning: NCCL not found, ...")
    endif()
endif()
```

The `NCCL::NCCL` imported target comes from a CMake-shipped `FindNCCL.cmake` (not standard with CMake 3.22 — needs upstream `Modules/FindNCCL.cmake` from the llama.cpp tree). Two implementation choices:

**(a)** Install Ubuntu's `libnccl2 libnccl-dev` (provided by the NVIDIA APT repo for CUDA 12.2). The `nvidia/cuda:12.2.2-devel-ubuntu22.04` base image already includes the NVIDIA APT key; one `apt-get install -y libnccl2 libnccl-dev` line in the Dockerfile.

**(b)** Vendor NCCL source and build. Heavier; rejected unless (a) fails.

**Decision:** Path (a). Verify `find_package(NCCL)` resolves before changing anything in CMake.

`cmake .. -DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=70` is the configure line. Build verification: `ldd build/bin/llama-server | grep nccl` must show `libnccl.so.2`.

### 3.4 Split mode: LAYER vs ROW (intent open question 1 — resolved)

`llama.cpp` exposes four split modes via `LLAMA_SPLIT_MODE_*`:

- **`LAYER`** (default for layered models): assign whole transformer layers to one GPU each; activations cross GPUs via point-to-point copies (one per layer boundary). No allreduce needed in the forward pass. Comm volume = `hidden_dim × n_tokens × bytes_per_elem` per layer boundary.
- **`ROW`**: split each linear's output rows across GPUs; needs `ncclAllReduce` after every output projection. Higher comm volume but better latency at high batch.
- **`NONE`**: single-GPU fallback. Not applicable.
- **`TENSOR`**: legacy synonym; same as LAYER in current code paths.

For DSv4-Flash's MoE topology — top-k routing, 256 experts per layer, sparse activation — `LAYER` is the correct first choice:

1. **No allreduce in the critical path.** Decode is M=1; allreduce on 7168-dim FP32 vectors costs ~50 µs per call on V100 PCIe. ROW would fire two allreduces per layer × 58 layers = 5800 µs per token. That alone caps decode at ~170 t/s before any GEMM work.
2. **MoE doesn't benefit from row-split anyway.** Each expert's GEMM is already small (N=2560, K=7168); splitting across 8 GPUs would shrink the M=1 GEMM to row count 320, well below useful WMMA utilization.
3. **TURBOMIND's per-device buft is layer-natural.** Each layer's expert weights live on one device's CUDA_TURBOMIND buft; activations cross device boundaries via the existing ggml graph allocator. No change to the buft layout.

**Decision:** SPRINT-025 ships `-sm layer`. ROW-split is not tested. Allreduce is *built* (via `GGML_USE_NCCL`) but not exercised by the decode hot path; it will fire only on whatever globally-reduced tensors the graph builder emits (typically none with LAYER split).

P5.3 measures comm cost to confirm the LAYER decision. If LAYER-mode per-token latency turns out > 200ms and the breakdown shows boundary copies dominate, SPRINT-026 picks up the ROW comparison.

### 3.5 Per-device CUDA_TURBOMIND init (the singleton bug)

Today `ggml-cuda-turbomind.cu:46-101` holds a *single global* `TmLib`:

```cpp
struct TmLib { /* one handle, one init_device */ };
TmLib & g_tm() { static TmLib t; return t; }

bool tm_ensure_loaded(int device) {
    if (t.init_device != device) {
        t.shutdown();
        if (t.init(device) != 0) return false;
        t.init_device = device;
    }
    return true;
}
```

This is **lethal for multi-GPU**: every cross-device dispatch tears down the turbomind context on the previous device and re-inits on the new one. `ggml_turbomind_init(int cuda_device)` allocates a `Gemm` instance, registers kernels for that device, and stamps the CUDA context — `shutdown()` frees the device-side `Gemm`. Tensors uploaded under `init(0)` will have packed weight pointers tied to device 0; if we then `shutdown(); init(1)` and try to dispatch on device 0's weights, behavior is undefined.

The buft itself is already per-device (line 363: `ggml_backend_cuda_turbomind_buffer_type(int device)`), so the data model is fine. Only the *loader* is wrong.

**Fix (P2):** Replace the singleton with `std::array<TmLib, GGML_CUDA_MAX_DEVICES>`, indexed by device. Each slot owns its own `dlopen` handle and its own `init(device)` call. Concretely:

```cpp
struct TmLib { /* unchanged fields */ };
std::array<TmLib, GGML_CUDA_MAX_DEVICES> g_tm_slots;
TmLib & g_tm(int device) { return g_tm_slots.at(device); }

bool tm_ensure_loaded(int device) {
    TmLib & t = g_tm(device);
    std::lock_guard<std::mutex> lk(t.mtx);
    if (t.tried_load) return t.loaded;
    t.tried_load = true;
    /* dlopen once per device — symbols are process-global so dlopen returns the
       same handle, but per-slot bookkeeping keeps init_device pinned. */
    t.handle = dlopen("libggml-turbomind.so", RTLD_NOW | RTLD_LOCAL);
    /* resolve symbols, then init(device) */
}
```

Caveats:
- `dlopen` of the same .so multiple times returns the same handle (ref-counted). Symbol resolution is process-global; the `Gemm` instance inside the library is *not* device-local unless turbomind's init explicitly per-device-stamps it. Need to verify in P0.3 whether the underlying `libggml-turbomind.so` actually supports concurrent multi-device init or whether it shares a single global `Gemm`.
- If the library has a singleton `Gemm` inside, we have a deeper problem: SPRINT-025 P2 either patches the library or serializes dispatches per-device with cuda-set-device wrappers and accepts the perf hit.

**P0.3 answers this before any integration work.** If the library is multi-device-safe, the wrapper fix above is sufficient. If not, P2 grows to include a library-side patch.

### 3.6 VRAM budget per GPU

Per-GPU usage on 8× 32 GiB with `-sm layer`:

| Component | Per-GPU bytes | Notes |
|---|---|---|
| Model weights (156 GiB / 8) | 19.5 GiB | Even split, give or take a few layers depending on layer-distribution heuristic |
| TURBOMIND packed scales | ~0.5 GiB | E8M0 / fp scale tables; SPRINT-023 measured ~3% overhead vs raw weight bytes |
| KV cache (FP16) for 2K context, 64 head, 128 head_dim | ~2 GiB | Per-layer-distributed; 58 layers / 8 GPU ≈ 7-8 layers per GPU |
| KV cache (FP16) for 8K context | ~8 GiB | Watch for overflow at >2K; fall back to `--cache-type-k q8_0 --cache-type-v q8_0` (4× reduction) |
| Activation workspace + temp tensors | ~2 GiB | ggml_cuda_pool transient |
| CUDA context overhead | ~1 GiB | cuBLAS handles, cuDNN if linked, NCCL comms |
| **Total at 2K ctx** | **~25 GiB** | 7 GiB headroom — comfortable |
| **Total at 8K ctx FP16 KV** | **~31 GiB** | 1 GiB headroom — tight; q8_0 KV recommended |

**Plan:**
- Default: 2K context, FP16 KV.
- Stretch: 8K context with `--cache-type-k q8_0 --cache-type-v q8_0` (KV drops to ~2 GiB).
- Hard cap: any context where any GPU exceeds 30 GiB triggers KV quantization.

P3.3 captures per-GPU memory breakdown via `llama-server`'s startup log (`load_tensors: CUDA0 buffer size = N MiB` lines) — confirms the model split is within budget on every device.

### 3.7 Allreduce path (when it fires)

With `-sm layer`, ggml builds the graph such that each tensor lives on one GPU; the splitter inserts `GGML_OP_VIEW` / contiguous copies across device boundaries. `ggml_backend_cuda_allreduce_tensor` (declared `ggml-cuda.h:31`, implemented `ggml-cuda.cu:1248-1310`) fires only when *the same logical tensor exists on multiple backends* — typically the output of a row-split linear, or the residual stream when using full TP. With LAYER split, this code path runs zero times in the decode hot path.

But it *must compile and link* correctly because:
1. The build sets `GGML_USE_NCCL` so the function body is non-empty.
2. `ncclCommInitAll` at `ggml-cuda.cu:466` runs at backend init, before split mode is even known.

**P1.5 sanity check:** add a small synthetic that forces an allreduce on a 1024-element FP32 tensor across 2 devices and checks the sum is correct. Lives in `tests/test-backend-ops` or a one-off harness in this sprint's scratch dir. Catches NCCL init/link issues without needing a model load.

### 3.8 Layer-distribution heuristic and `-ts`

`tensor_split[128]` lets the user weight per-GPU layer assignment. Default is uniform. For DSv4-Flash with 58 transformer layers across 8 GPUs, uniform gives 7-8 layers per GPU. We do not override `-ts` in this sprint — uniform is the test condition. If P5 measurement shows imbalance (one GPU bottlenecks), `-ts` tuning is a SPRINT-026 lever.

---

## 4. Implementation

### P0 — Hardware + dev image bring-up (1-2 days)

**Goal:** an 8-GPU k8s pod where the existing `llamacpp-build` workflow runs and `nccl-tests/build/all_reduce_perf` succeeds.

1. **P0.1** — Author `k8s/llamacpp-build-8gpu.yaml` (or modify the existing manifest) with `resources.limits.nvidia.com/gpu: 8`. Schedule on `gpu-01` (only node with 8 GPUs). Same image base as the 1-GPU pod for now; do NCCL install in P0.2.
2. **P0.2** — Extend the build image with `RUN apt-get install -y libnccl2 libnccl-dev`. Pin the version to match CUDA 12.2 (libnccl2=2.18.5-1+cuda12.2 or current Ubuntu APT). Push image; the pod manifest references the new tag.
3. **P0.3** — Inside the pod, build and run `nccl-tests/all_reduce_perf -b 8 -e 1G -f 2 -g 8`. Verify all-reduce completes across 8 GPUs and reports a reasonable bandwidth (V100 SXM2 expected ~80-120 GB/s ring bus on a 6-link mesh). **Also runs a smoke check on whether `libggml-turbomind.so` can be initialized on multiple devices in sequence within one process** — write a 30-line C program that calls `ggml_turbomind_init(0)`, `ggml_turbomind_init(1)`, allocates a small weight on each, runs `ggml_turbomind_mul_mat` on each, and checks for finite output. This validates the per-device assumption in §3.5.
4. **P0.4** — `nvidia-smi topo -m` from inside the pod. Document the NVLink/PCIe topology (which pairs are NVLink-connected). Informs SPRINT-026 row-split discussion; not actioned in this sprint.

**P0 Gate**:
- ✅ 8-GPU pod schedules on `gpu-01` and `nvidia-smi -L` shows 8 V100s
- ✅ Build image has `/usr/lib/x86_64-linux-gnu/libnccl.so.2` present
- ✅ `nccl-tests` runs to completion across 8 GPUs with non-zero bandwidth
- ✅ Multi-device turbomind init smoke (P0.3 program) prints finite output on both devices
- ✅ Topology dump committed to `docs/sprints/SPRINT-025-P0-summary.md`

### P1 — NCCL-enabled llama.cpp build (1 day)

**Goal:** `llama-server` binary that loads `libnccl.so.2` and exposes the allreduce path.

1. **P1.1** — Verify CMake finds NCCL: `cmake .. -DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release`. Configure log must include `Found NCCL: ...` (from `find_package(NCCL)`). Halt and inspect `cmake/Modules/FindNCCL.cmake` if not.
2. **P1.2** — Build `ggml-cuda`, `llama-server`, `llama-bench`, `llama-cli`. Should take ~25 min for a cold build on the 8-GPU pod.
3. **P1.3** — Link verification: `ldd build/bin/llama-server | grep -E 'nccl|cuda'` shows `libnccl.so.2`, `libcudart.so.12`, `libcuda.so.1`. Run `nm build/bin/llama-server | grep ncclAllReduce` — symbol must resolve (not a weak undef).
4. **P1.4** — Backend init smoke: run `llama-server --help` and verify it doesn't crash on `ncclCommInitAll`. The init at `ggml-cuda.cu:466` runs unconditionally when `GGML_USE_NCCL` is defined, even for non-TP loads — this catches NCCL init bugs without needing a model.
5. **P1.5** — Standalone allreduce harness (`tools/scratch/test_allreduce.cpp` — temp, not committed long-term unless useful): allocate FP32 tensor on devices 0 and 1, call `ggml_backend_cuda_allreduce_tensor` with both backends, verify the output is `2× input` (since same data on both). Build with the project's existing CMake.

**P1 Gate**:
- ✅ Build completes with `Found NCCL` in the configure log
- ✅ `ldd` shows `libnccl.so.2` linked
- ✅ `llama-server --help` runs without crash
- ✅ Standalone allreduce harness passes

### P2 — Per-device CUDA_TURBOMIND init (1-2 days)

**Goal:** lift the TmLib singleton; CUDA_TURBOMIND works on all 8 devices simultaneously.

1. **P2.1** — Refactor `ggml-cuda-turbomind.cu:46-101`:
   - Replace `static TmLib t` singleton with `std::array<TmLib, GGML_CUDA_MAX_DEVICES> g_tm_slots`.
   - `g_tm(int device)` returns the per-device slot.
   - `tm_ensure_loaded(int device)` initializes that slot's handle and calls `init(device)`. Never tears down a different device's slot.
2. **P2.2** — Audit every caller of `g_tm()` (search for the old function) and pass the device through. Buffer context already carries `device` (`ggml_backend_cuda_tm_buffer_context::device` at line 130); the dispatch path already has `ctx->device` available. Pure mechanical change.
3. **P2.3** — Shutdown ordering: `tm_shutdown_all()` (new) iterates slots and shuts down loaded ones. Wired to backend deinit if such a hook exists (otherwise relies on process exit; the .so cleans up via destructor). Keep this minimal — multi-device init bugs are the risk, multi-device shutdown is not on the critical path.
4. **P2.4** — Single-device regression test: run SPRINT-023 P5's MIN-16e smoke decode on 1 GPU with the new loader. Output must match the pre-refactor binary bit-for-bit (same kernel, same init, just a different slot-lookup path).
5. **P2.5** — Multi-device smoke: load a *small* model first (anything with MoE that fits 2 GPUs — DSv4-Flash-MIN-16e or AVG-16e) with `-sm layer -ngl 999 -ot 'exps=CUDA_TURBOMIND0,exps=CUDA_TURBOMIND1'`. Verify both GPUs allocate TURBOMIND buft and dispatch fires on both. Use `GGML_TM_VERBOSE=1` (SPRINT-024 P0.2 instrumentation) if landed; otherwise add a temporary `printf` in `ggml_cuda_mul_mat_turbomind` showing the device for each dispatch.

**P2 Gate**:
- ✅ Single-device output identical to SPRINT-023 P5 baseline
- ✅ Multi-device smoke shows dispatches on both target GPUs without device-flip thrashing
- ✅ No `cudaErrorInvalidDevice` or `cudaErrorInvalidContext` in the run

### P3 — 156 GiB model load on 8 GPUs (1-2 days)

**Goal:** the 256e model loads with `-sm layer -ngl 999` across 8 V100s and a smoke decode produces coherent English.

1. **P3.1** — Stage `DSv4-Flash-256e-fixed.gguf` on the pod's mounted storage (`/srv/dev/...` or `/models/`, per homelab convention — confirmed via the `homelab-k8s-dev` skill context). 156 GiB; copy once via the host's NFS share, not over network from a laptop.
2. **P3.2** — `llama-server -m /models/DSv4-Flash-256e-fixed.gguf -ngl 999 -sm layer -mg 0 -c 2048 --host 0.0.0.0 --port 8080 -t 8`. Confirm:
   - Model loads without OOM.
   - Per-GPU buffer sizes printed in startup log are ≤ 22 GiB each (model + scales).
   - All 8 GPUs reported as `CUDA0..CUDA7` in the backend list.
3. **P3.3** — `/v1/completions` smoke: 32-token greedy completion (`temperature=0, max_tokens=32`) on a fixed set of 10 prompts (subset of SPRINT-024's prompt list — see §4.P3.4). Each completion must:
   - Return within 60 s (i.e., no hang).
   - Produce ASCII / UTF-8 output, not gibberish bytes.
   - Pass an eyeball coherence check (full English sentences, no obvious repetition collapse).
4. **P3.4** — Save the 10 completions to `docs/sprints/SPRINT-025-P3-completions.md` for the report.
5. **P3.5** — Per-GPU activity check: run `nvidia-smi dmon -s pucm -d 1` in a sidecar pod / `kubectl exec` during decode. SM utilization should be non-zero on all 8 GPUs (LAYER split means activity is roughly serial per token; we expect ~12.5% average utilization per GPU during decode, with bursts on the layer-owner GPU). Memory-used per GPU must match the load-time breakdown ±5%.

**P3 Gate**:
- ✅ Model loads with `-sm layer` on 8 GPUs without OOM
- ✅ All 10 prompts decode 32 tokens without crash, hang, or gibberish
- ✅ Per-GPU memory budget within 22 GiB
- ✅ nvidia-smi confirms multi-GPU activity during decode

### P4 — TURBOMIND on all 8 GPUs (1 day)

**Goal:** experts dispatched through CUDA_TURBOMIND across every device.

1. **P4.1** — Re-load with explicit per-device TURBOMIND placement: `-ot 'blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=CUDA_TURBOMIND'`. The current `-ot` regex resolution in llama.cpp matches buft *names*; without a device qualifier, the placement is decided by which device owns the layer (per `-sm layer`). Confirm in startup log: each layer's `*_exps.weight` lands on the same device as that layer's other tensors and on the `CUDA_TURBOMIND<i>` buft.
2. **P4.2** — If the `-ot` regex syntax requires explicit per-device pinning (e.g., `CUDA_TURBOMIND0` etc.), test the explicit form: `-ot 'blk\.[0-7]\..*exps=CUDA_TURBOMIND0,blk\.[8-9]|blk\.1[0-5]\..*exps=CUDA_TURBOMIND1,...'`. Document the working form.
3. **P4.3** — Verify dispatch path: add a one-shot counter to the per-device TmLib slot's `pfn_mul_mat` wrapper that increments per call. After 32-token decode, dump per-device counts. Expect roughly equal totals across all 8 devices (each device handles ~7 layers × 32 tokens × 2-3 MoE linears × n_active_experts).
4. **P4.4** — Smoke decode again with the explicit TURBOMIND placement; verify same 10 prompts produce semantically equivalent output to P3.3 (no quality regression from TURBOMIND dispatch). 32-token leading-token match ≥ 75% against P3.3 baseline.

**P4 Gate**:
- ✅ TURBOMIND dispatch counter shows non-zero on all 8 devices
- ✅ Per-device dispatch counts within 30% of mean (rough load balance)
- ✅ Decode output passes coherence check
- ✅ Leading-token match ≥ 75% vs P3.3 (without TURBOMIND `-ot`, default CUDA path)

### P5 — Measurement + REPORT-19 (1-2 days)

**Goal:** capture the headline numbers and the where-the-time-goes narrative.

1. **P5.1** — `llama-bench -m /models/DSv4-Flash-256e-fixed.gguf -ngl 999 -sm layer -p 128 -n 32 -r 3` in two configurations:
   - Default CUDA (no TURBOMIND `-ot`)
   - TURBOMIND on experts (P4.1's regex)
   Record `pp 128` and `tg 32` mean ± stddev across 3 reps. Three rows total (well, two configs × the same model).
2. **P5.2** — Comm breakdown (LAYER mode confirmation): time a single-token decode with `CUDA_LAUNCH_BLOCKING=1` (forces sync — DO NOT use for the headline numbers) and a per-kernel trace. Aim is to confirm the layer-boundary copies are *not* the dominant cost; if they are, document for SPRINT-026 row-split work. Use `nsys profile -t cuda,nvtx --stats=true` for the breakdown.
3. **P5.3** — Per-GPU utilization: `nvidia-smi dmon -s pucm -d 1` capture during a 200-token decode; record mean SM% and mean VRAM-used per GPU. Confirm the model split is balanced.
4. **P5.4** — Memory headroom: log `nvidia-smi --query-gpu=memory.used,memory.free --format=csv` at peak decode. Confirm ≥ 10% free on every device.
5. **P5.5** — Write `docs/sprints/SPRINT-025-REPORT-19.md`. Required sections:
   - Pod manifest and image tag
   - NCCL version and link verification output
   - Bench table (prefill/decode TPS, 2 configs)
   - Per-GPU memory and utilization
   - Comm breakdown (NSYS summary)
   - Decision: SHIP (it works) / EXTEND (needs more work to be useful) / STOP
   - Followups for SPRINT-026 (multi-slot, row-split comparison if warranted)

**P5 Gate**:
- ✅ Bench table captured for both configs
- ✅ NSYS or equivalent breakdown captured
- ✅ Memory accounting per GPU recorded
- ✅ REPORT-19 ends with a SHIP / EXTEND / STOP verdict

### P6 — Close-out (0.5 day)

1. Update memory: add `dsv4_256e_multi_gpu_landed.md` with the key gotchas (per-device TmLib refactor, NCCL install path, 8-GPU minimum, `-ot` regex form that worked).
2. Tag `sprint-025-close`.
3. File `docs/sprints/SPRINT-025-FOLLOWUPS.md` with:
   - Row-split TP comparison (if comm breakdown suggests it'd help)
   - Multi-slot decode (deferred #1 from SPRINT-024-DEFERRED.md, now SPRINT-026's main thread)
   - 4-GPU partial-offload mode for smaller deployments
   - CUDA-graph capture across TP boundaries (SPRINT-024-DEFERRED #11; may need re-validation)
4. Push branch; do not open upstream PR (per AGENTS.md / SPRINT-023 hygiene).

---

## 5. Files Summary

### Modified

| Path | Change |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` | Per-device `TmLib` array; callers thread `device` through `tm_ensure_loaded`; per-slot dlopen + init. |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` | (Possibly) declaration of `tm_shutdown_all` if backend deinit hook is wired. |

### New

| Path | Purpose |
|---|---|
| `k8s/llamacpp-build-8gpu.yaml` (or similar) | 8-GPU pod manifest pinned to `gpu-01`. |
| Dockerfile delta for the build image | `apt-get install -y libnccl2 libnccl-dev` line. |
| `docs/sprints/SPRINT-025-P{0..6}-summary.md` | Per-phase summaries. |
| `docs/sprints/SPRINT-025-REPORT-19.md` | Headline measurement + ship decision. |
| `docs/sprints/SPRINT-025-FOLLOWUPS.md` | Row-split, multi-slot, partial-offload deferrals. |
| `docs/sprints/SPRINT-025-P3-completions.md` | The 10 smoke completions for the record. |

### Possibly modified (not expected; flagged for caution)

| Path | Change |
|---|---|
| `ggml/vendor/turbomind/api.cc` | Only if P0.3 reveals an internal singleton `Gemm` in the library and we must patch it for true multi-device. |
| `cmake/Modules/FindNCCL.cmake` | Only if `find_package(NCCL)` fails; vendor or write a finder. |

---

## 6. Definition of Done

1. ✅ 8-GPU k8s pod schedules on `gpu-01` and exposes 8 V100s to the build image.
2. ✅ Build image installs `libnccl2 libnccl-dev`; `llama-server` links against `libnccl.so.2` (verified via `ldd`).
3. ✅ CMake configure log shows `Found NCCL`; `GGML_USE_NCCL` defined; `ncclCommInitAll` runs at backend init without crash.
4. ✅ `TmLib` is per-device; single-device runs are bit-identical to SPRINT-023 baseline.
5. ✅ `DSv4-Flash-256e-fixed.gguf` (156 GiB) loads with `-sm layer -ngl 999` on 8 GPUs without OOM; per-GPU usage ≤ 30 GiB.
6. ✅ 10-prompt smoke decode at `temperature=0, max_tokens=32` produces coherent English on every prompt; no NaN, no hang, no crash.
7. ✅ TURBOMIND `-ot 'exps=...'` placement works across all 8 devices; per-device dispatch counter is non-zero everywhere.
8. ✅ `llama-bench -p 128 -n 32 -r 3` numbers captured for default-CUDA and TURBOMIND-exps configs.
9. ✅ REPORT-19 contains: pod manifest, NCCL link output, bench table, comm breakdown, per-GPU memory + utilization, SHIP/EXTEND/STOP verdict.
10. ✅ SPRINT-025-FOLLOWUPS.md filed with row-split, multi-slot, 4-GPU partial-offload as next sprint candidates.
11. ✅ Tag `sprint-025-close`.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | `libggml-turbomind.so` has an internal singleton `Gemm` that the per-device wrapper can't fix from outside | Medium | High | P0.3 multi-device init smoke before integration; if positive, P2 grows to include a library-side patch (extra ~1 day) |
| 2 | NCCL APT install pulls a version incompatible with CUDA 12.2 | Low-Medium | Medium | Pin libnccl2 version; fall back to NVIDIA's CUDA-repo libnccl2 package |
| 3 | 156 GiB model load OOMs on one GPU due to uneven layer split | Medium | Medium | Default uniform `-ts`; if imbalanced, hand-tune `tensor_split[]` floats; or drop to 1024 ctx + q8_0 KV |
| 4 | LAYER split per-token latency is worse than expected (boundary copies dominate) | Medium | Low (informational) | Not a ship gate; P5.2 documents for SPRINT-026 row-split decision |
| 5 | TURBOMIND dispatch on one device but weights on another (cross-device pointer) crashes | Medium | High | P2.5 multi-device smoke catches before P3; per-device buft means each weight is locked to a device |
| 6 | `-ot` regex with multiple `CUDA_TURBOMIND<i>` qualifiers doesn't match the way I expect | Low-Medium | Low | P4.2 tests both regex forms; document the working one |
| 7 | `ncclCommInitAll` hangs (NCCL topology detection broken on V100 SXM2) | Low | High | P0.3 `nccl-tests` validates before any llama.cpp work; if hangs, set `NCCL_P2P_DISABLE=1` as fallback |
| 8 | DSv4-Flash-256e weights are corrupted on disk / wrong checksum | Low | High | `sha256sum` check at P3.1 vs the upstream-published checksum |
| 9 | Decode produces gibberish — quality regression specific to multi-GPU | Medium | High | P3.3 eyeball + P4.4 leading-token match; if fails, bisect TP layer split (run on 1 GPU partial → 2 → 4 → 8) |
| 10 | Pod can't get scheduled on `gpu-01` due to existing 1-GPU pod holding resources | Low-Medium | Low | Coordinate: stop the existing `llamacpp-build` pod before scheduling the 8-GPU one |
| 11 | Test of `ggml_backend_cuda_allreduce_tensor` reveals an upstream bug we didn't expect | Low | Medium | P1.5 standalone harness catches before model load; if buggy, file followup, ship LAYER-only (no allreduce in critical path) |

---

## 8. Security

No network-facing surface introduced. `llama-server` HTTP port is bound inside the pod and not exposed beyond the tailnet (per homelab convention). NCCL uses shared-memory + CUDA IPC for intra-node comms; no socket binding.

- Model file is local-only; checksum verified at P3.1.
- `libggml-turbomind.so` and `libnccl.so` are vendor binaries — same trust model as the existing CUDA / cuBLAS dependencies.
- No new secrets, credentials, or sandbox surface.
- No upstream PR or external upload; all work lands on `rapatel0/deepseek-v100-build` (per AGENTS.md "Private forks are exempt"; no AI-attributed contributions to upstream).

---

## 9. Dependencies

1. **Hardware**: `gpu-01` with 8× V100-SXM2-32GB; microk8s scheduler; the existing 1-GPU `llamacpp-build` pod is stopped or reconfigured before scheduling the 8-GPU pod.
2. **NCCL**: `libnccl2 >= 2.18` + `libnccl-dev`, ABI-compatible with CUDA 12.2.
3. **CUDA**: 12.2.2 (matching existing build image).
4. **llama.cpp branch**: `sprint-022-dsv4-integration` HEAD with SPRINT-023's CUDA_TURBOMIND buft already landed (`commit b01f69c04` per status).
5. **SPRINT-023 artifacts**: `libggml-turbomind.so` built and installable into the pod's runtime path (`/usr/local/lib/` or wherever the dlopen search picks it up). This is already part of the existing build.
6. **Model**: `DSv4-Flash-256e-fixed.gguf` (156 GiB) staged on cluster-accessible storage (NFS via `/srv/dev/...` per `homelab-k8s-dev` skill; OR `/models/` shared volume).
7. **Tools**: `llama-bench`, `llama-server`, `nccl-tests` (built from source in P0), `nsys` for profiling (already in the CUDA 12.2 dev image).
8. **NOT a dependency**: SPRINT-024. The per-expert legacy path works for SPRINT-025; if SPRINT-024 ships before P3 runs, we get grouped MoE for free and the bench numbers improve, but the success criteria don't change.

---

## 10. Open Questions

1. **Does `libggml-turbomind.so` carry an internal singleton `Gemm`?** Determines whether P2 stays a 1-day wrapper refactor or grows to include a library patch. **Answered by P0.3 multi-device init smoke.**
2. **`-ot` regex form for per-device TURBOMIND placement** — does `CUDA_TURBOMIND` (no index) match the device-owned buft automatically, or do we need explicit `CUDA_TURBOMIND0..CUDA_TURBOMIND7` with per-device layer ranges? **Answered by P4.2 testing.**
3. **2K vs 8K default context.** Intent allows either; 2K is comfortable with FP16 KV, 8K needs q8_0 KV. Recommend default 2K for the headline number to avoid confounding the bench with KV quant. **Decided at P3.2 — proposing 2K, FP16 KV.**
4. **Do we run the legacy 1-device perf comparison?** SPRINT-023 measured MIN-Ne at 16.6 t/s on 1 V100. Running 256e on 1 GPU isn't possible (doesn't fit), so there's no apples-to-apples 1-vs-8 comparison. The closest is: 16e on 1 GPU vs 256e on 8 GPU. **Recommend: skip — the experiments measure different things.**
5. **NCCL version pinning.** Pin to whatever Ubuntu's NVIDIA repo ships for CUDA 12.2 at the time of P0.2, or pull a specific version? Recommend pin to a known-good version, log the version in REPORT-19.
6. **Should P5 also bench the AVG-16e on 8 GPUs as a sanity-check baseline?** Would show the multi-GPU overhead vs the SPRINT-023 single-GPU 16e baseline. ~30 min of additional bench time. Recommend YES if P5 schedule allows.
7. **CUDA-graph capture (SPRINT-024-DEFERRED #11) under multi-GPU TP.** Probably broken or disabled by the cross-device boundaries; verify and document. Not a ship gate.

---

## 11. Outcome contract

This sprint ships if:

- The 156 GiB DSv4-Flash-256e model loads on 8 V100s with `-sm layer -ngl 999`.
- 32-token greedy decode produces coherent English on 10 fixed prompts.
- TURBOMIND `-ot` placement works on every device.
- `llama-bench` numbers and per-GPU memory accounting are captured in REPORT-19.

No hard TPS gate; the report numbers are the new baseline for SPRINT-026. If decode hangs, OOMs, or returns gibberish on any prompt, the sprint STOPs and we diagnose before re-attempting — the architectural fix is most likely in the per-device TmLib refactor (P2) or in cross-device weight pointer plumbing.

The reusable primitive — per-device CUDA_TURBOMIND + NCCL-linked build — survives regardless of SHIP/EXTEND/STOP.

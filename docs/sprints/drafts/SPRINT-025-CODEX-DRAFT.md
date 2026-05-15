# SPRINT-025 — DSv4-Flash-256e multi-GPU landing on `gpu-01`

**Status:** DRAFT 2026-05-15  
**Predecessor:** SPRINT-024 (grouped MoE on single V100)  
**Successor:** SPRINT-026 (multi-slot decode / speculative decode, unless row-TP slips from this sprint)

## Overview

SPRINT-025 lands the full `DSv4-Flash-256e-fixed.gguf` on the cluster node
`gpu-01` by scaling from the current single-V100 path to an **8x V100-SXM2-32GB**
pod. The sprint is explicitly **layer-split first**: get the full model to load,
decode coherently, and produce the first 8-GPU TPS numbers before chasing a more
aggressive tensor-parallel shape.

Three constraints define the plan:

1. **`4 x 32 GiB = 128 GiB` cannot hold a 156 GiB GGUF** before KV, scratch, or
   activations. A 4-GPU pod is a bring-up harness, not a full-model success target.
2. **`LLAMA_SPLIT_MODE_ROW` currently hard-fails for `deepseek4`** in
   `src/llama-model.cpp`, so the first operational milestone must use
   `LLAMA_SPLIT_MODE_LAYER`.
3. **`CUDA_TURBOMIND` is not truly multi-device yet**. The current
   `ggml-cuda-turbomind.cu` / `ggml/vendor/turbomind/api.cc` lifecycle is
   single-device state that re-initializes when the device changes. That is
   acceptable on one V100 and unsafe as a foundation for 8 concurrent GPUs.

The sprint therefore delivers two things in order:

- a reliable **8-GPU layer-split DSv4-256e landing**
- an explicit verdict on whether **row-split TP** belongs in this sprint or in
  the next one

## Use Cases

| Phase | Useful output even if the sprint stops here |
|---|---|
| P0 | 4-GPU and 8-GPU pod runbook, GPU visibility checks, and a clean statement that 4 GPUs are for bring-up only. |
| P1 | NCCL-enabled build artifact and a reproducible smoke path proving the container sees the requested GPU count and links `libnccl`. |
| P2 | Multi-device-safe `CUDA_TURBOMIND` init/upload path, or a concrete blocker proving why it cannot be used on 8 GPUs yet. |
| P3 | A 4-GPU harness that validates multi-GPU layer placement, device-local expert upload, and server decode on a smaller DSv4 variant. |
| P4 | Full 8-GPU load of `DSv4-Flash-256e-fixed.gguf` with coherent greedy decode and per-GPU memory accounting. |
| P5 | First end-to-end `llama-bench` numbers for the full model plus a REPORT-19 closeout. |
| P6 | Conditional row-split investigation: either a measured win, or an explicit defer with the exact technical blockers. |

## Architecture

### 1. Pod topology: 4 GPUs vs 8 GPUs

- **4-GPU pod**
  - Purpose: fast iteration, NCCL/link smoke, smaller-model decode, multi-device
    `CUDA_TURBOMIND` validation.
  - Not a success target for the full 256e model.
  - Why: `128 GiB < 156 GiB` even before KV cache or runtime overhead.

- **8-GPU pod**
  - Purpose: the actual DSv4-Flash-256e landing target.
  - Gross VRAM: `8 x 32 GiB = 256 GiB`.
  - Only configuration that can hold the full model fully offloaded on this node.

- **GPU visibility rule**
  - Keep the container view aligned with the pod request count. `ggml_cuda_init()`
    and `ncclCommInitAll()` enumerate visible devices, so the pod should see
    either `0..3` or `0..7`, not the full host fleet plus ad-hoc masking.
  - Success check: `nvidia-smi -L`, `cudaGetDeviceCount`, and ggml startup logs
    all report the same device count.

### 2. Split-mode strategy: `LAYER` ships, `ROW` is conditional

- **`LLAMA_SPLIT_MODE_LAYER` is the sprint default**
  - It already exists.
  - It is the default CLI mode.
  - It matches the current llama.cpp scheduler path for multi-GPU pipeline-style
    execution.
  - It does not require NCCL allreduce for the functional full-model landing.

- **`LLAMA_SPLIT_MODE_ROW` is not the mainline path for this sprint**
  - `src/llama-model.cpp` currently throws:
    `LLAMA_SPLIT_MODE_ROW not implemented for architecture 'deepseek4'`.
  - `make_gpu_buft_list()` only injects the CUDA split-buffer type for row mode.
  - `CUDA_TURBOMIND<i>` is a separate buffer type, so row-split TP and
    `CUDA_TURBOMIND` are not just a flag flip; they need an ownership story for
    “row-sharded expert weight on device-local turbomind dispatch”.

- **Recommendation**
  - Treat **`LAYER` as the success path** for SPRINT-025.
  - Treat **`ROW` as P6 conditional work** after the 8-GPU layer path is already
    operational.

### 3. NCCL build wiring

The good news is that the repo already has most of the wiring:

- `ggml/CMakeLists.txt` exposes `GGML_CUDA_NCCL`.
- `ggml/src/ggml-cuda/CMakeLists.txt` already does `find_package(NCCL)`.
- `ggml/src/ggml-cuda/ggml-cuda.cu` already calls `ncclCommInitAll()` and
  implements `ggml_backend_cuda_allreduce_tensor()` behind `GGML_USE_NCCL`.

The real sprint work is operational:

1. ensure the build image actually contains NCCL headers and libs
2. ensure the final binaries link them
3. ensure runtime only initializes comms for the GPUs exposed to the pod
4. ensure there is at least one reproducible smoke path that proves allreduce is
   live, even if the full 256e landing uses `-sm layer`

This means NCCL is a **build-and-runtime prerequisite** for future row TP, but
not a reason to block the first successful 8-GPU `layer` decode.

### 4. Per-GPU `CUDA_TURBOMIND` init is a required refactor

The current implementation is not truly per-GPU:

- `ggml_backend_cuda_turbomind_buffer_type(int device)` creates one buft per GPU,
  which is good.
- But `tm_ensure_loaded(device)` in `ggml-cuda-turbomind.cu` holds one global
  `TmLib` with one `init_device`.
- `ggml_turbomind_init(int cuda_device)` in `ggml/vendor/turbomind/api.cc`
  also owns one global runtime state and tears it down when the device changes.

That means an 8-GPU run can accidentally serialize initialization around a
single global device state. For SPRINT-025 this must become one of:

- **Option A — preferred**: per-device runtime state inside the turbomind API
  (`state[device]` or map keyed by CUDA ordinal), with idempotent init per GPU.
- **Option B**: one dlopen handle plus per-device opaque contexts returned from a
  versioned API. More invasive, cleaner long term.

Either way, SPRINT-025 should not rely on “re-init on device hop” remaining
correct once experts are uploaded and dispatched on multiple GPUs in one process.

### 5. Device-local expert placement needs more than `-ot exps=CUDA_TURBOMIND0`

`-ot` overrides currently bind a tensor regex to one exact buffer type name.
That works on one V100. It is not enough for 8 GPUs because:

- `CUDA_TURBOMIND0` pins matching tensors to GPU 0
- `CUDA_TURBOMIND1` pins them to GPU 1
- there is no current “use the `CUDA_TURBOMIND` buft that belongs to the layer’s
  assigned device” semantic

SPRINT-025 must choose one of two approaches:

- **Fast path**: generate per-layer overrides programmatically, the same way
  `llama_params_fit_impl()` already generates layer-specific override patterns.
- **Better path**: add a device-local `CUDA_TURBOMIND` family alias that resolves
  after layer placement.

Tradeoff:

- generated overrides are easier to land quickly
- but `llama-context.cpp` disables layer-mode pipeline parallelism when
  `model.has_tensor_overrides()` is true

So the sprint should prefer:

1. generated overrides if they are enough to get the full model to run
2. a family-alias follow-up only if losing pipeline parallelism makes the layer
   path unusable

### 6. VRAM budget for the 156 GiB model + KV

Use the following budgeting model, not the optimistic “256 minus 156 = 100 GiB
free, done” model.

- **Gross VRAM on 8 GPUs**: `256 GiB`
- **Model payload**: `156 GiB` planned budget
- **Average model share**: about `19.5 GiB/GPU` if the layer split is close to
  even

Reserve explicit headroom:

- **CUDA / allocator / turbomind scratch / fragmentation**:
  plan `2-3 GiB/GPU` = `16-24 GiB` total
- **decode workspace, graph pools, transient activations, request overhead**:
  plan another `1-2 GiB/GPU` = `8-16 GiB` total

That leaves **roughly `60-76 GiB` cluster-wide for KV**, not the full 100 GiB.

Practical policy for the first landing:

- default to **single slot**
- default to **moderate context** (`4k` or `8k`, not training max context)
- default to **`--cache-type-k q8_0 --cache-type-v q8_0`**

Rationale:

- `q8_0` KV is not required to make the 8-GPU model fit in principle
- it *is* the right default to protect first-pass headroom against allocator
  variance, fragmentation, and the still-unknown cost of multi-device turbomind
  workspace

For the sprint report, the memory target should be:

- **after model load, before decode**: no GPU above `~28 GiB`
- **during 32-token decode**: no GPU above `~30 GiB`

If either threshold is exceeded, stop and rebalance before claiming success.

## Implementation

### P0 — Pod variants, visibility, and topology capture

**Goal:** make the hardware plan explicit and eliminate ambiguity around “4 GPUs
vs 8 GPUs”.

1. Document a **4-GPU pod** and an **8-GPU pod** for `gpu-01`.
2. Record GPU visibility from inside each pod:
   - `nvidia-smi -L`
   - `nvidia-smi topo -m`
   - ggml startup log device count
3. Verify both pods mount:
   - the repo workspace
   - `/models`
   - enough shared memory for build + runtime
4. Capture the rule that **4 GPUs are not a full-model target** and keep that in
   the runbook so this does not get re-litigated mid-sprint.

**P0 Gate**

- 4-GPU pod runs and sees 4 GPUs
- 8-GPU pod runs and sees 8 GPUs
- `gpu-01` topology is captured once for the sprint
- runbook states unambiguously that full 256e requires 8 GPUs

### P1 — NCCL-enabled build and smoke verification

**Goal:** make NCCL a real linked dependency, not a CMake option that silently
falls back off.

1. Add NCCL runtime + dev packages to the build image used by `llamacpp-build`.
2. Build with:
   - `-DGGML_CUDA=ON`
   - `-DGGML_CUDA_NCCL=ON`
   - existing sm70 / CUDA 12.2 settings
3. Verify:
   - CMake reports `NCCL_FOUND`
   - `ldd` on `llama-server` / `llama-bench` shows `libnccl`
   - `NCCL_DEBUG=INFO` emits communicator init on a smoke run
4. Exercise one allreduce-capable smoke path:
   - either a small row-capable model
   - or a tiny backend-level CUDA smoke if `deepseek4` row mode is still blocked

**P1 Gate**

- build links NCCL, not a warning-only fallback
- runtime communicator init is visible in logs
- at least one allreduce smoke path completes successfully

### P2 — Multi-device-safe `CUDA_TURBOMIND`

**Goal:** remove the single-device runtime assumption before attempting the
8-GPU model.

1. Refactor the turbomind runtime lifecycle so device 0 init does not tear down
   device 1 state.
2. Ensure upload-time packing is device-local and can occur on multiple GPUs in
   one process.
3. Add a direct smoke:
   - pack one supported tensor on GPU 0
   - pack another on GPU 1
   - dispatch both without shutdown/re-init churn
4. Decide the placement mechanism:
   - generated per-layer overrides now
   - or device-local `CUDA_TURBOMIND` alias now

**P2 Gate**

- no global “current turbomind device” assumption remains on the hot path
- two-device pack + dispatch works in one process
- placement strategy for multi-GPU expert tensors is chosen and documented

### P3 — 4-GPU harness with a smaller DSv4 model

**Goal:** validate the multi-GPU machinery on a configuration that fits.

1. Use a smaller DSv4 variant that fits comfortably on 4 GPUs.
2. Run `-sm layer` first.
3. Verify:
   - memory is spread across all 4 visible devices
   - expert tensors route to device-local `CUDA_TURBOMIND<i>`
   - server decode completes
4. If using generated per-layer overrides, confirm whether pipeline parallel is
   disabled and record the measured cost.

**P3 Gate**

- smaller DSv4 model decodes on 4 GPUs with `layer`
- per-device expert placement is visible in logs / memory breakdown
- no invalid-device or cross-device pack errors appear

### P4 — Full 8-GPU `DSv4-Flash-256e` landing with `LLAMA_SPLIT_MODE_LAYER`

**Goal:** first successful end-to-end run of the full model.

1. Launch the 8-GPU pod.
2. Start with:
   - `-sm layer`
   - `-ngl 999`
   - single slot
   - `q8_0` KV
   - conservative context
3. Load `DSv4-Flash-256e-fixed.gguf`.
4. Verify:
   - model load completes
   - per-GPU memory remains below the headroom thresholds
   - 32-token greedy decode completes on a fixed prompt set
   - output is coherent English, not crash-garbled output
5. Capture `nvidia-smi dmon` or equivalent device activity during decode.

**P4 Gate**

- full 256e model loads on 8 GPUs
- 32-token greedy decode succeeds
- memory accounting stays inside the planned budget
- all 8 GPUs show participation during the decode window

### P5 — Measurement and REPORT-19

**Goal:** establish the first headline numbers for the full model on this stack.

1. Run `llama-bench -p 128 -n 32 -r 3` on the 8-GPU layer path.
2. Report:
   - prefill TPS
   - decode TPS
   - per-GPU VRAM after load and during decode
   - whether generated overrides disabled pipeline parallel
3. Keep 4-GPU numbers only as diagnostic bring-up data, not as the headline.
4. Write REPORT-19 and update sprint memory notes.

**P5 Gate**

- full-model TPS captured and written down
- memory tables exist
- operational command line is reproducible from the report

### P6 — Conditional `LLAMA_SPLIT_MODE_ROW` investigation

**Goal:** decide whether row TP belongs in this sprint or should be the next one.

1. Remove or relax the `deepseek4` row-mode guard only after P4/P5 are green.
2. Re-evaluate the architectural mismatch:
   - row split uses split buffers
   - turbomind dispatch keys off `CUDA_TURBOMIND`
3. Choose one of:
   - row split only for dense weights
   - new split-capable turbomind buffer semantics
   - explicit defer to SPRINT-026 because this is a separate architecture step
4. If row mode reaches runnable state, compare it against the 8-GPU layer path.

**P6 Gate**

- either row mode runs and is measured
- or the sprint closes with a precise defer note naming the blocker:
  `deepseek4 guard`, buffer-type mismatch, correctness risk, or no speedup

## Files Summary

### In-repo code surfaces likely involved

- `ggml/src/ggml-cuda/CMakeLists.txt`
  - NCCL link verification path
- `ggml/cmake/FindNCCL.cmake`
  - fallback / discovery behavior if the image packaging is incomplete
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - communicator init and allreduce runtime path
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu`
  - multi-device runtime lifecycle, upload, and dispatch glue
- `ggml/vendor/turbomind/include/ggml-turbomind-api.h`
  - API version / lifecycle contract if per-device state changes
- `ggml/vendor/turbomind/api.cc`
  - turbomind runtime state ownership and workspace allocation
- `src/llama-model.cpp`
  - current `deepseek4` row-mode guard
- `src/llama-model-loader.cpp`
  - tensor override application and placement
- `src/llama.cpp`
  - programmatic per-layer tensor override generation if that path is chosen
- `src/llama-context.cpp`
  - pipeline-parallel interaction with tensor overrides

### In-repo docs / artifacts

- `docs/sprints/SPRINT-025.md` or `docs/sprints/REPORT-19.md`
  - final sprint closeout and measurements
- `docs/sprints/SPRINT-025-P0-topology.md` or similar
  - one-time topology / runbook capture if the team wants a separate artifact

### Out-of-repo but operationally required

- the homelab manifests repo entry that defines the 4-GPU and 8-GPU pod specs
- the build image definition used by `llamacpp-build`

## Definition of Done

1. A **4-GPU pod** exists as a bring-up harness and is documented as such.
2. An **8-GPU pod** exists and is the configuration used for the full-model run.
3. The build links **NCCL** successfully and a smoke path proves communicator
   init / allreduce are actually live.
4. `CUDA_TURBOMIND` no longer depends on one mutable global “current device”
   state for multi-GPU correctness.
5. `DSv4-Flash-256e-fixed.gguf` loads fully on **8 V100s** with
   `LLAMA_SPLIT_MODE_LAYER`.
6. Greedy decode of 32 tokens completes with coherent output.
7. Per-GPU VRAM is reported at load time and during decode, and stays within the
   planned budget.
8. `llama-bench -p 128 -n 32 -r 3` numbers are recorded for the full 8-GPU path.
9. The sprint closes with an explicit **`LAYER` vs `ROW` decision**:
   - `ROW` measured and accepted, or
   - `ROW` deferred with the exact blocker named

## Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Treating 4 GPUs as a full-model target | Wastes sprint time on an impossible fit | Make the 4-GPU role explicit in P0 and DoD. |
| Single-device turbomind runtime state | Can corrupt or serialize multi-GPU upload/dispatch | Make per-device init a P2 gate, not a hidden assumption. |
| Exact-name `-ot` overrides pin experts to one GPU | Breaks device-local placement for layer-split multi-GPU | Choose generated per-layer overrides or a device-local alias in P2. |
| Tensor overrides disable pipeline parallel | Could reduce the value of the layer path | Measure it in P3/P5 and only build a better alias if needed. |
| `ROW` mode is a second architecture problem, not a toggle | Could expand scope late in the sprint | Make `ROW` conditional P6 work only after P4/P5 are done. |
| NCCL present at build, absent at runtime | Produces false confidence | Require `ldd` plus runtime log proof in P1. |
| Shared-node contention on `gpu-01` | Can distort both memory headroom and TPS | Keep pod placement fixed to `gpu-01` and capture topology / usage during runs. |

## Security

- No new public-facing service is required for the sprint itself; all work stays
  inside the existing cluster and pod boundary.
- Keep the pod least-privileged: request GPUs and shared memory, but do not add
  broader host access than the existing build/run workflow needs.
- NCCL is single-node in this sprint; do not open or depend on cross-node
  communication paths.
- Model artifacts and measurement logs stay under the existing workspace and
  `/models` mount; no new credential surface should be introduced.

## Dependencies

1. SPRINT-023 / SPRINT-024 code paths:
   `CUDA_TURBOMIND`, grouped MoE dispatch, existing single-V100 DSv4 support.
2. `gpu-01` with **8x V100-SXM2-32GB** visible to the target pod.
3. Build environment locked to the current known-good stack:
   CUDA 12.2, gcc 11.4, cmake 3.22.
4. NCCL headers and libs available inside the build image and runtime pod.
5. `DSv4-Flash-256e-fixed.gguf` available from the mounted model path.
6. Homelab manifest changes for 4-GPU / 8-GPU pod requests.

## Open Questions

1. Should the first 8-GPU landing default to `4k` or `8k` context, given the
   unknown multi-device workspace cost?
2. Is it better to ship generated per-layer overrides now, or spend extra time
   preserving pipeline parallel with a device-local `CUDA_TURBOMIND` alias?
3. Does `gpu-01` present a topology where row TP is likely to beat layer split,
   or should `ROW` already be treated as a follow-on sprint unless `LAYER`
   underperforms badly?
4. Does SPRINT-024 need to land first for acceptable performance, or is
   functional 8-GPU `layer` good enough to ship before grouped-path perf work is
   fully closed?
5. If `ROW` remains blocked, should SPRINT-026 focus on multi-slot decode over
   the 8-GPU layer path rather than on row-TP at all?

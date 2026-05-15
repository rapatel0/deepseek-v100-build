# SPRINT-023 — Turbomind MXFP4 expert dispatch (V100 DSv4-Flash, MXFP4-first)

**Sprint number**: 023
**Date**: 2026-05-15
**Author**: Claude (independent draft)
**Predecessor**: SPRINT-022 (operational baseline)
**Successor candidates**: SPRINT-024 (F8_E4M3_B128 dense port, multi-GPU)

---

## 1. Overview

SPRINT-022 shipped an operational DSv4-Flash baseline on V100 (pp128 = 4.28
t/s, tg32 = 4.73 t/s) by forcing the 256-expert MoE stack to CPU
(`-ot exps=CPU`). The pp/tg ratio of ~0.9 is the diagnostic: every forward
pass is gated on host-RAM expert FFNs.

SPRINT-021 P0 (REPORT-15) measured the V100 sm70 ceilings turbomind's packed
mainloops actually achieve at production MoE shapes (M=2048 N=K=7168):

| Config            | TFLOPS | HMMA active% | vs v12 INT8 |
|-------------------|-------:|-------------:|------------:|
| v12_ms3 INT8      | 38.98  | 31.72%       | baseline    |
| `Config_MXF4`     | (~50+, group_size=32) | n/a* | +30%+    |
| `Config_E4M3`     | 59.07  | 55.70%       | +52%        |
| `Config_U4_g`     | 64.67  | 60.94%       | +66%        |

*MXF4 not run-measured in SPRINT-021 P0 because we used `group_size=128` in
the harness; registry only carries `group_size=32` entries for sm70. The
DSv4-Flash GGUF uses `QK_MXFP4=32`, so the production path actually matches
the registered tiles. P0 of this sprint re-runs MXF4 with the corrected
group size.

This sprint moves the **MXFP4 MoE experts** off the CPU and dispatches them
through turbomind's `Config_MXF4` mainloop. We scope SPRINT-023 to
**MXFP4 only** — the dense F8_E4M3_B128 layers stay on their existing
nisparks scalar path. Two reasons for the carve-out:

1. **Where the time goes**: the 256 MoE experts dominate the per-token cost
   in DSv4-Flash. The dense path (~5% of params) is not the bottleneck at
   tg32 = 4.73 t/s. Moving experts alone closes most of the gap.
2. **Scope discipline**: each turbomind dispatch path has a non-trivial
   weight-layout converter (`GetConverters()` + `Convert()` + `extend_to_u16`
   + transpose + per-config scale layout). Wiring two simultaneously
   doubles the surface area where layout bugs hide. MXFP4 first, F8 in
   SPRINT-024 once the dispatch shim is proven.

Success criterion that matters: **tg32 ≥ 20 t/s** at 32 GiB VRAM budget,
no correctness regression vs the SPRINT-022 baseline, and a build-flag
fallback that runs identical to baseline when off.

---

## 2. Use Cases (what users gain when this sprint ships)

- **DSv4-Flash inference at conversational latency on a single V100.** The
  SPRINT-022 4.73 t/s decode is below the "live typing" threshold most
  users notice as fluent (~15 t/s). 20 t/s puts us in the same band as
  Llama-70B Q4 on the same hardware, with a 284B-param model.
- **Reproducible recipe for V100-class hardware in the homelab**: the
  build flag `GGML_TURBOMIND_GEMM=ON` flips on the same code path that
  works on gpu-01. No CUDA-12.6+ requirement, no Hopper, no Ada. The same
  binary still runs (slower) on T4 / RTX 30-series with the flag off.
- **Off-ramp for the v12/v13 INT8 line.** REPORT-15 §2 established
  that turbomind FP4/FP8 ceilings dominate INT8 dequant on V100. Shipping
  the turbomind path here lets us deprecate the tc-grid INT8 line as a
  research artifact while keeping its docs as a learning record.
- **Headroom for SPRINT-024**: with MoE experts on GPU and a working
  dispatch shim, the F8_E4M3 dense port becomes a one-config-add change.

Non-goal use cases (explicitly deferred):
- Multi-GPU TP (SPRINT-024+)
- sm75/80/90 portability — the carve-out includes sm70-only registry
- Dynamic hot-expert promotion at runtime — load-time JSON profile only

---

## 3. Architecture

### 3.1 Where turbomind plugs in

llama.cpp's CUDA backend has two GEMM dispatch entry points for quantized
weights:

- `ggml/src/ggml-cuda/mmvq.cu` — mat-vec quantized (decode path; M=1..few).
  The MoE path enters here via `mul_mat_vec_q_switch_ncols_dst` with the
  `ids` tensor steering per-token expert selection.
- `ggml/src/ggml-cuda/mmq.cu` — mat-mul quantized (prefill path; M≥128).
  Same dispatch by `args.type_x`; MoE goes through
  `ggml_cuda_mul_mat_q_switch_type` (lines 6–72) with `ids_dst` +
  `expert_bounds` already computed by `ggml_cuda_launch_mm_ids_helper`.

Both files already have `case GGML_TYPE_MXFP4:` (mmvq.cu:947, mmq.cu:23).
Today these dispatch to nisparks's scalar `vec_dot_*_q8_1` paths. We
intercept **before** those switch cases and route to turbomind when:

1. Build flag `GGML_TURBOMIND_GEMM=ON` was set at CMake configure time
2. The src0 tensor was uploaded via the turbomind buffer type (more in
   §3.2) — detectable via `ggml_backend_buffer_get_type(src0->buffer)` ==
   `ggml_backend_cuda_turbomind_buffer_type()`
3. We're on sm70 (cc == 70). Other arches keep the scalar path even with
   the flag on — turbomind's `Config_MXF4` is registered only in
   `arch/sm70_884_8.cu`.

Concretely, the dispatch shim lives in a new file
`ggml/src/ggml-cuda/turbomind_gemm.cu` exposing:

```cpp
// Returns true if dispatch happened; caller falls through on false.
bool ggml_cuda_turbomind_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,   // packed MXFP4 weights (turbomind layout)
    const ggml_tensor * src1,   // FP16 / FP32 activations
    const ggml_tensor * ids,    // expert-selection ids, may be null
    ggml_tensor       * dst,
    cudaStream_t        stream);
```

Called from the top of both `ggml_cuda_mul_mat_q` (mmq.cu:74) and the
mmvq entry (mmvq.cu's `ggml_cuda_mul_mat_vec_q`), guarded by a single
predicate function `ggml_cuda_turbomind_can_dispatch(src0, ids)` defined
in `turbomind_gemm.cuh`.

### 3.2 Weight upload pipeline

The hard architectural decision: **where does packing happen?**

The intent doc names three options. We choose **option (c): a dedicated
CUDA buffer type** that performs packing at `init_tensor` time. Reasons:

- **Per-tensor opt-in via `-ot`.** Users already select where MoE weights
  live with `--override-tensor` (`common/arg.cpp:2279`). The natural UX
  is `--override-tensor 'exps=CUDA0_TURBOMIND'`. No new CLI surface, no
  hot-expert JSON wiring for the MVP — the user picks which layers (or
  which expert pattern) goes into the turbomind buffer.
- **Avoids `convert.cu` bloat.** `ggml/src/ggml-cuda/convert.cu` is the
  per-row dequant catalog (Q4_0 → F16, etc.). Inserting layout
  conversion there mixes concepts and is shared across every backend
  consumer of MXFP4. The buffer-type pattern keeps turbomind layout
  conversion isolated.
- **Bit-identical to GGUF source.** The mmap-loaded GGUF stays canonical.
  Packing reads from CPU staging memory once, writes to VRAM in
  turbomind's layout once. No in-place mutation.
- **Drop-in for existing `-ot exps=CPU` users.** Replacing the rhs with
  `CUDA0_TURBOMIND` is a one-flag flip; same regex parser.

The buffer type registers as a peer of `ggml_backend_cuda_buffer_type(0)`
in `ggml/src/ggml-cuda/ggml-cuda.cu`. Implementation:

```cpp
// ggml/src/ggml-cuda/turbomind_buffer.cu
struct ggml_backend_cuda_turbomind_buffer_context : public ggml_backend_cuda_buffer_context {
    // adds: turbomind::gemm::MatrixLayout per-tensor (k_desc, q_desc)
    //       turbomind scale buffer (cudaMalloc'd alongside weight buffer)
    std::unordered_map<const ggml_tensor*, turbomind_packed_meta> meta;
};

// Called once per tensor at model load.
static void init_tensor_turbomind(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    // Only MXFP4 (and later F8_E4M3_B128) get packed; other types passthrough.
    if (tensor->type != GGML_TYPE_MXFP4) {
        // Forward to base CUDA buffer init.
        return ggml_backend_cuda_buffer_init_tensor(buffer, tensor);
    }
    // Mirror gemm_bench_packed.cu's build_packed_weight():
    //   1. cudaMalloc the canonical MXFP4 staging area sized like the GGUF tensor
    //   2. memcpy the GGUF block bytes there
    //   3. extend_to_u16 → transpose if conv_w->order == kRowMajor
    //   4. conv_w->Convert(tmp, w_desc, packed, k_desc, stream)
    //   5. conv_s->Convert(scale_raw, s_desc, packed_s, q_desc, stream)
    //   6. Stash {k_desc, q_desc, scales_ptr} in buffer.meta[tensor]
    //   7. Free staging
}
```

For each MXFP4 tensor we hold **one** GPU buffer (the packed weight) and
**one** scale buffer. The canonical GGUF bytes are freed after packing —
we don't keep two copies on GPU. The CPU mmap stays canonical and is the
source of truth for re-init / model reload.

GGUF MXFP4 block layout (`QK_MXFP4=32`, 17 bytes/block: 1 E8M0 scale + 16
nibble-packed weights) maps to turbomind's `Operand_B_Pack<fp4_e2m1_t>` +
`Operand_V_Pack<uint8_t>` (kUint8 scale type per
`gemm_bench_packed.cu::scale_dtype`). The byte-for-byte correspondence
is:
- Weights: GGUF nibble pairs become turbomind sub-byte `uint4_t` view
  after `extend_to_u16` widens to a temporary u16 buffer.
- Scales: GGUF E8M0 (8-bit exponent) is already in turbomind's expected
  `kUint8` format. Direct passthrough through the scale converter.

### 3.3 Dispatch decision tree

```
ggml_cuda_mul_mat_q(src0, src1, ids, dst):
    if !GGML_TURBOMIND_GEMM_ENABLED:                  # CMake flag, compile-time
        goto scalar_dispatch
    if !ggml_cuda_turbomind_can_dispatch(src0, ids):  # see predicate below
        goto scalar_dispatch
    return ggml_cuda_turbomind_mul_mat(ctx, src0, src1, ids, dst, stream)

scalar_dispatch:
    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream)   # existing path


ggml_cuda_turbomind_can_dispatch(src0, ids):
    return cc == 70
        && src0->type == GGML_TYPE_MXFP4
        && buffer_type(src0) == ggml_backend_cuda_turbomind_buffer_type()
        && (ids == nullptr || ids->type == GGML_TYPE_I32)
        && src0->ne[0] % 32 == 0    # K divisible by group_size
        && src0->ne[1] % 8  == 0    # turbomind tile alignment
```

Inside `ggml_cuda_turbomind_mul_mat`:

- **No-ids path (dense MXFP4 mul_mat)**: straight `tmg::Gemm::Run` with
  `M = ne11 * ne12 * ne13`, `N = ne01`, `K = ne00`. The activation gets
  cast from F32 → F16 via existing `convert_unary_cuda<float, half>`
  (already in convert.cu) into a pool-allocated F16 buffer; output gets
  cast back F16 → F32.
- **MoE-ids path**: iterate the `expert_bounds` array (already computed
  by `ggml_cuda_launch_mm_ids_helper` upstream in mmq.cu:180). For each
  expert with `bounds[e+1] - bounds[e] > 0`, launch one `tmg::Gemm::Run`
  with the per-expert slice. The 8-active-experts-per-token design
  means at M=2048 prefill we average ~64 tokens/expert; at tg32 decode
  we have M=8. Both shapes match registered tile sizes (BM ∈ {8,16,32,64,128}).
- **Workspace reuse**: `tmg::Gemm::kBarriersSize` + a `partials` pool
  buffer scaled to `max_M * max_N * 4 * sizeof(float)`. Allocated once
  per context, kept across calls (the `ggml_cuda_pool` already provides
  this lifetime model).

---

## 4. Implementation (phased)

Six phases. Each phase has a single ship gate. P0/P1/P2 are mandatory for
correctness; P3/P4/P5/P6 are the performance lift. If we slip schedule,
we ship P0–P3 (the turbomind shim landing correct but routed via `-ot`,
no measured TPS gain yet) and pull P4–P6 into SPRINT-024.

### P0 — Recovery: MXF4 ceiling at production group_size

**Goal**: measure the `Config_MXF4` ceiling at `group_size=32` to confirm
the perf assumption underlying this sprint. SPRINT-021 P0 only ran MXF4
at gs=128 (no registry match → skipped).

- Re-run `gemm_bench_packed.cu --configs fp4 --group-size 32` at the
  DSv4 catalog shapes (M ∈ {1, 8, 128, 2048}, N=K=7168 plus 7168×18944
  asymmetric).
- Add `tools/tc-grid/docs/REPORT-15-mxf4-gs32.csv` and a 1-paragraph
  appendix to REPORT-15.
- **Ship gate**: MXF4 at M=2048 N=K=7168 ≥ 45 TF. If it doesn't clear
  that, this sprint's TPS target is at risk and we should fall back to
  Config_U4_g (would need a quant-conversion utility — that's a much
  bigger SPRINT-024 change).

Files:
- `tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu` (existing, no
  edits — already supports `--group-size 32 --configs fp4`)
- `tools/tc-grid/docs/REPORT-15.md` (append §7)
- `tools/tc-grid/docs/REPORT-15-mxf4-gs32.csv` (new)

### P1 — Build glue: GGML_TURBOMIND_GEMM CMake option

**Decision: CMake flag, not env var.** Reasoning:

- The turbomind carve-out adds ~30 turbomind .cu files plus fmt and
  cutlass to the link line. An env var means we always pay that
  compile + link cost on V100 builds and ship a heavier binary to users
  who don't want it. The flag lets distros opt in.
- Symbol visibility: turbomind's `tmg::Gemm` is a heavyweight C++
  symbol with constructors that touch CUDA globals. A compile-time
  guard keeps unrelated builds clean.
- We still keep a **runtime** `LLAMA_DISABLE_TURBOMIND=1` env var that
  forces the predicate `ggml_cuda_turbomind_can_dispatch` to return
  false even on a `=ON` build. This is the kill-switch for production
  rollouts when correctness regresses.

Concrete work:
- Add `option(GGML_TURBOMIND_GEMM "Enable turbomind sm70 GEMM dispatch" OFF)`
  to `ggml/src/ggml-cuda/CMakeLists.txt`.
- When ON, do the same FetchContent dance as
  `tools/tc-grid/turbomind_minimal/CMakeLists.txt`:
  - fmt (header-only, 11.0.2)
  - turbomind/utils, turbomind/core, turbomind/kernels/gemm
  - CUTLASS alias `nvidia::cutlass::cutlass`
  - `--expt-relaxed-constexpr --expt-extended-lambda -include cuda_bf16.h`
- Link the resulting `gemm2` + `core` + `utils` targets into the
  `ggml-cuda` library.
- Path to lmdeploy src: same convention as tc-grid —
  `cmake -DLMDEPLOY_SRC=/path/to/research/lmdeploy/src`. Default to the
  in-tree `research/lmdeploy/src` if it exists.

**Ship gate**: `cmake -DGGML_TURBOMIND_GEMM=ON ../` configures, builds
clean on V100 + CUDA 12.2 + gcc 11.4. Resulting binary has both
`-DGGML_TURBOMIND_GEMM=ON` and `=OFF` test runs of the SPRINT-022
baseline `llama-bench` invocation producing identical numbers (the OFF
path is unchanged; the ON path without an `-ot turbomind` override
trivially also unchanged because no buffer ever gets the turbomind type).

Files:
- `ggml/src/ggml-cuda/CMakeLists.txt` (modify)
- `ggml/src/ggml-cuda/turbomind_glue.cmake` (new — encapsulate the
  FetchContent + add_subdirectory)
- `CMakeLists.txt` (top-level — surface the option for visibility)

### P2 — Turbomind buffer type registration

**Goal**: a working `ggml_backend_cuda_turbomind_buffer_type()` that
the `-ot` parser can target as a string, identical to the existing
`CUDA0` buffer type registration pattern.

- Implement `ggml_backend_cuda_turbomind_buffer_type()` in a new file
  `ggml/src/ggml-cuda/turbomind_buffer.cu`.
- Mirror `ggml-cuda.cu`'s buffer-type registration: name returned is
  `"CUDA0_TURBOMIND"` (V100 is device 0; for multi-device we'd
  parameterize but that's SPRINT-024+).
- The buffer **wraps** a normal CUDA buffer for tensors that aren't
  MXFP4, so non-expert tensors landing on this buffer (e.g. a
  `ffn_gate_exps` that the user pattern-matched) still work.
- For MXFP4 tensors: implement `init_tensor` as the packing routine
  from §3.2. Store per-tensor metadata in a side `unordered_map<const
  ggml_tensor*, turbomind_packed_meta>` keyed off tensor address.
- Register the buffer type in `ggml_backend_cuda_reg()` so it's
  discoverable via the public `ggml_backend_dev_buffer_type` API.

**Ship gate**: a small standalone test
`tests/test-backend-ops-turbomind.cu` (new) that:
1. Allocates a synthetic 7168×7168 MXFP4 tensor on
   `CUDA0_TURBOMIND` buffer type
2. Confirms `init_tensor` runs and stores meta
3. Confirms `get_tensor` reads back the packed bytes (round-trip not
   byte-identical to source — that's the conversion — but
   re-running the converter on the same source yields the same packed
   output)

Files:
- `ggml/src/ggml-cuda/turbomind_buffer.cu` (new)
- `ggml/src/ggml-cuda/turbomind_buffer.cuh` (new)
- `ggml/src/ggml-cuda/ggml-cuda.cu` (modify — register buffer type)
- `tests/test-backend-ops-turbomind.cu` (new — gated by
  `GGML_TURBOMIND_GEMM`)

### P3 — Dispatch shim (no-ids path first)

**Goal**: route `ggml_cuda_mul_mat_q` for MXFP4 tensors on the turbomind
buffer through `tmg::Gemm::Run`, no MoE ids handling yet.

- Implement `ggml_cuda_turbomind_mul_mat` in
  `ggml/src/ggml-cuda/turbomind_gemm.cu`.
- Reuse the activation-cast logic: pull `convert_unary_cuda<float,
  half>` for input cast (allocate F16 staging via
  `ggml_cuda_pool_alloc<half>`); cast output back F16 → F32 the same
  way.
- Compose Adesc/Bdesc/Vdesc/Ddesc from the stored meta + the tensor
  shape (`src0->ne[0]` = K, `src0->ne[1]` = N, etc.).
- Wire the predicate into the top of `ggml_cuda_mul_mat_q` (mmq.cu:75)
  and `ggml_cuda_mul_mat_vec_q` (mmvq.cu). Falls through to scalar on
  any reason (cc != 70, type != MXFP4, buffer not turbomind, M/N/K
  alignment fail).
- **Correctness gate**: a new `tools/tc-grid/turbomind_minimal/test_dispatch.cu`
  that bit-compares turbomind MXFP4 output vs the existing nisparks
  scalar `vec_dot_mxfp4_q8_1` path for the same weight/activation pair.
  Tolerance: rel ≤ 2e-2 (per SPRINT-015 P2 col-parallel contract).
- 5 random (N, K) at M ∈ {1, 128} = 10 cases. All must pass.

**Ship gate**: dense MXFP4 mul_mat correctness test passes; SPRINT-022
baseline `llama-bench` with `-DGGML_TURBOMIND_GEMM=ON` but **no**
`-ot` override still matches OFF numbers.

Files:
- `ggml/src/ggml-cuda/turbomind_gemm.cu` (new)
- `ggml/src/ggml-cuda/turbomind_gemm.cuh` (new)
- `ggml/src/ggml-cuda/mmq.cu` (insert predicate at line ~84, before
  `const int cc =`)
- `ggml/src/ggml-cuda/mmvq.cu` (insert predicate at the equivalent
  entry — `ggml_cuda_mul_mat_vec_q`)
- `tools/tc-grid/turbomind_minimal/test_dispatch.cu` (new)

### P4 — MoE-ids path (the lift)

**Goal**: handle `ids != nullptr` (MoE expert selection). This is the
hot path for DSv4-Flash.

The existing mmq.cu MoE flow (lines 167–220) does this work upstream of
the type switch:
1. `ggml_cuda_launch_mm_ids_helper` produces `ids_src1`, `ids_dst`,
   `expert_bounds` (cumulative per-expert token counts).
2. The args struct gets fed to `mul_mat_q_case<TYPE>` which has its own
   per-expert tile iteration.

For turbomind we replace step 2 with: for each expert `e` in
`[0, ne02)`, take the rows `[expert_bounds[e], expert_bounds[e+1])`
of the gathered activation, the expert-`e` slice of `src0`
(stride `s02`), and call `tmg::Gemm::Run` with `M = bounds[e+1] -
bounds[e]`, `N = ne01`, `K = ne00`.

Per-expert dispatch trades CUDA-stream parallelism for kernel
launch overhead. At 256 experts × 8 active/token × 32 tokens we get up
to 256 launches per forward pass. We measure two strategies in P5:

- **Sequential per-expert**: simplest, no stream juggling. Estimated
  overhead 5–10 us/launch × ~200 active experts/token = 1–2 ms/token.
  At 20 t/s target = 50 ms/token total budget. 4% overhead.
- **Stream-batched**: 4–8 CUDA streams, round-robin across experts.
  Reduces wall-clock by ~3× if launches don't serialize on a single SM.
  Risk: workspace buffer sharing.

Ship strategy 1 first. Strategy 2 = stretch in P6.

**Ship gate**: end-to-end `llama-bench` with
`-ot exps=CUDA0_TURBOMIND` produces correct generation (text output
matches `-ot exps=CPU` baseline for the first 32 generated tokens,
greedy sampling, seed-pinned). tg32 ≥ baseline 4.73 t/s. No
correctness regression vs SPRINT-022.

Files:
- `ggml/src/ggml-cuda/turbomind_gemm.cu` (extend with MoE path)
- `ggml/src/ggml-cuda/mmq.cu` (the predicate already in P3 routes ids
  case — confirm no special-casing needed)

### P5 — VRAM budget validation + selective expert placement

**Goal**: fit the working set in <30 GiB. DSv4-Flash full MoE is 145
GiB; we only have ~22.5 GiB free per the intent doc.

**Decision on hot-expert selection: synthetic uniform first, JSON
profile in SPRINT-024.** Reasoning:

- The SPRINT-022 baseline already proved CPU-MoE works. Our scope
  question is just "which N% of experts get to live on GPU."
- A JSON expert-usage profile requires:
  1. A first profiling pass producing the JSON (cost: a few minutes
     of CPU-only runtime on a sample prompt set)
  2. JSON schema design (per-layer, per-expert hit count)
  3. Loader hook to consume it and steer `-ot` patterns
  
  That's a half-sprint of work on its own. Punt to SPRINT-024.
- For SPRINT-023, ship two simple modes selectable by `-ot` regex:
  - `--override-tensor 'blk\.([0-9]|1[0-5])\..*exps=CUDA0_TURBOMIND'`
    — first 16 layers' experts on GPU, rest on CPU. Empirically MoE
    routing has more entropy in early layers; this is a defensible
    default.
  - `--override-tensor 'exps=CUDA0_TURBOMIND'` — all experts on GPU
    (only fits if user has >145 GiB VRAM, which we don't on V100;
    tested as the "what if" case for SPRINT-024 multi-GPU).
- VRAM probe: `tools/tc-grid/scripts/measure_vram_after_load.sh` (new)
  runs llama-bench with each `-ot` pattern, captures `nvidia-smi --query-gpu=memory.used`
  after model load and after 32-token decode, validates <30 GiB.

The "first 16 layers" mode is the production recipe for this sprint.
Approximate VRAM footprint:
- 16 layers × 256 experts × (gate + up + down) × 7168 × 2048 × 0.5 B/wt
  (MXFP4) = ~12 GiB raw weights + ~1 GiB scales + 4 GiB existing dense
  = ~17 GiB. Comfortably under 30 GiB.

**Ship gate**: `nvidia-smi` after `llama-bench` decode shows <30 GiB;
the run completes 32-token decode without OOM.

Files:
- `tools/tc-grid/scripts/measure_vram_after_load.sh` (new)
- `docs/sprints/SPRINT-023-VRAM-PROFILE.md` (new — recipe + observed
  numbers)

### P6 — Performance measurement + ncu protocol

**Goal**: validate we hit tg32 ≥ 20 t/s and pp128 ≥ 20 t/s. If short,
identify the gap.

- Run the same `llama-bench` invocation as SPRINT-022 with the new
  `-ot` flag. Capture pp128 and tg32 from 3 runs (cold + 2 warm).
- ncu metric pack on one MXFP4 mul_mat per shape class (M=8 decode,
  M=128 prefill). Confirm HMMA active% tracks REPORT-15 (≥45% for
  MXF4 at gs=32). If active% is much lower than the bench number, we
  have a host-side launch-overhead bottleneck → stretch into stream-
  batched dispatch.
- Compare with `-DGGML_TURBOMIND_GEMM=OFF` baseline to compute uplift.

**Ship gate**: `tg32 ≥ 20.0 t/s` median of 3 warm runs. If we miss,
write a follow-up identifying the bottleneck (launch overhead,
activation cast cost, scalar-path leakage on non-MXFP4 tensors) and
file as SPRINT-024 input. We do not block the sprint on a perf
target if correctness and the dispatcher are landed — the merge
notes will document the gap.

Files:
- `tools/tc-grid/scripts/run_sprint023_bench.sh` (new — wraps the
  measurement protocol)
- `tools/tc-grid/docs/REPORT-17-SPRINT-023.md` (new — final results)
- `tools/tc-grid/docs/ncu-sprint023-mxf4-decode.txt` (new)
- `tools/tc-grid/docs/ncu-sprint023-mxf4-prefill.txt` (new)

---

## 5. Files Summary

| Path | Kind | Phase | Reason |
|---|---|---|---|
| `ggml/src/ggml-cuda/CMakeLists.txt` | modify | P1 | Add `GGML_TURBOMIND_GEMM` option + FetchContent |
| `ggml/src/ggml-cuda/turbomind_glue.cmake` | new | P1 | Encapsulate turbomind subdirectory + cutlass alias setup |
| `CMakeLists.txt` (top-level) | modify | P1 | Surface flag at top level |
| `ggml/src/ggml-cuda/turbomind_buffer.cu` | new | P2 | `CUDA0_TURBOMIND` buffer type + `init_tensor` packer |
| `ggml/src/ggml-cuda/turbomind_buffer.cuh` | new | P2 | Public buffer-type interface |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | modify | P2 | Register new buffer type in `ggml_backend_cuda_reg` |
| `ggml/src/ggml-cuda/turbomind_gemm.cu` | new | P3, P4 | `ggml_cuda_turbomind_mul_mat` dispatch entry |
| `ggml/src/ggml-cuda/turbomind_gemm.cuh` | new | P3 | Declarations + `can_dispatch` predicate |
| `ggml/src/ggml-cuda/mmq.cu` | modify | P3 | Insert turbomind predicate at top of `ggml_cuda_mul_mat_q` |
| `ggml/src/ggml-cuda/mmvq.cu` | modify | P3 | Same predicate at top of `ggml_cuda_mul_mat_vec_q` |
| `tools/tc-grid/turbomind_minimal/test_dispatch.cu` | new | P3 | Correctness gate vs scalar path |
| `tests/test-backend-ops-turbomind.cu` | new | P2 | Buffer-type round-trip test |
| `tools/tc-grid/scripts/measure_vram_after_load.sh` | new | P5 | nvidia-smi VRAM probe |
| `tools/tc-grid/scripts/run_sprint023_bench.sh` | new | P6 | Bench protocol wrapper |
| `tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu` | (no edit) | P0 | Already supports `--group-size 32` |
| `tools/tc-grid/docs/REPORT-15.md` | modify | P0 | Append §7 MXF4 gs=32 results |
| `tools/tc-grid/docs/REPORT-15-mxf4-gs32.csv` | new | P0 | MXF4 ceiling CSV |
| `tools/tc-grid/docs/REPORT-17-SPRINT-023.md` | new | P6 | Sprint close report |
| `tools/tc-grid/docs/ncu-sprint023-mxf4-decode.txt` | new | P6 | ncu raw output |
| `tools/tc-grid/docs/ncu-sprint023-mxf4-prefill.txt` | new | P6 | ncu raw output |
| `docs/sprints/SPRINT-023-VRAM-PROFILE.md` | new | P5 | VRAM recipe |

**Total: 15 new files, 5 modified files.**

---

## 6. Definition of Done

All criteria measurable; each maps back to an intent success criterion.

| # | Criterion | Measurement | Phase |
|---|---|---|---|
| DoD-1 | `cmake -DGGML_TURBOMIND_GEMM=ON` builds clean on gpu-01 (V100, CUDA 12.2, gcc 11.4) | `make -j` exits 0, no spill warnings on turbomind .cu files, `nm` confirms `tmg::Gemm::Run` symbol present | P1 |
| DoD-2 | `=OFF` build runs identical to SPRINT-022 baseline | `llama-bench -m DSv4-Flash.gguf -ot exps=CPU -ngl 99` matches recorded pp128/tg32 within run variance (±5%) | P1 |
| DoD-3 | Turbomind buffer-type registers correctly | `ggml_backend_dev_buffer_type(dev, "CUDA0_TURBOMIND")` returns non-null on V100 | P2 |
| DoD-4 | MXFP4 packing at `init_tensor` is bit-stable | Same tensor packed twice yields identical packed bytes (cudaMemcpy + memcmp) | P2 |
| DoD-5 | Dense MXFP4 mul_mat correctness | `test_dispatch.cu` passes 10 cases (M ∈ {1,128} × 5 random (N,K)), rel ≤ 2e-2 vs nisparks scalar | P3 |
| DoD-6 | `=ON` build with no `-ot` override = `=OFF` build | Same `llama-bench` invocation produces same pp128/tg32 ±5% | P3 |
| DoD-7 | MoE-ids path produces correct text | `llama-bench --temp 0 --seed 1 -ot 'blk\.([0-9]|1[0-5])\..*exps=CUDA0_TURBOMIND' -ngl 99` first 32 generated tokens byte-identical to baseline with same seed | P4 |
| DoD-8 | VRAM stays under 30 GiB | `nvidia-smi --query-gpu=memory.used --format=csv,nounits` after model load + 32 token decode ≤ 30720 MiB | P5 |
| DoD-9 | tg32 ≥ 20 tok/s | `llama-bench` median of 3 warm runs ≥ 20.0 (target = 4.2× over 4.73 baseline) | P6 |
| DoD-10 | pp128 ≥ 20 tok/s | Same protocol, median ≥ 20.0 (target = 4.7× over 4.28 baseline) | P6 |
| DoD-11 | ncu HMMA active% ≥ 45% on MXF4 mainloop | One ncu run per shape, `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_elapsed` ≥ 45 | P6 |
| DoD-12 | Runtime kill-switch works | `LLAMA_DISABLE_TURBOMIND=1 llama-bench` with `-ot ...=CUDA0_TURBOMIND` falls back to scalar without crashing; produces correct (slow) output | P3 |
| DoD-13 | Documentation closed | REPORT-17 + VRAM profile written, REPORT-15 §7 appended | P6 |

**Sprint ships when DoD-1 through DoD-8 + DoD-12 + DoD-13 are met.**
DoD-9, DoD-10, DoD-11 are the perf bar — if any miss, the sprint
still ships (correctness, dispatcher, infrastructure all landed) but
the close report flags the gap as SPRINT-024 input.

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MXF4 at gs=32 doesn't hit ~45 TF (P0 ship gate fails) | Medium | High — premise of sprint | P0 is the first phase precisely to catch this. If MXF4 stalls, fall back: pivot SPRINT-023 to F8_E4M3_B128 dense port (simpler — no MoE ids handling needed); MoE becomes SPRINT-024. |
| Turbomind layout converters break at unusual (N,K) DSv4 shapes | Medium | Medium | P3 correctness gate uses production shapes (7168×7168, 7168×18944). Catch at gate, not in production. The U4 illegal-mem-access at N≠K from REPORT-15 §6 is a known failure mode for U4; MXF4 has different code path but same risk class. |
| Buffer-type wrapper breaks other backends' assumptions about CUDA buffers | Medium | High — silent corruption risk | The new buffer type forwards to base `ggml_backend_cuda_buffer_*` for non-MXFP4 tensors. ggml-cuda's existing API expects raw VRAM pointers from `tensor->data`; we keep that contract — only the metadata side-table is new. Reviewer-flagged: confirm `tensor->data` always points at the **packed** weight, not the canonical GGUF, after `init_tensor`. |
| Activation cast F32 → F16 → F32 introduces accuracy regression | Low-Medium | Medium | F16 activation is what turbomind expects; DSv4 is already FP8/FP4 trained so activations are noise-tolerant. Validate with token-equivalence test (DoD-7). If we see drift, add a fallback to F32 activations (turbomind has a kFloat config but only on newer arches; we'd need to find/build an sm70 F32 mainloop). |
| Per-expert kernel launch overhead saturates host CPU | Medium | High — could cap us at <20 t/s | At 256 launches/forward × 32 tokens × 60 layers = 491k launches/decode. At 10 us/launch = 4.9 s overhead. P4 ships sequential; P6 must measure this. If it's the bottleneck, P6 stretch is stream-batched dispatch or cudaGraph capture. |
| VRAM fragmentation across many small per-expert allocations | Medium | Medium | Pre-allocate one contiguous block per layer (256 × packed_weight_size); set up MatrixLayout descriptors as offsets into the block, not separate cudaMallocs. This matches turbomind's own arena pattern. |
| Scope creep into F8_E4M3 dense port | High (temptation) | High — slips sprint | Explicit out-of-scope. F8 dense is SPRINT-024. The MXF4 / dense F8 share the dispatch shim — adding F8 is cheap once it's there, but doing it inside SPRINT-023 doubles correctness work. Resist. |
| nisparks's MXFP4 scalar path quietly disagrees with our turbomind output at edge shapes (M=1 specifically) | Medium | Medium | M=1 (true mat-vec) gets dispatched through mmvq.cu not mmq.cu; tile-1 fragment-1 fallback paths in turbomind may differ. P3 correctness test must include M=1. If it disagrees, gate turbomind to M ≥ 8 in `can_dispatch` and keep scalar for true decode. (Notably: this would still cover most decode, since DSv4 has multi-token KV updates and our M is per-active-expert which is rarely exactly 1.) |
| Build flag interaction with downstream forks | Low | Low | Default OFF preserves stock behavior. The flag is additive. Document the flag in `docs/build.md`. |
| Memory ordering bug in workspace reuse across CUDA streams | Low (sequential first) | High (silent corruption) | P4 ships sequential per-expert with shared workspace. P6 stream-batched would need per-stream workspace partitioning — explicitly flagged in P6 plan, not done blindly. |

---

## 8. Security

The new dispatch path introduces three potential OOB / memory-safety concerns:

**S-1: `init_tensor` packing reads from GGUF mmap-backed source.** The
GGUF block format is `QK_MXFP4=32` weights + scale per block. If the
loader hands us a malformed tensor (e.g. nb[0] inconsistent with the
expected block stride), our `extend_to_u16` call could read past the
mmap region. Mitigation: validate `ggml_nbytes(tensor) == n_blocks *
sizeof(block_mxfp4)` and `ne[0] % 32 == 0` before kicking off the
converter. Abort with `GGML_ASSERT` on mismatch — never silently
truncate. Same defensive pattern as the existing
`ggml_backend_cuda_buffer_init_tensor` does for stock GGUF types.

**S-2: Workspace buffer reuse across calls.** The `partials` buffer is
sized at construction for `max_M * max_N * 4 * sizeof(float)`. If a
later mul_mat exceeds that (e.g. a context-length extension), the
turbomind kernel will silently write past the end. Mitigation:
recompute `partials_bytes` per call from the actual `args` shape, grow
via `ggml_cuda_pool_alloc` if needed — never trust the construction-
time max. The pool's `_alloc` API already does grow-on-demand.

**S-3: Per-expert pointer arithmetic with attacker-controlled `ids`
tensor.** If `expert_bounds[e]` is malformed (e.g. bounds[e+1] <
bounds[e], or bounds beyond `M`), our per-expert slice computation
produces an out-of-bounds source pointer for the activation. The
upstream `ggml_cuda_launch_mm_ids_helper` already validates ids; we
don't re-implement that, but we add an `assert(bounds_sorted &&
bounds_in_range)` check on the host side before the kernel launch loop.
Production builds ifdef this out; debug builds catch the bug. The
`ids` tensor is computed by the model graph from logits, not user
input, so the attack surface is "model file produces malicious
ids" — a low-severity threat model but worth the assert.

**S-4 (not specific to this sprint but worth noting): `LMDEPLOY_SRC`
is a CMake variable.** If a user points it at an attacker-controlled
directory, CMake's `add_subdirectory` will execute arbitrary CMake
from that path. We document that `LMDEPLOY_SRC` must be trusted —
same posture as the existing `tools/tc-grid/turbomind_minimal/`.

No new network surface, no new file I/O paths, no privilege boundaries
crossed.

---

## 9. Dependencies

**External libraries (already vendored or fetched):**
- **turbomind** sources: from `research/lmdeploy/` (untracked in main
  tree; we maintain `cuda-patches/` for our adjustments). Path is
  `cmake -DLMDEPLOY_SRC=…`-configurable to allow OOT setups.
- **CUTLASS**: turbomind links `nvidia::cutlass::cutlass`. Already
  pulled by tc-grid via FetchContent; we alias `cutlass_SOURCE_DIR`
  the same way.
- **fmt** 11.0.2: header-only, FetchContent. Same as tc-grid.
- **concurrentqueue**: pulled by turbomind/core's own CMakeLists; no
  action needed.

**Sprint preconditions:**
- SPRINT-022 baseline shipped and reproducible (✅ confirmed in intent
  doc orientation summary).
- SPRINT-020 P1.5 turbomind carve-out validated (✅ — that's what
  `gemm_bench_packed.cu` is).
- SPRINT-021 P0 ceilings measured (✅ — REPORT-15).
- `cuda-patches/` directory present with any required turbomind
  source patches (✅ per intent doc).

**Hardware / environment:**
- gpu-01 V100 SXM2 32 GiB, sm_70 only. CUDA 12.2 / gcc 11.4 / cmake
  3.22.1.
- 251 GiB host RAM (already used for SPRINT-022 — overflow for
  non-hot experts).
- `llamacpp-build` pod provisioned (✅).
- `nvidia.com/gpu.deploy.dcgm-exporter=false` node-label toggle access
  for ncu runs (per REPORT-15 §3.3 protocol).

**Software toolchain:**
- ncu (Nsight Compute) for HMMA active% measurements (P6).
- nvidia-smi for VRAM probes (P5).
- The standard llama.cpp build env (already in use).

**Sprint precondition NOT met (and we have to do it ourselves):**
- MXF4 ceiling at `group_size=32`. SPRINT-021 P0 didn't measure it.
  P0 of this sprint takes that on.

---

## 10. Open Questions

Carried forward from intent doc + new ones surfaced by this draft.

1. **MXFP4 `group_size=32` registry coverage at production shapes.**
   The intent doc §"Open questions" #3 flags this. SPRINT-021 P0
   only measured `gs=128` (no registry match). We hit P0 of THIS
   sprint as the first action. If `Config_MXF4` at `gs=32` doesn't
   give us ≥45 TF at N=K=7168, this whole sprint's premise weakens
   and we should pivot to **Config_U4_g** (which would need a
   GGUF MXFP4 → U4 quant conversion at `init_tensor` time — a
   different, harder problem).

2. **Per-expert kernel launch overhead at decode (M small).** We have
   no good estimate. The closest data point is SPRINT-021 P0 which
   measured a single GEMM at M ∈ {64, 2048}. We don't know per-launch
   overhead for M=1 / M=8 (decode-band per-expert M). If it's > 50
   us/launch, sequential dispatch caps us well below 20 t/s and we
   need stream-batching from day 1. P0 should add a launch-overhead
   sub-experiment.

3. **Should `can_dispatch` admit M=1 in the MoE path?** True decode
   produces M=1 per active expert (one token's activation routed to
   8 of 256 experts). turbomind's mainloop is registered with
   `BM ∈ {8, 16, 32, 64, 128}`. M=1 may dispatch via the BM=8 tile
   with 7 padding rows, but we haven't measured. If it's pathological
   we keep scalar for M=1 and only turbomind for M ≥ 8.

4. **Activation cast F16 vs F32 at the boundary.** ggml's MoE
   activation tensors are F32. turbomind expects F16 for the A
   operand. We cast at the dispatch boundary; question is whether to
   keep a pool-allocated F16 buffer per-call (cheap, GC by ggml pool)
   or pre-allocate per-layer at model load (saves the per-call alloc
   but ties us to layer-aware allocation). Default to per-call pool
   alloc; revisit if profiling shows allocation overhead.

5. **Does the existing `-ot exps=…` regex support the buffer type
   string `CUDA0_TURBOMIND`?** `common/arg.cpp:2280` calls
   `parse_tensor_buffer_overrides` which presumably enumerates the
   registered buffer types by name. P2 ship gate must explicitly test
   the string flows end-to-end. If the parser is case-sensitive or
   doesn't see the new buffer type until after llm_load_tensors, we
   need a hookpoint earlier.

6. **F8_E4M3_B128 dense port — definitely SPRINT-024?** The intent
   doc lists it as a success criterion (item: "F8_E4M3_B128 GPU path
   correctness"). This draft scopes it out. **The sprint planner
   should resolve this before P0**: if we must include F8, P1 grows
   to register two converters and P3/P4 add a parallel `case
   GGML_TYPE_F8_E4M3_B128:`. ~30% more work; ~10% more risk per phase.
   Strong recommendation: ship MXFP4-only in SPRINT-023, F8 in
   SPRINT-024.

7. **WMMA-MMVQ MoE port (deferred from SPRINT-017 P2+P3, 61f9ebebe +
   5ec3a9b41)** — intent doc flags this as "complete here or hold".
   Recommendation: **hold for SPRINT-024**. It's an INT8 path that
   becomes deprecated if MXFP4/FP8 turbomind dispatch ships and lands
   the perf target. No reason to invest in INT8 plumbing for a precision
   regime we're abandoning per REPORT-15 §2.

8. **NVFP4 in-tree but unused — drop the dispatch cases or leave
   them?** GGUF supports `GGML_TYPE_NVFP4` but DSv4 doesn't use it.
   Out of scope for this sprint — leave the existing scalar dispatch
   alone. If we want to consolidate later, that's a cleanup sprint.

9. **Hot-expert profile JSON for SPRINT-024.** This sprint uses the
   "first 16 layers" heuristic. Before SPRINT-024, design the JSON
   schema:
   ```json
   { "model": "DSv4-Flash-256e", "n_experts": 256, "n_layers": 60,
     "hits": [[layer, expert, hit_count], ...] }
   ```
   Threshold for "hot" determined by sum-of-hits coverage
   (e.g. top 30% of experts cover 80% of routing — that's the cut).
   Out of scope here; placeholder noted for SPRINT-024 planning.

10. **Cross-checking against the V100 WMMA bank-conflict and
    half/float frag-layout lessons in MEMORY.md.** Our turbomind path
    inherits whatever turbomind's mainloop does; we don't reimplement
    the mma. So those lessons don't apply to the kernel itself. They
    DO apply if we ever need to write a custom fixup kernel (e.g.
    for an activation cast or per-expert reduction). If we go there,
    re-read MEMORY.md before writing the kernel.

---

*End of draft. ~2.5 pages of substantive content. Phase P0 is the
first thing to schedule; phase P1+P2 can proceed in parallel with
P0 since they're build/buffer scaffolding rather than perf-dependent.*

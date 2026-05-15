# SPRINT-023 — Smallest-Slice GPU-Resident Hot Experts via Turbomind Carve-Out (Skeptical Draft)

**Author perspective**: senior systems engineer, GGML/CUDA integration scars,
strong bias toward smallest-mergeable-slice and toward refusing optimistic
top-line targets that the host loop can't physically deliver.

**Status**: alternative draft to `SPRINT-023-INTENT.md`. Disagrees with the
intent on (a) the 20 t/s decode target, (b) doing MXFP4+FP8+hot-expert in one
sprint, (c) underspecified hot-expert mechanism, (d) underestimating the
GGUF↔turbomind layout conversion cost, (e) the ABI/build risk of dragging
`gemm2 + core + parser + cuda_utils + fmt` into the ggml-cuda `.so`.

---

## 1. Overview

This sprint moves the **smallest correct slice** of MoE expert compute from
CPU to V100 GPU using turbomind's `Config_E4M3` (FP8) kernel, behind a feature
flag, on a **single named expert tensor at a time**. We deliberately defer
MXFP4, NVFP4, and hot-expert *selection* policy to SPRINT-024. The "20 t/s
decode" target from the intent is **not adopted** as a Definition-of-Done
gate. We adopt a measured floor that is achievable given V100 sm70 limits and
the host-side bottlenecks observed in SPRINT-022.

The integration vehicle is a **new ggml-cuda dispatcher layer** —
`ggml-cuda/turbomind_dispatch.{cu,cuh}` — that consumes a **separate
GPU-resident packed-weight handle** (not the raw GGUF block), produced once
at upload time and cached. We do **not** route turbomind through the existing
`block_*` SOA layouts; that's a category error (different packing, different
scale layout, different K-tile alignment) and trying to do it in-place will
cost us the sprint.

### Why this scope, not the intent's scope

| Intent goal | This draft | Reason |
|---|---|---|
| MXFP4 + FP8 + hot-expert | **FP8 only**, on one tensor type | Three independent integration risks at once; we cannot keep correctness gates honest if any of them fails halfway |
| `tg32 ≥ 20 t/s` (4.2× over 4.73) | **`tg32 ≥ 8 t/s` floor, 12 t/s stretch** | See §7 — host MoE dispatch loop, PCIe activation transfer for cold experts, and per-token KV/attention serialization put 20 t/s out of single-GPU sm70 reach |
| Hot-expert profile (unspecified mechanism) | **Forced decision: offline static JSON from a profile pass, no runtime promotion** | The intent left this open; deferring the decision into execution will burn week 2 |
| `GGML_TURBOMIND_GEMM=ON` links gemm2+core+parser+cuda_utils+fmt into ggml-cuda | **Carve-out is built as a separate shared library `libggml-turbomind.so`** loaded only when env-var enabled; ggml-cuda exports stable C ABI shim | See §7.5 — pulling fmt + parser into the ggml-cuda translation unit will balloon link time, may break BUILD_SHARED_LIBS, and bleeds C++ template instantiations into a backend that strives for C ABI cleanliness |

---

## 2. Use Cases

1. **Decode hot path** (primary): a single `ffn_*_exps` tensor for a hot
   layer is resident in VRAM as a turbomind-packed FP8 weight. During
   decode, when the router selects that expert ID, `mul_mat_id` for that
   tensor dispatches to `turbomind::gemm::Gemm::Run` instead of the scalar
   `vec_dot_*_q8_1` path. All other experts continue to use the SPRINT-022
   CPU path.
2. **Prefill** (secondary, gated): same tensor; same kernel; M=128 instead
   of M=1. Lower priority — `pp128 ≥ tg32` is the SPRINT-022 reality, and
   small-M is where turbomind kernels are weakest on sm70.
3. **Fallback** (always available): with `GGML_TURBOMIND_GEMM=OFF` (default
   in this sprint) the binary behaves exactly like SPRINT-022. Required for
   regression bisecting and for users without the carve-out built.
4. **Out of scope this sprint**: MXFP4 dense layers, NVFP4 anything, full
   model expert residency (we move 8–16 *named* experts, not 256×60), online
   hot-expert promotion, multi-GPU.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  llama.cpp model load                                            │
│  └─ src/llama-model-loader.cpp                                   │
│     ├─ mmap GGUF (canonical, untouched)                          │
│     └─ For each tensor named in hot-expert JSON:                 │
│        └─ ggml-backend-cuda: alloc + upload                      │
│           └─ NEW: if tensor type==F8_E4M3_B128 AND env enabled   │
│              AND name in hot-list → side-channel pack to         │
│              turbomind layout, store handle in `tm_weight_cache` │
└─────────────────────────────────────────────────────────────────┘

  Inference (mul_mat_id over experts):
  ┌──────────────────────────────────────────────────────────────┐
  │ ggml-cuda/mmvq.cu :: ggml_cuda_mul_mat_vec_q                 │
  │   case GGML_TYPE_F8_E4M3_B128:                               │
  │     if (turbomind_dispatch_available(tensor, M)) {           │
  │       turbomind_dispatch_fp8_mmv(...);  // NEW               │
  │     } else {                                                  │
  │       mul_mat_vec_q_switch_ncols_dst<...>(...); // existing  │
  │     }                                                         │
  │     break;                                                    │
  └──────────────────────────────────────────────────────────────┘

  NEW: ggml-cuda/turbomind_dispatch.{cu,cuh}
    ├─ tm_weight_cache: map<tensor_ptr, PackedWeight>
    ├─ pack_f8_e4m3_b128_to_turbomind(...)
    │    Reuses gemm_bench_packed.cu's build_packed_weight() logic
    │    but reads from real GGUF blocks (block_f8_e4m3_b128) instead
    │    of random bytes. Outputs PackedWeight handle.
    ├─ turbomind_dispatch_fp8_mmv(...): wraps Gemm::Run
    └─ Loads `libggml-turbomind.so` via dlopen (so ggml-cuda.so does
       not transitively link gemm2/parser/cuda_utils/fmt). Symbol
       lookup gated by env var.
```

Hot-expert selection mechanism (forced decision): **offline static JSON**
at a known path (`GGML_TURBOMIND_HOT_EXPERTS_JSON`, e.g.
`models/dsv4-flash-hot.json`), with schema `{"layer": int, "expert": int}[]`.
The list is consulted **once at load time only**; no online promotion. If the
file is absent, the feature is a no-op even when the env flag is on. The
profile pass that generates this JSON is a separate one-shot tool
(`tools/profile-experts/`) — out of scope this sprint, **mocked with a
manually-authored 16-entry JSON** for the perf measurement.

---

## 4. Implementation (phased, smallest-slice-first)

Each phase is independently mergeable and gated by `GGML_TURBOMIND_GEMM=ON`.
**Do not start phase N+1 until phase N's DoD ticks green on V100.**

### Phase 1 — Carve-out as a separate shared library (1.5–2 days)

**Goal**: produce `libggml-turbomind.so` that exports a tiny C ABI and links
`gemm2 + core + cuda_utils + fmt` internally. No ggml-cuda changes yet.

Files:
- `tools/turbomind-carveout/CMakeLists.txt` *(new — lift from `tools/tc-grid/turbomind_minimal/CMakeLists.txt`)*
- `tools/turbomind-carveout/include/ggml_tm.h` *(new — C ABI header)*
  - `tm_handle tm_pack_fp8_b128(const void* gguf_block_ptr, int N, int K, int group_size, cudaStream_t)`;
  - `int tm_gemm_fp8_mmv(tm_handle w, const __half* A, int M, int N, int K, __half* D, cudaStream_t)`;
  - `void tm_handle_free(tm_handle)`;
  - `int tm_version()` for ABI handshake.
- `tools/turbomind-carveout/src/tm_pack_fp8.cu` *(new — reuses
  `build_packed_weight()` logic from `gemm_bench_packed.cu` lines 188–348,
  but takes real `block_f8_e4m3_b128` GGUF data rather than random bytes)*
- `tools/turbomind-carveout/src/tm_gemm_fp8.cu` *(new — wraps `Gemm::Run` for
  M-batched FP8 path, parallels `gemm_bench_packed.cu` lines 474–520)*

**Exit criteria for Phase 1**:
- `libggml-turbomind.so` builds clean on V100 + CUDA 12.2 + gcc 11.4.
- `nm -D` shows only the four `tm_*` symbols externally; no `fmt::*`,
  no `turbomind::*` leaked.
- Standalone unit test `tm_smoke.cu` (in same dir) calls `tm_pack_fp8_b128`
  on a synthetic 7168×7168 block buffer and `tm_gemm_fp8_mmv` for
  M ∈ {1, 128} — gets `Gemm::Run` rc=0.

### Phase 2 — GGUF block → turbomind packed converter (1–2 days)

**Goal**: replace synthetic byte fill in `tm_pack_fp8_b128` with a real
mapping from GGML's `block_f8_e4m3_b128` SOA layout to turbomind's expected
packed layout + per-128-K-group scale tensor.

Files:
- `tools/turbomind-carveout/src/tm_pack_fp8.cu` *(extend)* — add a
  CUDA kernel `unpack_block_f8_e4m3_b128` that:
  - Reads `block_f8_e4m3_b128` (1 byte scale `d` per 128 K-values, then 128
    e4m3 bytes `qs`).
  - Materializes a `(N, K)` raw e4m3 buffer **plus** a `(N, K/128)`
    f32-or-fp16 scale buffer in the format turbomind's
    `conv_w->Convert` + `conv_s->Convert` expects.
  - **CRITICAL**: GGML's scale is one byte (`uint8_t d`) interpreted per
    convention (clarify whether it's `E8M0` or scaled-fp16-as-uint8 — the
    block name suggests E8M0 but `convert.cu`'s `dequantize_f8_e4m3_b128`
    is the only source of truth; read it before writing this).
  - Turbomind `Config_E4M3` expects per-K-group scales as `kFloat` (see
    `gemm_bench_packed.cu` line 73). A scale type conversion kernel is
    required.

**Exit criteria for Phase 2**:
- `tools/turbomind-carveout/tests/test_fp8_pack_correctness.cu`: for 4
  random `(N, K)` ∈ `{(7168,7168), (2048,7168), (7168,2048), (16384,7168)}`:
  1. Fill a `block_f8_e4m3_b128` buffer with deterministic values.
  2. Dequant via existing `ggml-cuda/convert.cu :: dequantize_row_f8_e4m3_b128`
     to FP16 reference D₁.
  3. Pack via new path, run `tm_gemm_fp8_mmv` with `A = I` (or near-I) to
     get D₂.
  4. Gate: `max_rel_err(D₁, D₂) < 5e-3` for M=1, `< 2e-2` for M=128.

### Phase 3 — ggml-cuda dispatch shim, dlopen-loaded (1 day)

**Goal**: route `GGML_TYPE_F8_E4M3_B128` mul_mat through the shim *only*
when env enabled AND tensor is in hot list AND `M ≤ 256` (skip prefill
for now; FP8 sm70 at small M is the lowest-risk window).

Files:
- `ggml/src/ggml-cuda/turbomind_dispatch.cu` *(new)*
- `ggml/src/ggml-cuda/turbomind_dispatch.cuh` *(new)*
  - Defines `tm_dispatch_init()` (dlopen on first call), `tm_dispatch_for_tensor(...)`,
    `tm_dispatch_available_for_type(...)`.
- `ggml/src/ggml-cuda/mmvq.cu` *(edit line 959–964 — `case GGML_TYPE_F8_E4M3_B128`)*
  - Wrap the existing `mul_mat_vec_q_switch_ncols_dst` call in
    `if (!tm_dispatch_try(...))`.
- `ggml/src/ggml-cuda/CMakeLists.txt` *(edit)*
  - Add `turbomind_dispatch.{cu}` to source list under
    `if(GGML_TURBOMIND_GEMM)`.
- `ggml/CMakeLists.txt` *(edit)*
  - Add `option(GGML_TURBOMIND_GEMM "..." OFF)`. **Do NOT** add
    `turbomind-carveout` as a build subdir of ggml — keep them independent.
    The dispatch shim discovers `libggml-turbomind.so` at runtime.

**Exit criteria for Phase 3**:
- `GGML_TURBOMIND_GEMM=ON` build succeeds; ggml-cuda `.so` does not export
  any `turbomind::*` or `fmt::*` symbols.
- With env unset and JSON absent, behavior is bit-identical to SPRINT-022
  on a 32-token decode (compare via `bit-compare` discipline noted in
  homelab skills memory).
- With env set + JSON listing 1 expert tensor, `ltrace`/dlopen log shows
  `libggml-turbomind.so` load exactly once at first `mul_mat`.

### Phase 4 — Upload-time packing hook (1 day)

**Goal**: when a tensor lands in VRAM and matches the hot list, run
`tm_pack_fp8_b128` against the freshly-uploaded GGUF block and stash the
handle in `tm_weight_cache` keyed by tensor pointer.

Files:
- `ggml/src/ggml-cuda/ggml-cuda.cu` *(edit `ggml_backend_cuda_buffer_set_tensor`
  or equivalent set-tensor entry point)* — after the memcpy, call
  `tm_pack_tensor_if_hot(tensor)`. Cache lives in `turbomind_dispatch.cu`.
- `ggml/src/ggml-cuda/turbomind_dispatch.cu` *(extend)* — implement
  `tm_pack_tensor_if_hot`, `tm_lookup_handle(tensor)`,
  `tm_free_all_handles()` (called from buffer free).

**Memory accounting** (do this in code, log on each pack):
- Original `block_f8_e4m3_b128` weight: `N*K * (1 + 1/128)` bytes ≈ N*K * 1.008.
- Turbomind packed: at minimum `N*K * 1` (FP8) + `N*K/128 * 4` (fp32 scales)
  = N*K * 1.031.
- **We DUPLICATE the weight on GPU** during this sprint — original stays
  because the fallback dispatch reads it. That's a ~2× memory cost per hot
  tensor. Folded into §7 budget concerns.

**Exit criteria for Phase 4**:
- Pack hook runs once per hot tensor at load time, log line shows it.
- VRAM after model load is within 1 GiB of the expected budget per §7.6.

### Phase 5 — End-to-end measurement + tolerance gate (1 day)

Files:
- `scripts/sprint-023-bench.sh` *(new)* — runs `llama-bench` with hot list
  populated for the top-K experts in layers 5–10 (a hand-authored mock
  list; profile-driven list is SPRINT-024).
- `tools/profile-experts/README.md` *(new — stub)* — describes how the
  hot-list JSON will be generated in SPRINT-024; out of scope this sprint.
- `docs/sprints/drafts/SPRINT-023-REPORT.md` *(written at close)*

**Exit gates** (see §6 for full DoD):
- `tg32` floor: ≥ **8 t/s** (1.7× over SPRINT-022's 4.73). Stretch ≥ 12 t/s.
- `pp128` floor: ≥ **6 t/s** (1.4× over 4.28). Stretch ≥ 10.
- VRAM < 28 GiB total (leaves 4.5 GiB headroom — see §7.6).
- Fallback (`OFF`) bit-equivalent to SPRINT-022.

---

## 5. Files Summary

**New (16 files, all under feature flag)**:
- `tools/turbomind-carveout/CMakeLists.txt`
- `tools/turbomind-carveout/include/ggml_tm.h`
- `tools/turbomind-carveout/src/tm_pack_fp8.cu`
- `tools/turbomind-carveout/src/tm_gemm_fp8.cu`
- `tools/turbomind-carveout/src/tm_dlopen_stub.cpp`  *(for soft-load symbol resolution)*
- `tools/turbomind-carveout/tests/test_fp8_pack_correctness.cu`
- `tools/turbomind-carveout/tests/tm_smoke.cu`
- `ggml/src/ggml-cuda/turbomind_dispatch.cu`
- `ggml/src/ggml-cuda/turbomind_dispatch.cuh`
- `scripts/sprint-023-bench.sh`
- `scripts/sprint-023-verify-vram.sh`
- `tools/profile-experts/README.md`  *(stub)*
- `models/dsv4-flash-hot.json`  *(mock hot list, 16 entries, hand-authored)*
- `docs/sprints/drafts/SPRINT-023-CODEX-DRAFT.md`  *(this file)*
- `docs/sprints/drafts/SPRINT-023-REPORT.md`  *(at close)*
- `docs/sprints/SPRINT-023-DEFERRED.md`  *(at close)*

**Edited (4 files)**:
- `ggml/CMakeLists.txt` *(add `GGML_TURBOMIND_GEMM` option)*
- `ggml/src/ggml-cuda/CMakeLists.txt` *(add `turbomind_dispatch.cu` source)*
- `ggml/src/ggml-cuda/mmvq.cu` *(line 959–964: wrap F8_E4M3_B128 case)*
- `ggml/src/ggml-cuda/ggml-cuda.cu` *(set-tensor hook for upload-time pack)*

**Unchanged this sprint** (despite intent suggesting otherwise):
- `ggml/src/ggml-cuda/convert.cu` — we **don't** plumb turbomind through here.
  Conversion happens in the dispatch shim, not in `ggml_get_to_*_cuda`.
- `ggml/src/ggml-cuda/mmq.cu` — prefill path; defer until Phase 5 shows
  decode wins.
- `src/llama-memory-deepseek4.{cpp,h}` — no MoE memory layout changes.
- `src/models/deepseek4-family.cpp` — graph is untouched.
- `cuda-patches/` — frozen; we use the existing carve-out CMake pattern.

---

## 6. Definition of Done

| # | Criterion | Measurement |
|---|---|---|
| D1 | `libggml-turbomind.so` builds on V100 sm70 with no spill warnings, no fmt/turbomind symbols leaked | `cmake --build && nm -D` |
| D2 | `test_fp8_pack_correctness` passes for 4 shapes at tolerance gates of §Phase 2 | `ctest -R fp8_pack` |
| D3 | With `GGML_TURBOMIND_GEMM=OFF`: `llama-bench` matches SPRINT-022 within ±1% on `tg32` and `pp128` | bench artifact diff |
| D4 | With `GGML_TURBOMIND_GEMM=ON` + 16-entry hot JSON: `tg32 ≥ 8.0 t/s` median-of-3 | bench CSV |
| D5 | VRAM after model load + 32-token decode < 28 GiB | `nvidia-smi --query-gpu=memory.used` |
| D6 | ncu metric pack on one hot expert mul_mat shows `hmma_active% ≥ 45%` (lower than REPORT-15's 56% because M is small for decode) | `tools/tc-grid/scripts/run_ncu3.sh` (promote from /tmp per REPORT-15 §6) |
| D7 | Stretch (not blocking): `tg32 ≥ 12 t/s` median-of-3 | bench CSV |
| D8 | Sprint report committed with TFLOPS table, ncu numbers, and explicit list of what was deferred to SPRINT-024 | `docs/sprints/drafts/SPRINT-023-REPORT.md` |

**Not in DoD** (intent goals we are explicitly refusing this sprint):
- `tg32 ≥ 20 t/s`. Reasons in §7.1. May be feasible after SPRINT-024 + 025
  (full hot-expert residency + prefill path + KV-cache tuning).
- MXFP4 dispatch. Reason: `group_size=32` constraint, sm70 registry only has
  partial tile coverage; adds a second packing path and a second tolerance
  contract.
- Hot-expert *selection* (profile-driven). Reason: load-time JSON is enough
  to prove the dispatch hook works; profile pipeline is a separate sprint.

---

## 7. Risks (DETAILED)

### 7.1 The 20 t/s target is not realistic this sprint

**Why I'm pushing back hard on it**:

- SPRINT-022 baseline is `tg32 = 4.73 t/s`. The `pp ≈ tg` ratio (0.9) tells us
  we are **end-to-end CPU-expert bound**. Moving expert compute to GPU
  doesn't automatically remove that bottleneck — it relocates it:
  - The MoE dispatch path in `src/models/deepseek4-family.cpp` still does
    host-side gather/scatter for top-K routing per token (verify, but this
    is the GGML norm).
  - Per-token launch overhead on V100: a single `Gemm::Run` for M=1 costs
    ~30–80 µs (kernel launch + descriptor build + workspace bind). For 8
    active experts × 60 layers × per-token = ~30 ms/token just in launch
    overhead = 33 t/s ceiling **before any compute**. And only if launches
    can overlap, which they can't through GGML's serialized backend graph.
  - The KV cache attention path is unchanged; at 2K context that's already
    ~5–10 ms/token on V100.
- REPORT-15 measured `Config_E4M3` at 59 TF for **M=2048 N=K=7168**. At
  **M=1 N=K=7168** (decode), the kernel is bandwidth-bound and won't reach
  59 TF — closer to the per-tensor memory bandwidth limit, which for FP8 at
  ~700 GB/s effective HBM2 is `7168*7168 / 700e9 ≈ 73 µs per gemv per
  expert`. Multiply by active experts and layers and you get the same
  ~30 ms/token compute time. So even with **infinitely fast launch**, the
  combined kernel time is the floor.
- Honest estimate: **8–12 t/s** is the achievable decode TPS this sprint
  with FP8 hot experts and CPU fallback for cold experts. The intent's
  "4.2× over baseline = 20 t/s" math ignores that we are not moving 100% of
  expert work; we're moving the top-K most-routed ones from a hand-authored
  mock list (~10% of layer-expert pairs).

If the intent insists on 20 t/s as a gate, this sprint will be marked failed
even when correctness, build cleanliness, and a 2× speedup all ship. That's
the wrong incentive. **Recommendation: floor at 8, target 12, document
20 as a multi-sprint stretch.**

### 7.2 GGUF block → turbomind packed: the conversion is the sprint's true cost

The intent treats weight repacking as "a one-time conversion" footnote.
It's not. Concrete frictions:

- GGML `block_f8_e4m3_b128` is `{uint8_t d; uint8_t qs[128]}` per 128
  K-values, stored **row-major in N×K-blocks order** (one tensor row =
  K/128 blocks). Turbomind's `Operand_B_Pack<fp8_e4m3_t>` expects a
  **kernel-friendly tile-packed layout** (`gemm_bench_packed.cu` lines
  240–293 documents the dance: transpose if `conv_w->order == kRowMajor`,
  build `w_desc` with possibly-swapped rows/cols, then `Convert(tmp, w_desc,
  packed, kd, stream)`). That `Convert` call is opaque — its source is in
  `research/lmdeploy/src/turbomind/kernels/gemm/convert.cu` and there is no
  inverse documented. **We cannot debug a misalignment without reading the
  kernel.**
- Scale type mismatch: GGML's `d` is 1 byte (likely E8M0 per the field name
  convention; *confirm by reading `convert.cu :: dequantize_f8_e4m3_b128`
  before writing the packer*). Turbomind `Config_E4M3` expects `kFloat`
  scales per K-group. So we materialize a `(N, K/128)` fp32 scale tensor
  on GPU and feed it through `conv_s->Convert`. That's a second packing
  kernel and a second tolerance source.
- **K alignment**: turbomind tile sizes are powers-of-two; if DSv4 expert
  K is not a multiple of the registered tile K, `Gemm::Run` returns
  non-zero rc. `gemm_bench_packed.cu` only tested N=K=7168 (a clean
  multiple of 128). DSv4-Flash expert shapes are not all square — verify
  every hot-listed tensor's K against the sm70_884 registry before
  packing.
- **Block-vs-row scale stride**: GGML stores the scale **inline** with the
  128 quants. Turbomind expects scales as a **separate `(N, K/G)` matrix**.
  The unpacking kernel must split these out. That's the kernel that's
  easy to write wrong and pass a smoke test while producing junk at
  the 2nd-token GEMM.

### 7.3 Hot-expert *selection* is underspecified and will eat the sprint if left open

The intent §"Open questions" §2 punts on this. **My forcing function:**

- This sprint uses a **manually-authored 16-entry JSON** at
  `models/dsv4-flash-hot.json`. No profile pass. No runtime promotion.
- The selection mechanism for production is a separate sprint
  (SPRINT-024). Without this forcing, we'll spend week 2 on a profile
  harness instead of integration.
- Risk if not forced: someone tries to wire up online expert tracking,
  which requires per-token expert-ID telemetry → atomic counters → host
  readback → new IPC channel → blown sprint.

### 7.4 Build complexity: linking gemm2 + core + parser + cuda_utils + fmt

The intent says "`GGML_TURBOMIND_GEMM=ON` flag pulls in `gemm2` carve-out
and links". Concretely:

- `gemm2` has ~40+ TU's of `.cu` with heavy template instantiations
  (every `Config_*` × every tile size). Adding these to the ggml-cuda
  static archive will **balloon `libggml-cuda.so` by 50–150 MiB** of code
  and bring its link time from ~30 s to several minutes.
- `core` and `cuda_utils` pull in `<fmt/format.h>` which is a header-only
  template behemoth. Every TU that includes a turbomind header now
  instantiates fmt. **In a Debug build this can hit per-TU compile times
  of 60–120 s.**
- `parser` brings yaml + json deps if we're not careful. Audit before
  linking.
- **Mitigation that the intent does not propose**: build the carve-out as
  a **standalone `.so`** with a tiny C ABI (this draft's §3) and have
  ggml-cuda **dlopen it on demand**. This keeps the ggml-cuda backend
  small, keeps the ABI clean (see §7.5), and lets us ship the feature
  without forcing every llama.cpp build to grow.

### 7.5 ABI: turbomind is template-heavy C++; ggml-cuda exports a stable C-style ABI

GGML's backend interface (`ggml-backend.h`) is C-compatible. If we expose
`turbomind::gemm::Gemm` as a non-opaque type through any ggml header, we:

1. Pollute the C ABI with C++ template symbols.
2. Force every consumer of `ggml-cuda.so` to be compiled with the **same
   CUTLASS version, same nv_bfloat16 visibility, same `--expt-relaxed-
   constexpr`** as turbomind. The turbomind_minimal CMakeLists explicitly
   forces `-include cuda_bf16.h` and `--expt-relaxed-constexpr` globally.
   Bleeding this into ggml-cuda's flags is a regression for anyone NOT
   building the carve-out.
3. Break versioning: turbomind has no stable ABI; the next upstream sync
   will likely change template signatures and break consumers.

**Mitigation**: keep all turbomind types **strictly inside
`libggml-turbomind.so`**. The only surface is the four `tm_*` C functions.
The ggml-cuda shim sees `void*` handles and plain numeric/half pointers.

### 7.6 VRAM budget: 22.5 GiB free is OPTIMISTIC

The intent claims 22.5 GiB free for hot expert weights. The math:

- 32494 MiB total
- minus 7500 MiB SPRINT-022 dense baseline (verify — context size assumed)
- minus 1024 MiB context (2K context only; longer = more)
- minus 1500 MiB compute buffers
- = 22.5 GiB

Problems with this math:
- **Compute buffers grow with prefill batch**. At M=128 the workspace for
  `Gemm::Run` partials is `M*N*4*sizeof(float) = 128*7168*16 ≈ 15 MiB` per
  active GEMM; 60 layers × multiple GEMMs in flight via streams could push
  this past 1.5 GiB.
- **KV cache grows with context**. At 2K context, DSv4-Flash KV is
  ~512 MiB. At 4K it's 1 GiB. At 8K it's 2 GiB. The intent's 1024 MiB
  reservation is for the 2K case **only**.
- **Per-hot-tensor we DUPLICATE the weight** (Phase 4 note): original
  GGUF block stays resident so fallback works; turbomind packed copy is
  separate. Net cost per hot tensor at 7168×7168 FP8 ≈ 50 MiB original +
  50 MiB packed + 0.4 MiB scale = ~100 MiB. For 16 hot tensors = 1.6 GiB.
  For 60 layers × 8 hot-experts = 48 GiB — **physically impossible**.
- Allocator fragmentation on V100 with frequent alloc/free is real;
  budget 1 GiB.
- DCGM exporter and other tenants on gpu-01 can hold 200–500 MiB.

**Honest VRAM budget for this sprint**:
- 32 GiB total
- 7.5 GiB dense
- 1.0 GiB context
- 1.5 GiB compute (conservative)
- 1.0 GiB allocator/fragmentation
- 1.5 GiB hot-expert duplication (16 tensors × ~100 MiB)
- = **~19.5 GiB used, ~12.5 GiB free** — fine for 16 hot tensors, **not**
  fine for any scheme that puts hundreds of experts resident.

If anyone proposes moving to "all 60 × 8 = 480 hot experts", the answer is
"no, that's 48 GiB; we'd need a different model or multi-GPU."

### 7.7 The `Convert` call returns rc with no documented error codes

`build_packed_weight()` in `gemm_bench_packed.cu` line 281: if
`conv_w->Convert` returns non-zero, we currently print and skip. **In the
ggml-cuda dispatch path this becomes a load-time correctness silent fail
mode.** Mitigation: convert rc != 0 to a hard `GGML_ABORT` with the
descriptor printed, **never silently fall back to scalar**.

### 7.8 V100 cuBLAS / cuBLASLt may already be in ggml-cuda link line — symbol collisions

`gemm_bench_packed.cu` links `CUDA::cublas`. ggml-cuda also links cublas.
Different CUDA versions / library variants can produce ODR violations at
link or runtime. Phase 1's dlopen approach sidesteps this; do not let the
carve-out get statically merged.

### 7.9 The `M=1` decode case is the worst case for turbomind's mainloop

REPORT-15 measured at M=2048. At M=1, the mainloop is fundamentally
bandwidth-bound, and the registered tile sizes (BM ∈ {8, 16, 32, ...})
do not include M=1. The dispatcher will **round M up to the smallest tile
BM** and waste 87.5–96.9% of compute. **This is fine** (it's still
bandwidth-bound), but: don't expect "59 TF" decode numbers. Expect 5–15
TF effective at M=1 due to padding.

### 7.10 The CONTRIBUTING.md / AGENTS.md gotcha

`AGENTS.md` says llama.cpp does not accept AI-generated PRs. **This sprint
must stay in private fork.** Any documentation, commit message, or PR
description must be **human-authored**. AI assists for mechanical
generation only. None of the artifacts written this sprint should land on
upstream `ggml-org/llama.cpp` without manual rewriting.

### 7.11 nisparks's existing scalar path is the correctness oracle — preserve it

The fallback `vec_dot_*_q8_1` path is the bit-equivalence reference.
Phase 3 must NOT replace the existing case-branch; it must wrap it in
`if (!tm_dispatch_try(...)) { existing_path; }`. A `git diff` that
*removes* the scalar dispatch is a code-review blocker.

---

## 8. Security

Low surface area. Risks:

- `dlopen("libggml-turbomind.so")` — must use `RTLD_NOW | RTLD_LOCAL` and
  search only `LD_LIBRARY_PATH`-resolved paths. Do **not** accept a path
  from an environment variable beyond on/off. If a path-override env var
  is added, it's a local code-execution vector for anyone who can set
  env in the inference process.
- Hot-expert JSON parsing: use a strict JSON parser (no eval, no
  schema-less load). Path is taken from env var
  `GGML_TURBOMIND_HOT_EXPERTS_JSON` — validate the resolved path is under
  the model dir or an explicit allowlist.
- `tm_pack_fp8_b128` reads from a tensor pointer the caller asserts is
  valid GGUF block data. Do not validate the contents; treat any
  in-process buffer as trusted (consistent with GGML's threat model).
- No new network or filesystem write paths.

---

## 9. Dependencies

**Hard**:
- CUDA 12.2 (already on gpu-01).
- gcc 11.4 (already on gpu-01).
- cmake 3.22.1 (already on gpu-01).
- `research/lmdeploy/src/turbomind/` checkout (already there; carve-out
  uses it via `LMDEPLOY_SRC`).
- CUTLASS source tree (already fetched by tc-grid CMake).
- `nvidia::cutlass::cutlass` alias hack (already in
  `turbomind_minimal/CMakeLists.txt`).
- `fmt` 11.0.2 via FetchContent (already wired).

**Soft (not blocking this sprint)**:
- Profile-pass tooling (SPRINT-024).
- nvbench (not needed — we use the `gemm_bench_packed` nvbench-free
  pattern).
- DCGM exporter pause script (REPORT-15 §3.3 protocol; promote
  `/tmp/run_ncu3.sh` to repo as part of D6).

**Prior-art dependencies**:
- `tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu` — **the only
  working integration we have**. Phase 1 is "make this code reusable as a
  library." Read it before writing anything.
- `ggml-cuda/convert.cu :: dequantize_f8_e4m3_b128` — correctness oracle
  for FP8 block scale interpretation. Read before Phase 2.
- `src/llama-memory-deepseek4.{cpp,h}` — referenced by intent but **not
  edited** this sprint.

---

## 10. Open Questions

These are the questions a reviewer should force answers to **before
starting Phase 1**. Don't skip them and discover the answer mid-sprint.

1. **GGML `block_f8_e4m3_b128.d` interpretation**: E8M0 exponent, or
   fp16-scaled-stored-as-u8, or something else? Single source of truth is
   `ggml/src/ggml-cuda/convert.cu` line ~767+ (`dequantize_f8_e4m3_b128`).
   Read it, write the scale-conversion path to match, gate on tolerance
   test. **Answer needed before Phase 2.**
2. **Hot-list scope**: 16 tensors in 1–2 layers, or 16 tensors spread
   across all 60 layers? Smaller blast radius (1–2 layers) makes
   bit-compare easier but limits perf uplift. **Recommendation**: 16
   tensors in layers 5–8 (4 layers × 4 experts), gives meaningful
   end-to-end signal without large VRAM cost.
3. **DSv4-Flash expert K dimension**: confirm every hot-listed tensor's
   K is a multiple of 128 AND of the smallest registered turbomind tile K.
   If not, the tensor is skipped and falls back to scalar. **Answer needed
   before Phase 4.**
4. **Fallback policy when `Gemm::Run` returns rc != 0 at inference time**:
   abort, or silently fall back, or log-and-fall-back? **My
   recommendation: abort with descriptor dump.** Silent fallback masks
   real bugs.
5. **Do we accept that pp128 won't move this sprint?** Prefill on FP8 at
   M=128 is on the bandwidth-bound side of the curve; it'll improve a
   little but the big win is decode. **My recommendation: yes, accept
   pp128 stays in 4–8 range; revisit in SPRINT-024 with `mmq.cu`
   integration.**
6. **NVFP4 explicitly OUT or implicitly OUT?** Intent §"Out of scope"
   lists NVFP4 — **good**. Confirm this means no `GGML_TYPE_NVFP4` case
   is added to `turbomind_dispatch`. Yes; this draft confirms.
7. **WMMA-MMVQ MoE port (SPRINT-022 deferred)**: should it block phase 5
   or run in parallel? **My recommendation: defer to SPRINT-024.** It's
   orthogonal and adds review surface.
8. **Reporting cadence**: per-phase mini-report or single end-of-sprint
   report? **My recommendation: per-phase one-page status note in
   `docs/sprints/drafts/SPRINT-023-PROGRESS.md`**, full report at close.

---

## Appendix A — Why this draft disagrees with the intent's success criteria

| Intent target | This draft floor | This draft stretch | Rationale |
|---|---|---|---|
| `tg32 ≥ 20 t/s` | 8 t/s | 12 t/s | §7.1: kernel-time floor + launch overhead + cold experts staying on CPU |
| `pp128 ≥ 20 t/s` | 6 t/s | 10 t/s | §7.9 + §Question 5: M=128 is still bandwidth-bound for sm70 FP8 |
| Bit-equivalent to scalar (5e-3 tolerance) | 5e-3 M=1, 2e-2 M=128 | same | §Phase 2: packing kernel adds error; widen for M=128 |
| MXFP4 + FP8 + hot-expert | FP8 only, 1 type | — | §7.2: three risks at once is too many |
| VRAM < 30 GiB | < 28 GiB | < 26 GiB | §7.6: real budget accounting |

The intent's targets aren't wrong as aspirations. They're wrong as gates,
because a sprint that gets all the integration plumbing right and ships 2×
faster decode would be marked "failed" against a 4.2× target. **Build the
plumbing right; let the target be honest.**

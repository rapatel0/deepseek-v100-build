---
sprint: 025-patch
title: SPRINT-025-PATCH — Multi-GPU CUDA_TURBOMIND gibberish investigation report
status: FIXED — root cause isolated and patched
date: 2026-05-16
final_commit: 782df824f
---

# SPRINT-025-PATCH — Investigation Report

## 1. Problem statement

8-GPU DSv4-Flash-256e (146 GiB, 256 experts top-6) with `-sm layer` and per-layer `-ot 'pat=CUDA_TURBOMIND<N>'` override produces **incoherent token loops** instead of decoded text. Symptom pattern: `"\(n:? (# # # # # # ..."` — a few partially-coherent tokens then degenerate into a single token (`#`) loop.

Same multi-GPU configuration WITHOUT TURBOMIND override (default cuda buft) decodes coherently (decode 11.35 t/s, output `"if n <= 1: return n else: ..."`).

The TURBOMIND path is supposed to give +13–22% decode TPS (per SPRINT-024 single-GPU MIN-Ne measurements). The multi-GPU integration was broken until the MXFP4 nibble-lane mapping fix described below.

### 2026-05-16 resolution

Root cause: `ggml/vendor/turbomind/ggml-turbomind-deinterleave.cu` unpacked MXFP4 bytes as adjacent K positions:

- old mapping: low nibble -> `k = 2*j`, high nibble -> `k = 2*j + 1`
- actual GGML `block_mxfp4` mapping: low nibble -> `k = j`, high nibble -> `k = j + 16`

That permuted every 32-value MXFP4 block before TurboMind packing, so every routed expert weight was wrong. The existing TurboMind correctness test missed this because its host reference used the same adjacent-nibble interpretation. The grouped-vs-single test also missed it because both paths consumed the same incorrectly packed weights.

Patch:

- `ggml-turbomind-deinterleave.cu`: map MXFP4 high nibbles to `k + QK_MXFP4/2`.
- `test_correctness.cpp`: update MXFP4 host reference to match llama.cpp's `dequantize_row_mxfp4`.

Verification after rebuild in `llamacpp-build-8gpu`:

- `test_ggml_turbomind_correctness ./libggml-turbomind.so`: PASS.
  - MXFP4: `max_abs=1.3622e-02`, `rel=1.8053e-04`.
- `test_ggml_turbomind_grouped_compare ./libggml-turbomind.so`: PASS for DSv4 down/gate/up decode and prompt shapes.
- Full 43-layer `CUDA_TURBOMIND0..7` DSv4-Flash-256e server on 8x V100 now decodes coherently:
  - 96-token Fibonacci probe begins with valid recursive Python and continues with explanatory text instead of the previous `#` loop.
  - Measured probe decode speed: `13.09 tok/s` for the 96-token run.

Remaining separate issue: repeated identical `/completion` requests can still hit the DeepSeek4 server slot/KV reuse bug (`Invalid input batch`, stale sequence position). This is independent of TurboMind packing; the first all-layer TurboMind request after server start is now correct, and the bug reproduces through server slot reuse after a successful generation.

### 2026-05-16 slot/KV follow-up resolution

The remaining DeepSeek4 slot/KV reuse bug was isolated and patched in `src/llama-memory-deepseek4.cpp`.

Root cause: `llama_memory_deepseek4::seq_rm()` only accepted `seq_id == 0` or `seq_id < 0`. The server uses slot ids as sequence ids (`0..3` in the current run), so full-sequence clears for slots 1, 2, and 3 returned `false` and left DeepSeek4 sequence-position metadata stale. On the next request to the same slot, the memory backend still reported the old `seq_pos_max`, while the server submitted the new prompt from position 0. That produced:

```text
init: the tokens of sequence 3 in the input batch have inconsistent sequence positions
llama_decode: failed to decode, ret = -1
Invalid input batch.
```

Patch behavior:

- Normalize `p0 < 0` to 0 and `p1 < 0` to infinity, matching the public memory API contract.
- Support no-op empty/non-intersecting removals.
- Support whole-sequence removal for any valid nonnegative `seq_id`, not only sequence 0.
- Continue returning `false` for true partial removals, because DeepSeek4 memory cannot preserve arbitrary prefixes.

Verification in pod `llamacpp-build-8gpu`:

```bash
kubectl -n llm cp src/llama-memory-deepseek4.cpp \
  llamacpp-build-8gpu:/workspace/llamacpp/src/llama-memory-deepseek4.cpp
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/llamacpp && cmake --build build --target llama-server -j 8'
```

The rebuilt full 43-layer TurboMind server was restarted with:

```bash
export LD_LIBRARY_PATH=/workspace/llamacpp/build/bin:/workspace/llamacpp/ggml/vendor/turbomind/build_so:${LD_LIBRARY_PATH:-}
./llama-server \
  -m /models/DSv4-Flash-256e-fixed.gguf \
  -ngl 99 -sm layer \
  -ot 'blk\.([0-5])\..*exps.*=CUDA_TURBOMIND0,blk\.([6-9]|10)\..*exps.*=CUDA_TURBOMIND1,blk\.(1[1-6])\..*exps.*=CUDA_TURBOMIND2,blk\.(1[7-9]|2[0-2])\..*exps.*=CUDA_TURBOMIND3,blk\.(2[3-7])\..*exps.*=CUDA_TURBOMIND4,blk\.(2[8-9]|3[0-2])\..*exps.*=CUDA_TURBOMIND5,blk\.(3[3-8])\..*exps.*=CUDA_TURBOMIND6,blk\.(39|4[0-2])\..*exps.*=CUDA_TURBOMIND7' \
  -t 8 --port 12500 --no-warmup --cache-ram 0 --slot-prompt-similarity 0.0 -c 4096
```

Slot validation results:

- Explicit `id_slot:3`, same prompt, same request body, two passes: `HTTP 200` twice. This was the direct failing repro before the patch.
- Explicit `id_slot:0..3`, two passes each with `cache_prompt:false`: all eight requests returned `HTTP 200`.
- Default slot selection, four repeated requests with `cache_prompt:false`: slots 0, 1, 2, and 3 each returned `HTTP 200`.
- Log scan after restart found no `Invalid input batch`, no `inconsistent sequence positions`, no `failed to initialize batch`, and no `failed to truncate tokens`.
- The restart used the TurboMind shared library through `LD_LIBRARY_PATH`; no `dlopen(libggml-turbomind.so)` failure or plain-upload fallback appeared in the patched run log.

### 2026-05-16 follow-up amendment

Additional testing after this report narrows the bug further:

- Forced FP32 `D` output in the sm70 TURBOMIND registry and wrapper did **not** fix all-layer correctness. Full 43-layer TURBOMIND still decoded incoherently (`#s%#n##t...`) at ~13.3 tok/s.
- Added `test_ggml_turbomind_grouped_compare`, a DSv4-shape grouped-vs-single TURBOMIND compare. It passes for MXFP4 decode/prompt shapes with 256 sparse experts:
  - down: `N=4096 K=2048`, active=6, tokens/expert=1 and 4
  - gate/up: `N=2048 K=4096`, active=6, tokens/expert=1 and 4
- `GGML_TM_DISABLE_GROUPED=1` was re-tested on the full model and remains broken (`#n#:n#...`), so grouped MoE alignment/scatter is not the primary fault.
- TURBOMIND on only `ffn_down_exps.weight` for all 43 layers is also broken, so the failure is not specific to gate/up orientation.
- Layer 5 alone is coherent. Consecutive layers 0-4 are still token-level coherent but visibly degraded. Consecutive layers 0-5 reproduce the quality cliff (`endend`, syntaxhighlight leakage, broken formatting), confirming a compounding threshold rather than one independently bad layer.
- Current operational configuration: only layers `0,10,22,33` routed through TURBOMIND remains coherent and is running on port `12500` in `llamacpp-build-8gpu`:
  - `-ot "blk\.0\..*exps.*=CUDA_TURBOMIND0,blk\.10\..*exps.*=CUDA_TURBOMIND1,blk\.22\..*exps.*=CUDA_TURBOMIND2,blk\.33\..*exps.*=CUDA_TURBOMIND3"`
  - probe output is coherent recursive Python
  - probe decode speed: ~15.1 tok/s versus ~10.7 tok/s for the no-override baseline in the same rebuilt environment

Conclusion update: this was not MoE route alignment and not FP16 output saturation. It was MXFP4 intra-block nibble alignment in the GGML -> TurboMind pack path.

---

## 2. Bisection matrix (what reproduces the bug)

| Configuration | Result | Decode TPS | Output |
|---|---|---|---|
| Single-GPU AVG-16e + full TURBOMIND override | ✅ matches SPRINT-024 baseline | — | `"i++ i++ i++..."` (averaged-weights fixture degeneracy, same on default cuda) |
| 8-GPU 256e + default cuda buft (no `-ot`) | ✅ coherent | 11.35 t/s | `"if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2"` |
| 8-GPU 256e + 1 TURBOMIND layer (layer 0) | ✅ coherent | 16.91 t/s | proper recursive Python |
| 8-GPU 256e + 2 TURBOMIND layers (0, 10) on 2 devices | ✅ coherent | 16.48 t/s | proper recursive Python |
| 8-GPU 256e + 4 TURBOMIND layers (0, 10, 22, 33) on 4 devices | ✅ coherent | 14.52 t/s | proper recursive Python |
| **8-GPU 256e + 6 TURBOMIND layers (0–5) on 1 device** | **❌ BROKEN** | 8.4 t/s | `"\r if n < 0: return"` truncated |
| **8-GPU 256e + all 43 TURBOMIND layers on 8 devices** | **❌ BROKEN** | 11.4 t/s | `"\(n:? (# # # # ..."` garbage loop |

**Threshold**: bug appears when ≥6 TURBOMIND-routed layers feed each other in one forward pass. The triggering condition spans BOTH single-device (6 layers on one TM buft) AND multi-device (5–6 layers per TM buft × 8 devices).

---

## 3. What was tried — chronological table

| # | Hypothesis | Test design | Result | Limitations / gaps |
|---|---|---|---|---|
| H0 | libggml-turbomind.so kernels broken multi-device | `test_multi_device_simultaneous.cpp`: dispatch on GPU 0 + GPU 1 without intervening sync, compare each to single-device baseline | **PASS**: 0/2048 bytes differ | Test fixture: random F8_E4M3_B128 weights, N=K=256, M=8. Doesn't replicate real model's weight distribution, K=2048, or M=1 decode shape. **Phase D** extended to M=1, N=K=2048 (real production shape) — also PASS bit-identical. |
| H1 | `-ot` regex doesn't match `-sm layer`'s natural device placement | Use same per-layer regex but route to default `CUDA<N>` instead of `CUDA_TURBOMIND<N>` | **PASS**: buffer sizes IDENTICAL to baseline (regex was no-op vs natural placement); decode coherent | Default-cuda path doesn't exercise TURBOMIND kernels, so this only validates regex correctness, not the kernel-side dispatch. |
| H2 | Workspace `barriers` (split-K counters) accumulate stale state across Runs | Added `cudaMemsetAsync(s->d_barriers, 0, ..., stream)` before each Run | **FAIL**: same gibberish | Stream-ordered memset should serialize with the kernel launch. If cudaGraph capture skips memsets in stream-capture mode, fix is silently dropped. Mitigated by H6 below (no effect there either). |
| H3 | Workspace `flags` accumulate stale state | Same as H2 for `s->d_flags` | **FAIL**: same gibberish | Same as H2. |
| H4 | Workspace `partials` (split-K accumulator) accumulate stale state | Same as H2 for `s->d_partials` (256 MiB) | **FAIL**: same gibberish | Same as H2. |
| H5 | Stream race despite same-stream submission | Added `cudaStreamSynchronize(stream)` after `s->gemm->Run` returns | **FAIL**: same gibberish | Brute-force serialization. Eliminates any Run-to-Run async overlap. |
| H6 | cudaGraph capture eats fix-ups (memsets) | `GGML_CUDA_DISABLE_GRAPHS=1` (force every op to re-submit fresh) | **FAIL**: same gibberish | Disables graph capture but doesn't change the underlying compute. Confirms the memset failure isn't a "graph swallowed it" artifact. |
| H7 | Grouped MoE dispatch path specifically (`ggml_cuda_mul_mat_grouped_turbomind`) | `GGML_TM_DISABLE_GROUPED=1` → fall back to per-expert path (`ggml_cuda_mul_mat_turbomind` called once per active expert) | **FAIL**: different gibberish pattern (`"#n#:n# #n# ..."`) but still broken | Per-expert path uses `num_experts=1` per call, exactly the shape Phase C/D verifies. Different broken pattern is informative — bug is shared between both dispatch paths. |
| H8 | Empty-expert kernel handling (kernel mishandles 250 zero-token experts in the offsets array) | Filter to ACTIVE experts only in host code: build `num_active=6` dense offsets `[0,1,2,3,4,5,6]` and per-active strided pointer arrays, send to grouped kernel | **FAIL**: still gibberish (different pattern again) | Eliminates the "kernel sees 250 empty experts" surface. Doesn't eliminate the per-active-expert sparse layout — each expert still has M=1 token (top-6 routing). |
| H9 | Shared `tmg::Gemm` instance accumulates internal cache/tuning state across Runs | `delete s->gemm; s->gemm = new tmg::Gemm();` before each Run (10× decode TPS hit but conclusive) | **FAIL**: same gibberish | Heavyweight test conclusively rules out Gemm-instance accumulation. The cache is shape-keyed and re-derives identically each call. |
| H10 | `ggml_cuda_pool_alloc` address reuse causes stale-data reads for `A_fp16` / `D_fp16` across consecutive calls | Replaced pool with fresh `cudaMallocAsync` + `cudaMemsetAsync` + `cudaFreeAsync` per call | **FAIL**: same gibberish | Forces every call to use a brand-new buffer. Eliminates any pool-reuse race. |
| **Phase C** | Kernel bit-stable across REPEATED sequential Runs on same device State | Extended `test_multi_device_simultaneous.cpp`: 16 sequential Runs on GPU 0, identical input, compare all to Run 0 | **PASS**: 0/2048 differ across all 16 Runs | Toy shape M=8 N=K=256 random weights. **Phase D** extended to real production shape M=1 N=K=2048 — also PASS. Doesn't replicate: real model weight distribution, the full forward pass (norm → attn → moe → ...) wrapping the mul_mat. |
| H11 | CTA_M=8 tile alignment — when total_tokens isn't divisible by 8, kernel reads garbage padding rows into accumulator | top-K probe via `--override-kv expert_used_count`: K=1→random, K=6→garbage, K=8→real-word-loop, K=256→structured-loop. THEN host-side padding: pad A_fp16 to next multiple of 8, zero pad rows, extend last expert's offset, scatter only first `total_routes` rows | **FAIL**: identical pre-fix gibberish | The top-K output variation actually reflects the model going off-distribution at routing K it wasn't trained for, NOT kernel-level correctness. Same kernel bug at all K — just produces differently-degenerate model outputs. |

---

## 4. What was discovered

### Definitive eliminations

| Subsystem | How eliminated |
|---|---|
| libggml-turbomind.so kernel correctness (single-device) | Phase D test PASS at production shape M=1, N=K=2048 |
| libggml-turbomind.so kernel correctness (multi-device simultaneous) | P0 simultaneous-dispatch test PASS |
| libggml-turbomind.so State[N] workspace across repeated Runs | Phase C 16-Run bit-identical test PASS |
| `-ot` regex mismatch with `-sm layer` placement | P1 buffer-size match confirms regex is no-op |
| Workspace state corruption (barriers/flags/partials) | H2/H3/H4 memsets had no effect |
| Stream race | H5 cudaStreamSynchronize had no effect |
| cudaGraph capture | H6 disable had no effect |
| Grouped-vs-per-expert path | H7 both broken |
| Empty-expert kernel handling | H8 active-experts filter (num=6 dense) still broken |
| Shared Gemm-instance state | H9 delete+new per call still broken |
| Pool allocator reuse | H10 fresh cudaMallocAsync still broken |
| CTA_M=8 tile alignment | H11 padding fix had zero observable effect |

### Affirmative findings

1. **Single-GPU TURBOMIND fundamentally works.** AVG-16e single-GPU with `-sm none -ot 'exps=CUDA_TURBOMIND0'` (full TURBOMIND override on all 43 layers) produces the exact SPRINT-024 baseline output. The TURBOMIND code path is sound in isolation.
2. **The bug requires ≥6 TURBOMIND-routed layers feeding each other in one forward pass.** Single layer works (P2). Even 4 layers on 4 different devices works (P3.1). Six layers on one device fails (P3.2). The failure is independent of how layers are distributed across devices.
3. **The kernel produces bit-identical output across repeated calls** with the same input (Phase C / Phase D). Sequential dispatch into the same `State[N]` is correct at the kernel-test level.
4. **The pipeline that exercises the kernel differs from the test** in: a real-model FP32→FP16 conversion of src1 via `get_rows_cuda` (rather than direct host-write of A_fp16), a real-model FP16→FP32 scatter of output via `get_rows_cuda` (rather than direct readback), the kernel call being part of a longer dependency chain (norm → attn → moe routing → mul_mat_id → sum-reduce → residual → ...), and the activations flowing through that chain have specific FP16 numerical characteristics (large dynamic range, near-zero values, etc.).

---

## 5. Superseded hypothesis on the fix

The FP16-drift hypothesis below is retained as investigation history only. The follow-up testing above disproved it: forced FP32 output did not fix the model, no A/D saturation was observed through the captured layers, and the actual fix was the MXFP4 nibble-lane mapping in the pack path.

After all 11 hypotheses tested, **the most parsimonious remaining explanation is FP16 numerical instability compounded across MoE layers** — but specifically a kind that the kernel test fixtures don't excite.

### Concrete hypothesis

The grouped TURBOMIND kernel writes its output as **FP16** (`D_fp16`). This is then converted to FP32 via `get_rows_cuda`'s scatter, summed across active experts (gated by router weights), and added to the residual stream. The FP32 sum-reduce is fine; the **FP16 intermediate has limited dynamic range**.

When MoE layer N's input activations are LARGE (e.g., late layers near LM head where residual norms grow), and the expert weights have certain values, the FP16 matmul output **saturates near FP16 max** (65504) or **underflows to FP16 denormals**. Either case loses precision. After 5+ such saturation-prone layers, the activation chain has drifted enough that the LM-head logits collapse into a degenerate distribution → repeating token argmax.

This explains why:
- Phase C/D PASS: synthetic FP16 inputs in [-0.1, 0.1] range never saturate
- Single-GPU AVG-16e WORKS: averaged weights have small values that don't saturate
- 1–4 TURBOMIND layers WORK: drift hasn't compounded enough
- 6+ layers BREAK: compound drift pushes activations off-distribution
- Default-cuda baseline WORKS: default cuda path may use mixed FP16/FP32 internally, more headroom

### Predicted fix (untested)

Change the kernel's accumulator and output to **FP32 instead of FP16**. Specifically in `ggml/vendor/turbomind/api.cc::ggml_turbomind_mul_mat_grouped`:

```cpp
Ddesc.type = turbomind::kFloat;  // instead of turbomind::kHalf
Cdesc = Ddesc;
```

And the `D_fp16` buffer in `ggml_cuda_mul_mat_grouped_turbomind` becomes `D_fp32`, with the scatter directly producing FP32 dst. This eliminates the FP16 output bottleneck.

Trade-off: 2× memory bandwidth for D, possibly slower decode. But correctness over performance.

### Confidence

Medium-low. The hypothesis fits the bisection (compound drift threshold, weight-distribution dependence implicit in why AVG-16e works) but **isn't directly verified**. Two cheaper validations before committing the FP32 change:

1. **Layer-by-layer dump**: instrument the TURBOMIND and default-cuda paths to write `D_fp16` and `dst` to disk per layer, then compare element-by-element. Find the first layer where they diverge meaningfully. If divergence is small early and compounds → drift hypothesis confirmed. If divergence is large from layer 0 → different bug.

2. **Force FP16 → FP32 conversion EARLIER in the chain**: keep the kernel writing FP16 but cast to FP32 BEFORE the scatter. If drift was the issue, this delays it by a few ops but doesn't fix it. If a specific scatter-path bug, this would fix.

### What would actually take to fix

If the FP32-accumulator hypothesis is right: ~half-day of work in `api.cc` to swap output type and a coordinated change in `ggml-cuda-turbomind.cu` to receive FP32. Rebuild, test.

If wrong: the next investigator should write the layer-by-layer dump tooling (~1 day) and find the actual first-divergence layer.

---

## 6. Repro infrastructure (lasting contribution)

`ggml/vendor/turbomind/test_multi_device_simultaneous.cpp` contains:
- **Phase B**: simultaneous dispatch on 2 GPUs (proves kernel multi-device safety)
- **Phase C**: 16 sequential Runs on one GPU's State (proves kernel bit-stability)
- **Phase D**: same as C but at production shape M=1, N=K=2048

These tests run in <60 seconds (vs 9-minute 256e load) and provide fast bisection for any future investigation.

The 9-test elimination matrix above also gives the next investigator a tight search space: the bug is NOT in any of the 11 places we checked, leaving FP16 drift / layer-by-layer compound effect as the leading suspect.

---

## 7. Critical files for the next investigator

Listed in order from most-to-least-likely-relevant. All paths relative to repo root.

### Tier 1 — the dispatch path the bug lives in

| File | Why it matters | Key sites |
|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` | Top-level dispatch wrapper into libggml-turbomind. Owns pool allocs for `A_fp16` / `D_fp16`, calls `get_rows_cuda` for gather/scatter, caches `extra->weight_ptrs_dev`. **The integration layer where the bug lives** per Phase C/D elimination. | `ggml_cuda_mul_mat_grouped_turbomind` (line 621+), `ggml_cuda_mul_mat_turbomind` (line 462+), `tm_ensure_grouped_ptr_tables` (line 564+) |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` | Public predicates + `ggml_turbomind_tensor_extra` struct (per-tensor cached pointer tables, k_pack, scales) | `ggml_turbomind_tensor_extra` struct, `ggml_cuda_mul_mat_grouped_turbomind` prototype |
| `ggml/vendor/turbomind/api.cc` | C ABI inside libggml-turbomind.so. Wraps `tmg::Gemm::Run`. Per-device `State[32]` (workspace + Gemm). | `ggml_turbomind_mul_mat` (line 441+), `ggml_turbomind_mul_mat_grouped` (line 596+), `ggml_turbomind_pack_weight_expert` (line 233+), `State` struct (line 68+) |

### Tier 2 — the FP16 accumulator hypothesis target

If the FP16-drift hypothesis is correct, the fix is here:

| File | Why it matters | Key sites |
|---|---|---|
| `ggml/vendor/turbomind/api.cc` | `Ddesc.type = turbomind::kHalf` at line ~715 (grouped) and ~593 (single). Changing to `turbomind::kFloat` is the proposed FP32-output fix. | `Ddesc` setup in both `ggml_turbomind_mul_mat_grouped` and `ggml_turbomind_mul_mat` |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` | `ggml_cuda_pool_alloc<half> D_fp16(...)` at line 723. If kernel output becomes FP32, this allocator + the downstream `get_rows_cuda` scatter need to switch type. | `D_fp16` allocation, `get_rows_cuda` scatter at line 748 |

### Tier 3 — kernel internals (only if hypothesis 2 is wrong)

| File | Why it matters | Key sites |
|---|---|---|
| `research/lmdeploy/src/turbomind/kernels/gemm/gemm.cu` | `tmg::Gemm::Run` entry. Picks kernel spec via `impl_->Dispatch`. | `Gemm::Run` (line 247+) |
| `research/lmdeploy/src/turbomind/kernels/gemm/kernel_impl.h` | The `Launch` template that wires (Adesc, Bdesc, ..., workspace) into `gemm_kernel<Gemm><<<grid, block>>>`. `EpilogueParam` setup. `GetWorkspaceSize`. | `Launch` (line 130+), `GetWorkspaceSize` (line 218+) |
| `research/lmdeploy/src/turbomind/kernels/gemm/scheduler_sm70.cuh` | `SchedulerSm70` — `find_group`, `get_group_offset`, `linear_tile_id` math. **Already investigated**: per-expert tile_offset = `((offset/tile_m) + g) << log_unit` gives every expert a tile slot regardless of token count. | `find_group` (line 98), `get_group_offset` (line 85), tile/grid math in constructor (line 56+) |
| `research/lmdeploy/src/turbomind/kernels/gemm/arch/config_sm70_s884.h` | `Config_E4M3<kColMajor, 0>` is the config our path uses. Defines CTA_M=8/16/32/64/128 variants. | `Config_E4M3` template, `Scheduler` typedef line 86 |
| `research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_8.cu` | Registry of CTA_M=8 kernel variants. The smallest CTA_M=8 kernel is what gets selected for M=1 decode. | `Registry::sm70_884_8()` |

### Tier 4 — the ggml dispatch surrounding the bug

| File | Why it matters | Key sites |
|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_mul_mat` (line 2543+) detects CUDA_TURBOMIND buft and forwards to `ggml_cuda_mul_mat_turbomind`. `ggml_cuda_mul_mat_id` (line 2638+) detects CUDA_TURBOMIND buft and forwards to `ggml_cuda_mul_mat_grouped_turbomind` at line 2665. `supports_buft` at line 5415 includes CUDA_TURBOMIND. Graph compute sets `cudaSetDevice(cuda_ctx->device)` at line 4445. | Lines 2543, 2638, 4445, 5099, 5415 |
| `ggml/src/ggml-cuda/getrows.cu` | The gather/scatter used by `ggml_cuda_mul_mat_grouped_turbomind` for FP32→FP16 input gather and FP16→FP32 output scatter. Has type-conversion paths between FP32/FP16/BF16/quantized. | `get_rows_cuda` (line 213), `get_rows_cuda_float` (line 131) — the k_get_rows_float kernel is what runs |

### Tier 5 — test + build infrastructure

| File | Why it matters |
|---|---|
| `ggml/vendor/turbomind/test_multi_device_simultaneous.cpp` | Phase B/C/D bisection harness. <60s repro. Add Phase E for layer-by-layer FP16 drift simulation here. |
| `ggml/vendor/turbomind/CMakeLists.txt` | Builds libggml-turbomind.so + tests. Add new test executables here. |
| `ggml/CMakeLists.txt` | `GGML_TURBOMIND` option gating, `GGML_CUDA_NCCL` option |
| `manifests/llamacpp-build-8gpu.yaml` | k8s pod spec for 8× V100-SXM2-32GB on gpu-01 |

### Tier 6 — model / sprint context

| File | Why it matters |
|---|---|
| `docs/sprints/SPRINT-024-REPORT-18.md` | The +13–22% TURBOMIND TPS measurements; the AVG-16e `"i++ i++"` baseline output is documented here. |
| `docs/sprints/SPRINT-025-REPORT-19.md` | The 8-GPU 256e default-buft measurements (decode 11.35 t/s baseline to compare against). |
| `docs/sprints/SPRINT-025-FOLLOWUPS.md` | §1 family-alias buft, §4 slot KV-position 500 bug (separate from this), §5 multi-GPU TURBOMIND correctness (this work). |

### Where to put debug instrumentation

To validate the FP16-drift hypothesis:

1. In `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu::ggml_cuda_mul_mat_grouped_turbomind` immediately after `g_tm().mul_mat_grouped(...)` succeeds, add a guarded `cudaMemcpyAsync(host_buffer, D_fp16.ptr, size, cudaMemcpyDeviceToHost, stream)` + `cudaStreamSynchronize` + `fprintf` to dump the first few rows × first few cols of `D_fp16` to a debug log keyed by layer index.
2. In a parallel default-cuda run (same prompt), dump the corresponding `dst` values from `ggml_cuda_mul_mat_id`'s scatter at the equivalent site.
3. Compare element-by-element. The first layer where TURBOMIND and default-cuda differ >FP16-ULP is the answer.

The compare can happen offline (Python script reading the two dumps). Total instrumentation effort: ~half-day.

## 8. Cluster usage — pod management & deploy playbook

The investigation uses an on-prem k8s cluster with a single node `gpu-01` hosting 8× V100-SXM2-32GB. All work happens in pods on that node.

### Node & namespace

- Cluster context: default (whatever `kubectl` is set to)
- Namespace: `llm`
- Node selector: `kubernetes.io/hostname: gpu-01` (only node with the V100 GPUs)
- PVC: `llm-models-local` mounted as `/models` (read-only). Contains DSv4-Flash GGUFs.

### GPU inventory & reservation etiquette

```bash
# Who currently has GPUs on gpu-01?
kubectl get pods -n llm -o json | python3 -c "
import json, sys
d = json.load(sys.stdin)
for p in d['items']:
    if p['spec'].get('nodeName') != 'gpu-01': continue
    if p['status'].get('phase') != 'Running': continue
    g = p['spec']['containers'][0].get('resources',{}).get('limits',{}).get('nvidia.com/gpu','0')
    if int(g): print(f\"{p['metadata']['name']}: {g}\")"

# Detailed VRAM per GPU
kubectl exec -n llm <pod> -- nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

**Etiquette:** `tcg-dev` and `llamacpp-build` are long-running 1-GPU dev pods belonging to ongoing tc-grid / build work. **Don't delete them without explicit user authorization.** Your own pods (whatever you create) are fair game.

For the 256e tests you need 8 GPUs. If less are available, you can:
1. Negotiate with the user to delete one of the long-running pods (they've authorized this in past sessions when explicit).
2. Test at 6 GPUs (delete only build pods you created); 256e fits at 24 GiB/GPU with `-c 4096` headroom.
3. Use `CUDA_VISIBLE_DEVICES=0,1,2,3` inside an 8-GPU pod to simulate 4-GPU (cheaper than recreating pods).

### Pod manifests (templated)

Three variants in `manifests/`:

- `manifests/llamacpp-build-4gpu.yaml` (deleted; recreate from 8gpu template if needed)
- `manifests/llamacpp-build-6gpu.yaml`
- `manifests/llamacpp-build-8gpu.yaml`

Template structure (8-GPU example):
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llamacpp-build-8gpu
  namespace: llm
  labels:
    sprint: sprint-025
    role: full-256e-target
spec:
  containers:
  - command: ["sleep", "infinity"]
    image: nvidia/cuda:12.2.2-devel-ubuntu22.04
    name: build
    resources:
      limits:
        nvidia.com/gpu: 8
      requests:
        nvidia.com/gpu: 8
    volumeMounts:
    - mountPath: /workspace
      name: workspace
    - mountPath: /models
      name: models
      readOnly: true
  nodeSelector:
    kubernetes.io/hostname: gpu-01
  restartPolicy: Never
  volumes:
  - emptyDir:
      sizeLimit: 50Gi
    name: workspace
  - name: models
    persistentVolumeClaim:
      claimName: llm-models-local
```

To create or update: `kubectl apply -f manifests/llamacpp-build-8gpu.yaml`
To delete: `kubectl delete pod -n llm llamacpp-build-8gpu`

### Bootstrap script for a fresh build pod

After `kubectl apply`, the pod is bare Ubuntu+CUDA. Bootstrap:

```bash
POD=llamacpp-build-8gpu

# 1. Install build deps + runtime libs
kubectl exec -n llm $POD -- bash -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
    build-essential cmake git curl libssl-dev libcurl4-openssl-dev \
    libnccl2 libnccl-dev pkg-config ca-certificates rsync
'

# 2. Copy source tree (git archive HEAD + manually-tar lmdeploy submodule)
cd /Users/ravi/repos/deepseek
git archive --format=tar HEAD -o /tmp/deepseek-src.tar
tar cf /tmp/lmdeploy.tar -C research/ lmdeploy
kubectl exec -n llm $POD -- mkdir -p /workspace/llamacpp /workspace/research
kubectl cp /tmp/deepseek-src.tar -n llm $POD:/tmp/
kubectl cp /tmp/lmdeploy.tar -n llm $POD:/tmp/
kubectl exec -n llm $POD -- bash -c '
  cd /workspace/llamacpp && tar xf /tmp/deepseek-src.tar
  cd /workspace/research && tar xf /tmp/lmdeploy.tar
  rm -f /tmp/deepseek-src.tar /tmp/lmdeploy.tar
'

# 3. Build llama-server + libggml-turbomind.so
kubectl exec -n llm $POD -- bash -c '
  cd /workspace/llamacpp
  cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DGGML_CUDA_NCCL=ON \
    -DGGML_TURBOMIND=ON \
    -DGGML_TURBOMIND_LMDEPLOY_SRC=/workspace/research/lmdeploy/src \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
  nice -n 5 cmake --build build -j 32 --target llama-server

  cd ggml/vendor/turbomind
  cmake -B build_so \
    -DLMDEPLOY_SRC=/workspace/research/lmdeploy/src \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DGGML_TURBOMIND_TEST=ON
  nice -n 5 cmake --build build_so -j 32 --target ggml-turbomind
  nice -n 5 cmake --build build_so -j 32 --target test_ggml_turbomind_multi_device_simultaneous
'
```

Total cold-start time on the 8-GPU pod: ~10 minutes (5 min apt + 5 min build).

### Iterating fast — incremental rebuild

After editing one file locally, push + rebuild:

```bash
# For api.cc edit:
kubectl cp ggml/vendor/turbomind/api.cc -n llm $POD:/workspace/llamacpp/ggml/vendor/turbomind/api.cc
kubectl exec -n llm $POD -- bash -c '
  cd /workspace/llamacpp/ggml/vendor/turbomind
  nice -n 5 cmake --build build_so -j 32 --target ggml-turbomind
'

# For ggml-cuda-turbomind.cu edit:
kubectl cp ggml/src/ggml-cuda/ggml-cuda-turbomind.cu -n llm $POD:/workspace/llamacpp/ggml/src/ggml-cuda/ggml-cuda-turbomind.cu
kubectl exec -n llm $POD -- bash -c '
  cd /workspace/llamacpp
  nice -n 5 cmake --build build -j 32 --target llama-server
'
```

Incremental rebuild: <60 seconds for either change.

### Launching llama-server

```bash
kubectl exec -n llm $POD -- bash -c '
  export LD_LIBRARY_PATH=/workspace/llamacpp/ggml/vendor/turbomind/build_so:/workspace/llamacpp/build/bin:/usr/lib/x86_64-linux-gnu
  cd /workspace/llamacpp/build/bin

  # The TURBOMIND -ot regex distributes experts across 8 GPUs matching
  # -sm layer natural placement (inferred from buffer sizes).
  OT="blk\.([0-5])\..*exps.*=CUDA_TURBOMIND0,blk\.([6-9]|10)\..*exps.*=CUDA_TURBOMIND1,blk\.(1[1-6])\..*exps.*=CUDA_TURBOMIND2,blk\.(1[7-9]|2[0-2])\..*exps.*=CUDA_TURBOMIND3,blk\.(2[3-7])\..*exps.*=CUDA_TURBOMIND4,blk\.(2[8-9]|3[0-2])\..*exps.*=CUDA_TURBOMIND5,blk\.(3[3-8])\..*exps.*=CUDA_TURBOMIND6,blk\.(39|4[0-2])\..*exps.*=CUDA_TURBOMIND7"

  nohup ./llama-server \
    -m /models/DSv4-Flash-256e-fixed.gguf \
    -ngl 99 -sm layer \
    -ot "$OT" \
    -t 8 --port 12500 \
    --no-warmup \
    -c 4096 \
    > /tmp/server.log 2>&1 &
  echo $! > /tmp/server.pid
'
```

Load time for 256e (146 GiB) into 8× V100: **~9 minutes cold**, ~3 min if page cache hot from a prior load.

### Probing the server

```bash
# 1. Wait for bind (server returns 503 "Loading model" until weights are uploaded)
kubectl exec -n llm $POD -- bash -c '
  while ! curl -sm 2 http://127.0.0.1:12500/health 2>/dev/null | grep -q "ok"; do
    sleep 10
  done
  echo "ready"
'

# 2. Decode probe
kubectl exec -n llm $POD -- bash -c '
  curl -sm 60 http://127.0.0.1:12500/completion \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"def fibonacci(n):\",\"n_predict\":64,\"temperature\":0,\"top_k\":1}" \
    | grep -oE "\"content\":\"[^\"]*\"|\"timings\":\{[^}]*\}"
'
```

**Known prompts and their expected outputs**:

| Prompt | Default-cuda 256e (expected good) | TURBOMIND broken |
|---|---|---|
| `"def fibonacci(n):"` | `"\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2"` | `"\\(n:? (# # # # # # # # # # ..."` |
| `"The capital of France is"` | `" Paris.\nThe capital of France is Paris..."` | (often hits "Invalid input batch" — see FOLLOWUPS §4) |
| AVG-16e single-GPU TURBOMIND `"def fibonacci(n):"` | — | `"i++ i++ i++ i++..."` (this IS the SPRINT-024 baseline; averaged-weights fixture degeneracy) |

### Killing & cleanup

```bash
# Kill server (and any zombie llama-server processes from prior runs)
kubectl exec -n llm $POD -- bash -c 'pkill -9 -f "llama-server -m" 2>&1; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader'

# Sometimes processes go zombie. They don't hold VRAM but show in pgrep. Safe to ignore.

# Free GPUs entirely — delete the pod
kubectl delete pod -n llm $POD --wait=false
```

### Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `error: libcuda.so.1: cannot open shared object file` at build | Build container's libcuda stub vs runtime path | Ensure `LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs` for build only |
| `cmake error: held packages` on apt-install of libnccl | `nvidia/cuda:12.2.2-devel-ubuntu22.04` pins libnccl version | Use `--allow-change-held-packages` |
| NCCL `ncclCommInitAll` fails at server startup | Stale peer mappings from prior crashed process | Wait 30s and retry; or recreate pod |
| `Invalid input batch` HTTP 500 on multi-request testing | Pre-fix DeepSeek4 slot KV-position desync for nonzero slot ids | Fixed by `src/llama-memory-deepseek4.cpp::seq_rm`; rebuild `llama-server` and restart |
| `tar EOF` errors when copying source via `kubectl cp` | kubectl exec stdin/stdout pipes are fragile for >100 MB | Use local intermediate: `kubectl cp local:src/ pod:dst/` — proven reliable |
| Server returns 503 `"Loading model"` indefinitely | Still mmap'ing 146 GiB from PVC | Check `cat /proc/<pid>/io` for `read_bytes` progress; 256e takes 9 min cold |
| 8-GPU pod stuck `Pending` | Insufficient GPUs free on gpu-01 | `kubectl describe pod` shows reason. Delete one of YOUR build pods (not user's tcg-dev/llamacpp-build) to free slots |

### Running the bisection test (fast, no model load needed)

```bash
kubectl exec -n llm $POD -- bash -c '
  cd /workspace/llamacpp/ggml/vendor/turbomind/build_so
  ./test_ggml_turbomind_multi_device_simultaneous
'
```

Runtime: <60 seconds. Output ends with `[simul] PASS` or `[simul] FAIL`. Use this for fast hypothesis testing on changes to `api.cc` or kernel code, before committing to the 9-minute 256e load test.

### Monitoring during a long load

The harness uses `Monitor` (background polling) to detect bind + decode emit. Pattern:

```bash
# Poll until health is "ok", then issue decode
until kubectl exec -n llm $POD -- bash -c 'curl -sm 2 http://127.0.0.1:12500/health 2>/dev/null | grep -q "ok"'; do sleep 20; done
# decode probe here
```

Note that `curl -sf` fails on 503 (which is what /health returns during model load). Use `grep -q "ok"` against the response body instead.

## 9. Sprint disposition

- **SPRINT-025 (8-GPU 256e default cuda)**: SHIPPED at `sprint-025-close`. Decode 11.35 t/s on production-quality output.
- **SPRINT-025-PATCH (multi-GPU TURBOMIND fix)**: FIXED. MXFP4 nibble-lane mapping patched and verified with correctness tests plus full 43-layer server decode.
- **DeepSeek4 slot/KV follow-up**: FIXED. Nonzero slot full-sequence removal now works; repeated slot reuse no longer produces stale-position `Invalid input batch` failures in the full server validation.
- **SPRINT-026 (speculative decoding)**: UNBLOCKED with respect to TurboMind packing and the identified slot/KV reset bug. DeepSeek4 runtime state export/import remains unimplemented and should be treated as a separate feature gap if prompt-state persistence is required.

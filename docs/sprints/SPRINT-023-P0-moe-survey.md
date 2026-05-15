# SPRINT-023 P0.3 — Turbomind grouped MoE primitive survey

## TL;DR

**Yes — and it covers sm70.** Turbomind already exposes a single-launch grouped GEMM that
batches every active expert in one kernel call: `Gemm::Run` with `Operation::batch_dim` set
and `MatrixLayout::{num, offsets, idxs}` populated. `LlamaLinear::Forward(input, weight,
indices, offsets, output)` is the public MoE entry, used directly by
`MoeFfnLayer::Forward`. The sm70 kernel registry registers grouped variants for U4 (AWQ),
MXFP4 (4-bit), E4M3 (FP8) and F16 weight dtypes — all with FP16 activations. There is
**no** registered sm70 INT8-weight grouped kernel (and none registered as a dense kernel
either; the closest 8-bit option is `Config_E4M3` / FP8).

## API surface inventory

| Entry point | File | sm support | Grouped? |
| --- | --- | --- | --- |
| `Gemm::Run(Operation, …, MatrixLayout A, U, B, V, C, D, Workspace, stream)` | `kernels/gemm/gemm.h:23` | sm70 / sm75 / sm80 / sm90 (+ sm100 cuBLAS) | Yes, via `Operation::batch_dim` + `MatrixLayout::{num, offsets, idxs}` |
| `LlamaLinear::Forward(input, weight, output)` | `models/llama/LlamaLinear.h:17` | all archs above | No (dense) |
| `LlamaLinear::Forward(input, weight, indices, offsets, output)` | `models/llama/LlamaLinear.h:21` | all archs above | **Yes — this is the MoE grouped entry** |
| `LinkLinearExperts(get_expert_fn, n, &dst)` | `models/moe_weight.cc:29` | host-side weight-pointer fixup | builds the "batched block view" consumed by grouped GEMM |
| `MakeStridedPtrs` / `MakeBlockedPtrs` | `kernels/gemm/convert.h:30-31`, impl `convert_v3.cu:188 / 223` | host helper | turns `vector<{ptr, stride}>` into the device-side per-expert pointer table |
| `invokeMoeGate_V2` / `invokeMoeGate_NoAuxTC` | `kernels/gemm/moe_utils_v2.h:17 / 65` | device | produces the `offsets[expert_num+1]` prefix-sum + `f2n`/`en2f` indices that feed the grouped GEMM |
| `invokeMoeDispatch` / `invokeMoeCombine` | `kernels/gemm/moe_utils_v2.h:34 / 46` | device | scatter/gather for tokens around the grouped GEMM |

There is **no** separate `GroupedGemm::Run` / `MoeGemm::Run` class. The single class
`turbomind::gemm::Gemm` handles both regimes; selection is driven entirely by whether
`offsets` / `idxs` are non-null and `num > 1` in the `MatrixLayout` descs.

`Gemm::Run` signature (verbatim, `kernels/gemm/gemm.h:23-39`):

```cpp
[[nodiscard]] int Run(const Operation&    operation,
                      float               alpha,
                      const void*         A, const MatrixLayout& Adesc,
                      const void*         U, const MatrixLayout& Udesc,
                      const void*         B, const MatrixLayout& Bdesc,
                      const void*         V, const MatrixLayout& Vdesc,
                      float               beta,
                      const void*         C, const MatrixLayout& Cdesc,
                      void*               D, const MatrixLayout& Ddesc,
                      const Workspace&    workspace,
                      cudaStream_t        stream);
```

Relevant fields (`kernels/gemm/types.h`):

```cpp
struct Operation {
    DispatchPolicy dispatch;
    Epilogue       epilogue;
    QuantDesc      quant_a;
    QuantDesc      quant_b;
    int            batch_dim;   //  0 = group along M (rows of A and D),
                                //  1 = group along N (cols of B and D)
};                              //  types.h:179-186

struct MatrixLayout {
    DataType type;
    Order    order;
    int      rows;
    int      cols;
    int      ld;
    Pack     pack;
    int      num;       // == expert_num for grouped (count of sub-GEMMs)
    int*     offsets;   // device-resident prefix-sum, size num+1
    int*     idxs;      // optional indexed-mode token map (kIndexed)
};                       // types.h:195-205

enum class Striding {
    kFlat,     // [1111,2222,3333]
    kRagged,   // [11,2222222,333]   offsets=[0,2,9,...]
    kIndexed,  // [xx xxxxxxx xxx]   idxs=[01,2345678,9ab]
    kBlocked,  // [11][22222][333]   separate per-expert allocations
};               // types.h:75-81
```

The `Striding` mode is derived from the layout (`types.h:218-227`):

- `idxs != nullptr`                                → `kIndexed`  (A side: per-token gather)
- `ld == 0` or `offsets != nullptr`                → `kBlocked`  (B side: per-expert weight pointers)
- otherwise                                        → `kFlat`     (ordinary dense)

## How `moe_ffn_layer.cc` currently dispatches

`MoeFfnLayer::Forward` (`models/llama/moe_ffn_layer.cc:64-197`) issues **exactly two grouped
GEMM launches per layer** for the fused-w1w3 path, regardless of `expert_num` /
`experts_per_token`:

```cpp
// moe_ffn_layer.cc:171  — gate+up, one launch over all experts
TM_SCOPE_CALL(linear_.Forward(p.input, *block.w1w3, indices, offsets_, inter));

// moe_ffn_layer.cc:178  — down, one launch over all experts
TM_SCOPE_CALL(linear_.Forward(inter.slice({0, 0}, {-1, inter_size}),
                              *block.w2, {}, offsets, temp_));
```

For the unfused path (`moe_ffn_layer.cc:181-191`) it is three launches per layer (w1, w3,
w2). In neither case is there a per-expert loop calling `Gemm::Run` separately.

The token routing pipeline that produces inputs to those grouped calls:

```cpp
// moe_ffn_layer.cc:98-114 (noaux_tc) or :129-144 (V2)
invokeMoeGate_NoAuxTC(f2n_.data(),     // token → flat-expert slot
                       f2E_.data(),     // flat slot → expert id
                       en2f_.data(),    // expert-local → flat slot (for combine)
                       offsets_.data(), // size expert_num+1, prefix-sum of tokens-per-expert
                       scales_.data(),
                       masks_.data(),
                       accum_.data(),
                       logits.data(), …);
```

`offsets_` and the f2n index buffer are then passed straight into
`LlamaLinear::Forward(..., indices=f2n_, offsets=offsets_, ...)`. After the experts run,
`invokeMoeCombine` (`moe_ffn_layer.cc:204`) does the weighted sum back to per-token output.

## Available grouped primitive — full spec

### Function signature

`models/llama/LlamaLinear.h:21-25`:

```cpp
void Forward(const Tensor&       input,    // [tokens, hidden_dim]
             const LinearWeight& weight,   // batched block view from LinkLinearExperts
             const Buffer_<int>& indices,  // f2n: flat slot → token id   (size = experts_per_token * tokens)
             const Buffer_<int>& offsets,  // size = expert_num + 1, monotonic
             Ref<Tensor>         output);
```

Implementation (`LlamaLinear.cu:116-177`) sets `op.batch_dim = 0` (group along M),
populates `desc_A.idxs = indices`, `desc_A.offsets = desc_B.offsets = desc_D.offsets =
offsets`, and `desc_*.num = weight.k_desc.num` (== `expert_num`). Single `gemm_.Run(...)`
call — `LlamaLinear.cu:156`.

### Input contract (shapes, strides, offsets)

For the standard M-axis grouped MoE GEMM (`batch_dim = 0`):

- **A (activations)**: `[total_flat_slots, k]` row-major. With `idxs != nullptr` it is
  consumed in **indexed mode** — the iterator does a per-token gather using
  `idxs[flat_slot]` (the f2n table). `offsets[e+1] - offsets[e]` flat slots fall to expert
  `e`. K is hidden_dim.
- **B (weights)**: blocked layout — `weight.weight` is a device array of `num` pointers
  produced by `MakeStridedPtrs` (one row-stride shared, line 121
  `moe_weight.cc:69`) or `MakeBlockedPtrs` (FP8 path, line 113 `moe_weight.cc:61`). Each
  pointer is `[n, k]` for the per-expert weight slab. `desc_B.num = expert_num`,
  `desc_B.offsets = (int*)1` for blocked-FP8 / per-expert-stride for strided.
- **U / V (scales)**: same shape contract as A / B respectively. For grouped weight-only
  quant, V is per-expert blocked just like B.
- **C / D (output)**: `[total_flat_slots, n]` row-major, kBlocked: `desc_D.num =
  expert_num`, `desc_D.offsets = offsets`. Each expert's output rows are
  `offsets[e] .. offsets[e+1]`, contiguous within the same buffer.
- **`offsets[]` semantic**: monotonically increasing, `offsets[0] = 0`,
  `offsets[expert_num] = total_flat_slots = experts_per_token * tokens`. This is a
  **ragged** layout — no padding between experts, but rows for the same expert *are*
  contiguous (because dispatch reshuffles tokens by `f2n` before the GEMM logically).

### Supported data types per arch (sm70 only — what we care about)

From `kernels/gemm/kernel/sm70_884_{4,8,16}.cu` and
`kernels/gemm/arch/config_sm70_s884.h:130-164`:

| Config | Weight dtype | Activation dtype | Scale dtype | group_axis | Where |
| --- | --- | --- | --- | --- | --- |
| `Config_U4_d<kColMajor>` | `uint4_t` (AWQ U4) | `half` | `uint32_t` | -1 (dense) | `sm70_884_4.cu:18, 55` |
| `Config_U4_g<kColMajor>` | `uint4_t` (AWQ U4) | `half` | `uint32_t` | **0 (grouped)** | `sm70_884_4.cu:39, 82` (hard-coded `0` at `config_sm70_s884.h:128`) |
| `Config_MXF4<kColMajor, 0>` | `fp4_e2m1_t` | `half` | `uint8_t` | **0 (grouped)** | `sm70_884_4.cu:106` |
| `Config_E4M3<kColMajor, 0>` | `fp8_e4m3_t` | `half` | `uint16_t` | **0 (grouped)** | `sm70_884_8.cu:18` |
| `Config_F16<kColMajor, 0>` | `half` | `half` | (VoidOperand) | **0 (grouped)** | `sm70_884_16.cu:18` |

**INT8 weight is not in this list.** There is no `Config_INT8` in the sm70 registry — the
8-bit lane is FP8 e4m3, not INT8. This matches the MEMORY note
`turbomind_no_sm70_int8.md`.

The MMA atom is `SM70_MMA_884` (`config_sm70_s884.h:63`) — i.e. the same `mma.m8n8k4.f16`
instruction the v11 kernel uses. Accumulator output type `Tc = half` for U4_g, MXF4, E4M3
and F16 (`config_sm70_s884.h:114, 126, 138, 150, 161`).

### Scheduler

`SchedulerSm70<order, CTA_M, CTA_N, CTA_K, CHUNK_K, SplitK, group_axis>`
(`kernels/gemm/scheduler_sm70.cuh:15-218`). When `group_axis >= 0`:

- `__host__` ctor (`:74-78`): inflates the grouped-axis tile count upper bound from
  `gemm_shape[group_axis] / tile_shape[group_axis]` plus `gemm_shape[3]` (== num experts).
- `__device__ find_group(...)` (`:98-123`): every CTA does a strided scan across
  `offsets_[g]` to figure out which expert it belongs to, using only `__syncthreads_or` —
  no inter-block sync, no cooperative groups.
- `__device__ init(...)` (`:125-168`): after a successful `find_group`, subtracts
  `base_tile_id` so the CTA sees per-expert local tile coords and the right `shape`
  (clipped to that expert's row count).

So at runtime, a single grid is launched whose Z (or M) extent covers the *sum* of
per-expert tile counts; each CTA self-routes to one expert. There is no host-side
per-expert loop, no per-expert kernel launch.

### File paths

Core sources:
- `research/lmdeploy/src/turbomind/kernels/gemm/gemm.h:14-50` — public class
- `research/lmdeploy/src/turbomind/kernels/gemm/gemm.cu:247-330` — `Gemm::Run` impl
- `research/lmdeploy/src/turbomind/kernels/gemm/types.h:75-227` — Striding, Operation, MatrixLayout
- `research/lmdeploy/src/turbomind/kernels/gemm/context.cu:19-156` — desc & filter, `desc.group_axis = batch_dim` when `num>1` (line 68-70)
- `research/lmdeploy/src/turbomind/kernels/gemm/scheduler_sm70.cuh` — sm70 grouped scheduler
- `research/lmdeploy/src/turbomind/kernels/gemm/iterator_sm70.h:59-60` — `is_indexed = (mode == kIndexed)` plumbed all the way to gmem loads
- `research/lmdeploy/src/turbomind/kernels/gemm/arch/config_sm70_s884.h` — the five Config_* aliases
- `research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`,  `…_8.cu`, `…_16.cu` — registration sites
- `research/lmdeploy/src/turbomind/models/llama/LlamaLinear.cu:67-177` — descriptor wiring
- `research/lmdeploy/src/turbomind/models/llama/moe_ffn_layer.cc:64-197` — caller
- `research/lmdeploy/src/turbomind/models/moe_weight.cc:29-150` — `LinkLinearExperts` + MoeWeight::prepare
- `research/lmdeploy/src/turbomind/kernels/gemm/convert_v3.cu:188-241` — `MakeStridedPtrs` / `MakeBlockedPtrs`
- `research/lmdeploy/src/turbomind/kernels/gemm/test/testbed_v3.h:80-127, 376-455` — reference (per-expert loop) vs production (single grouped call) — direct A/B comparison harness
- `research/lmdeploy/src/turbomind/kernels/gemm/test/gemm_bench.cu:1-87` — nvbench harness, exposes `e_num` and `e_tok` axes

## Implications for SPRINT-023

We do **not** need to design a grouped MoE GEMM from scratch on V100 — turbomind already
ships one for sm70 with the exact `MMA_884` atom v11 uses. What we *don't* have is an
INT8-weight grouped kernel; the registered sm70 8-bit lane is FP8 e4m3 (`Config_E4M3`),
not INT8. That means SPRINT-024's options are: (a) consume the existing sm70 grouped path
with our v11 INT8 dequant fused in as a new `Config_INT8` alias registered through the
same `Sm70_s884` scaffold (cheapest — reuses scheduler, iterator, mainloop, epilogue), or
(b) prove FP8-E4M3 weight ceiling is good enough and just use `Config_E4M3` as-is. Either
way, the dispatch path (`MoeFfnLayer` → `LlamaLinear::Forward(input, weight, indices,
offsets, output)`) and the weight-prep path (`MoeWeight::prepare` →
`LinkLinearExperts` → `MakeStridedPtrs`) are already production-quality; we plug into
them, we don't replace them. The launch-overhead question (P0.4) is now decoupled from
MoE — turbomind never launches per-expert, so empty-kernel overhead × experts_per_token
isn't on the critical path; the only question is single-launch grouped-GEMM occupancy and
the `find_group` scan cost in the sm70 scheduler.

## File index

| Path | Notes |
| --- | --- |
| `research/lmdeploy/src/turbomind/kernels/gemm/gemm.h` | Sole public GEMM class; only one `Run` overload |
| `research/lmdeploy/src/turbomind/kernels/gemm/gemm.cu` | Dispatcher; chooses kernel from cache, calls `spec.kernel->Launch(...)` once |
| `research/lmdeploy/src/turbomind/kernels/gemm/types.h` | Operation::batch_dim, MatrixLayout::num/offsets/idxs, Striding enum |
| `research/lmdeploy/src/turbomind/kernels/gemm/context.cu` | Builds GemmDesc, sets `group_axis = operation.batch_dim` iff `num > 1` |
| `research/lmdeploy/src/turbomind/kernels/gemm/registry.h` / `registry.cu` | Lists arch buckets: sm70_884_{4,8,16}, sm75/80/90, cuBLAS |
| `research/lmdeploy/src/turbomind/kernels/gemm/scheduler_sm70.cuh` | Grouped scheduler — `find_group`, `get_group_offset`, `offsets_` |
| `research/lmdeploy/src/turbomind/kernels/gemm/iterator_sm70.h` | Honors `Striding::kIndexed` for token gather |
| `research/lmdeploy/src/turbomind/kernels/gemm/arch/config_sm70_s884.h` | Templated `Sm70_s884<…, int group_axis>` configs; aliases Config_U4_d / U4_g / MXF4 / E4M3 / F16 |
| `research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu` | Registers U4 (dense + grouped) and MXF4 (grouped) tile sets |
| `research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_8.cu` | Registers E4M3 (grouped) tile set — 5 shapes |
| `research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu` | Registers F16 weight (grouped) — 11 shapes |
| `research/lmdeploy/src/turbomind/kernels/gemm/convert_v3.cu` | Builds per-expert pointer tables on device |
| `research/lmdeploy/src/turbomind/kernels/gemm/moe_utils_v2.h` / `.cu` | Gate, dispatch, combine — produces `offsets[]` and `f2n[]` |
| `research/lmdeploy/src/turbomind/models/llama/LlamaLinear.h` / `.cu` | Public dense and grouped `Forward` overloads |
| `research/lmdeploy/src/turbomind/models/llama/moe_ffn_layer.cc` | Caller: 2-3 grouped GEMM launches per layer total |
| `research/lmdeploy/src/turbomind/models/moe_weight.cc` | `LinkLinearExperts` — turns per-expert weights into one "batched block view" |
| `research/lmdeploy/src/turbomind/kernels/gemm/test/testbed_v3.h` | Reference vs production harness; reference uses a per-expert for-loop, production a single grouped call |
| `research/lmdeploy/src/turbomind/kernels/gemm/test/gemm_bench.cu` | nvbench harness with `e_num`, `e_tok` axes for MoE perf sweeps |

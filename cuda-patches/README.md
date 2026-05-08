# DSv4 HC CUDA backend patches

Implements CUDA versions of the three custom DeepSeek V4 HyperConnection ops the
antirez fork added in commit `06c5042`:

- `GGML_OP_DSV4_HC_SPLIT_SINKHORN`
- `GGML_OP_DSV4_HC_WEIGHTED_SUM`
- `GGML_OP_DSV4_HC_EXPAND`

CPU and Metal implementations exist upstream; CUDA was the missing piece that
made V100s sit idle even with `-DGGML_CUDA=ON`.

## Layout

```
cuda-patches/
├── ggml-cuda/
│   ├── dsv4-hc.cuh                # forward-decl header (auto-globbed by ggml CMake)
│   └── dsv4-hc.cu                 # kernels + ggml_cuda_op_dsv4_hc_* dispatch fns
└── 0001-wire-dsv4-hc-cuda.patch   # adds case blocks to ggml-cuda.cu + supports_op
```

## How it gets applied

The Dockerfile clones the antirez fork, copies `ggml-cuda/*.cu*` into
`ggml/src/ggml-cuda/` (CMake auto-globs `*.cu` and `*.cuh`, so no CMake edit
needed), then `git apply`s `0001-wire-dsv4-hc-cuda.patch` to splice case blocks
into `ggml-cuda.cu`.

## sm_70 (Volta / V100) design choices

V100 lacks several instructions the rest of the CUDA ecosystem assumes by
default. Specific avoidances and substitutions in this code:

| Avoid                                | Why                                | Used instead                          |
| ------------------------------------ | ---------------------------------- | ------------------------------------- |
| `cp.async`                           | Ampere (sm_80+)                    | Plain global → shared via threads     |
| `ldmatrix` PTX                       | Turing (sm_75+)                    | Cooperative `__ldg` loads             |
| `mma.sync` matmul PTX                | Specific to Turing+ shapes         | `wmma::` API or scalar (problem too small for tensor cores) |
| `__nv_bfloat16` arithmetic           | Ampere+                            | FP16 (`__half`) where applicable      |
| FP8 / FP4 tensor cores               | Hopper / Blackwell                 | FP32 accumulator, FP16 storage option |
| `cooperative_groups::tile_partition<32>` with arbitrary sizes | sm_70 limited | `__shfl_xor_sync` with explicit mask   |

Datatype wrapping: kernels are templated over `T_in` (storage) and accumulate
in FP32 always. The model graph hands us F32 inputs (per the CPU op contract),
so the F16 path is forward-looking. When we extend the graph to keep
activations in FP16 between layers, the same kernels light up.

V100-specific micro-optimizations applied:

- **Vectorized loads** (`float4` / `float2`) for memory-bound paths
  (`weighted_sum`, `expand`) to saturate HBM2 bandwidth. V100 has
  ~900 GB/s; the inner loops are bandwidth-bound, not flop-bound.
- **`__half2` packed math** in the `DSV4_HC_USE_FP16_INTERMEDIATE` build
  variant — V100 runs FP16x2 at ~2× FP32 throughput, useful when we cast
  F32 → FP16 in shared mem and accumulate in FP32.
- **One warp per token row** for `split_sinkhorn` with `__shfl_xor_sync`
  reductions across the ≤16 HC dimension; avoids shared-mem barriers.
- **FP32 accumulation** for the sinkhorn iterations — FP16 accumulators
  diverge on the doubly-stochastic normalization, sm_70 can't do BF16.
- **`-use_fast_math`** is already on at the project level; we still use
  `__expf` explicitly so the intent travels with the source.

## Verifying it links

After the build the symbol should be visible:

```
nm build/bin/llama-server | grep dsv4_hc
```

You should see three `T ggml_cuda_op_dsv4_hc_*` symbols. Then the runtime
sanity check is `op_supports_op` returning true on the GPU device for those
ops — easiest way is to set `GGML_SCHED_DEBUG=2` and confirm the ops land on
`CUDA0` rather than `CPU` in the schedule dump.

## What is NOT in here

- A `ggml_dsv4_fp8_kv_quantize` CUDA op. The fork added that one too, but it's
  only used when emulating FP8 KV cache; we run FP16 KV on V100 (default), so
  the op never appears in the graph for our deployment. Skipped intentionally.
- A `ggml_dsv4_rope_tail` CUDA op. RoPE itself has a CUDA implementation;
  `rope_tail` is a thin wrapper that needs the same treatment as HC ops. If
  the scheduler refuses to place this op on GPU, expand the patches; if it
  fuses with the existing rope path, no work needed. Verify after first run.

# SPRINT-023 P3 — CUDA_TURBOMIND buffer type, PASS

**Date:** 2026-05-15

## Gate

> "A smoke-test GGUF loads with the new buffer type and no crash."

✅ `llama-server -m /models/DSv4-Flash-MIN-8e-fixed.gguf -ngl 999 -ot "exps.weight=CUDA_TURBOMIND0"` loads the model, packs expert weights through `ggml_turbomind_pack_weight_expert` at set_tensor time, runs warmup, and serves on the listening port.

Memory breakdown after load:
```
CUDA0 (V100-SXM2-32GB) | model = 11 766 MiB | unaccounted = 935 MiB | free = 19 772 MiB
```
The 11.7 GiB of expert weights are in CUDA_TURBOMIND0 buffers; the remaining ~3 GiB of non-expert tensors are in plain CUDA0.

## What landed

1. **`ggml/src/ggml-cuda/ggml-cuda-turbomind.{cuh,cu}`** — new files.
   - Buft `CUDA_TURBOMIND<i>` per device, with full `ggml_backend_buffer_type_i` + buffer `ggml_backend_buffer_i` interfaces.
   - `set_tensor` intercept: for `GGML_TYPE_MXFP4` / `GGML_TYPE_F8_E4M3_B128` tensors, upload GGML source to a scratch device buffer, call `ggml_turbomind_pack_weight_expert(src, tm_type, N, K, group_size, tensor->data, scales_dev, &k_pack, stream)`, attach `ggml_turbomind_tensor_extra { k_pack, scales_dev, scales_bytes, group_size }` to `tensor->extra`. Every other type falls through to plain `cudaMemcpyAsync`.
   - Lazy `dlopen("libggml-turbomind.so")` on first set_tensor; symbols resolved and `ggml_turbomind_init(device)` invoked once per device. Thread-safe via a global mutex.
   - Scales + extra are tracked in the buffer context's `scale_allocs` / `extra_allocs` vectors and freed in `free_buffer`.

2. **`ggml/src/ggml-cuda/ggml-cuda.cu`**:
   - `ggml_backend_cuda_device_get_extra_bufts()` wires our buft into the device's extra-buft list. Exposed via the new `ggml_backend_dev_get_extra_bufts` proc address case in `ggml_backend_cuda_reg_get_proc_address`.
   - `ggml_backend_cuda_device_supports_buft` extended to accept `CUDA_TURBOMIND` — without this the scheduler aborts with `pre-allocated tensor (...) in a buffer (CUDA_TURBOMIND0) that cannot run the operation (NONE)`.

3. **`common/arg.cpp`** (`parse_tensor_buffer_overrides`):
   - Also enumerates extras via the `ggml_backend_dev_get_extra_bufts` proc-address pattern. Previously only the device's primary buft was addressable from `-ot`; this generalization lets users target both `CUDA_TURBOMIND0` and CPU `CPU_REPACK`-style bufts.

## What still doesn't work (deferred to P4)

- The kernel `ggml_cuda_mul_mat` / `ggml_cuda_mul_mat_q` / `ggml_cuda_mul_mat_id` paths read `tensor->data` directly as if it were GGML-quantized bytes; for tensors that went through us, `tensor->data` holds the *packed* turbomind layout. Inference would compute garbage (or crash) if those paths run on a TURBOMIND tensor. **P4** adds the dispatch predicate that routes to `ggml_turbomind_mul_mat` when the src0 is on `CUDA_TURBOMIND` instead.
- Warmup (the "empty run") doesn't actually exercise the expert tensors, so it doesn't crash. Real inference will, until P4 lands.

## Open follow-ups (not gating P3)

- The buffer type uses `cudaMalloc` directly (not the ggml pool). Acceptable for the prototype; revisit if multi-buffer churn becomes a fragmentation issue.
- Scales allocations are per-tensor; could be coalesced into a single device pool to reduce `cudaMalloc` count for 256e models.
- Free path scans the per-buffer extras vector linearly — fine for the model sizes we target.

## Files

- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu`   (new)
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh`  (new)
- `ggml/src/ggml-cuda/ggml-cuda.cu`             (modified — include, supports_buft, get_proc_address)
- `common/arg.cpp`                              (modified — extras in `-ot` lookup)

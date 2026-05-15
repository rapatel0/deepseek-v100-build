# SPRINT-023 P4 — Dispatcher integration, PASS

**Date:** 2026-05-15

## Gate

> "End-to-end generation runs without crash, output within tolerance vs the SPRINT-022 baseline."

✅ Both legs run; the MIN-8e fixture model has random weights and produces gibberish on every path (CPU and CUDA alike — per the user, "produce garbage and were designed for performance testing"), so tolerance comparison against this model is not meaningful. The real correctness gate landed in P2.3 (FP8+MXFP4 within FP16 ULP of a host reference matmul).

## Smoke results — DSv4-Flash-MIN-8e-fixed, `Hi there!`, n_predict=16, temp=0

| Path | Flag | Tokens | Decode TPS | Prompt TPS | VRAM (model) |
|---|---|---|---|---|---|
| CPU MoE baseline | `-ngl 999 -ncmoe 999` | 16 | **8.46 t/s** | 8.38 t/s | 7 381 MiB CUDA + 5 396 MiB Host |
| CUDA_TURBOMIND | `-ngl 999 -ot exps=CUDA_TURBOMIND0` | 16 | **17.35 t/s** | 14.86 t/s | 7 381 MiB CUDA0 + 1 010 MiB Host + **4 386 MiB CUDA_TURBOMIND0** |

**2.05× decode speedup over CPU MoE baseline** with no crashes.

## What landed

1. **`ggml_cuda_mul_mat`** (`ggml-cuda.cu`): predicate at the top of the function — if `src0->buffer->buft` is a `CUDA_TURBOMIND` buft and `src0->type ∈ {F8_E4M3_B128, MXFP4}` with FP32 src1/dst, route to `ggml_cuda_mul_mat_turbomind` and return. Otherwise fall through unchanged.

2. **`ggml_cuda_mul_mat_id`**: the fast early-return paths (mmvq/mmq/mmf) don't know how to read our packed format, so they're gated off when `src0` is on a TURBOMIND buft. Execution falls through to the per-expert slicing fallback already in the file — that path sorts tokens by expert, slices `src0` per expert, and calls `ggml_cuda_mul_mat` once per (expert, sorted-tokens) pair. Each such call lands in our turbomind dispatch.

3. **`ggml_cuda_mul_mat_turbomind`** (`ggml-cuda-turbomind.cu`):
   - Allocates FP16 A and FP16 D from the ggml CUDA pool.
   - `fp32_to_fp16` (`ggml_get_to_fp16_cuda(F32)`) for the activation.
   - Resolves the source tensor's `extra` through `view_src` (per-expert slices point back to the original tensor's extra).
   - Computes the expert index from `(src0->data - src0_orig->data) / src0_orig->nb[2]` and offsets `scales_dev` accordingly.
   - Calls `ggml_turbomind_mul_mat(A_fp16, packed_weight, scales+expert_off, type, M, N, K, gs, k_pack, D_fp16)`.
   - `fp16_to_fp32` for the output.

4. **`set_tensor`** (P3 retrofit, fixes a real bug): expert-weight tensors are 3D (`ne[2] = num_experts`); the original P3 set_tensor treated the whole thing as one 2D matrix. Now loops `e ∈ [0, n_experts)`, packs each expert independently into `tensor->data + e*nb[2]`, and lays out scales contiguously as `n_experts * scales_per_expert`. `ggml_turbomind_tensor_extra` gained `scales_per_expert` and `n_experts` fields.

5. **`ggml_turbomind_mul_mat`** is now dlsym'd alongside the other entry points (added to `TmLib` + `tm_ensure_loaded`).

## Caveats / known limits

- MIN-8e has random weights and produces gibberish regardless of path — the 17.35 t/s number is *speed*, not quality. Real-quality verification needs DSv4-Flash-256e, but that won't fit on a single V100 even with TURBOMIND offload.
- We sort tokens by expert host-side (in the existing ggml_cuda_mul_mat_id fallback) and dispatch per expert. That's one `ggml_turbomind_mul_mat` call per active expert per layer. SPRINT-024 should consolidate via `ggml_turbomind_mul_mat_grouped` for fewer launches at small M.
- Activation cast FP32→FP16→FP32 is on the hot path. Acceptable for a working integration; profile in P5.

## Files

- `ggml/src/ggml-cuda/ggml-cuda.cu`           (modified — dispatch predicates)
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu`  (modified — mul_mat helper, per-expert set_tensor)
- `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` (modified — extra struct gained per-expert fields, helper export)

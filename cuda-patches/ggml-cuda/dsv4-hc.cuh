// CUDA implementation of DeepSeek V4 HyperConnection ops.
//
// Op signatures match ggml_dsv4_hc_* declared in ggml/include/ggml.h.
// CPU reference: ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_dsv4_hc_*
// Metal reference: ggml/src/ggml-metal/ggml-metal.metal::kernel_dsv4_hc_*
//
// Tuned for sm_70 (V100). See ../README.md for the design notes.

#pragma once

#include "common.cuh"

// Splits [(2 + n_hc) * n_hc, n_rows] mix tensor into pre/post sigmoid blocks
// and a Sinkhorn-normalized n_hc x n_hc combinator. F32 in, F32 out.
//
// op_params: [0]=n_hc (i32), [1]=sinkhorn_iters (i32), [2]=eps (f32)
// src[0] = mixes  [(2 + n_hc) * n_hc, n_rows] f32
// src[1] = scale  [3]                          f32 (pre, post, comb)
// src[2] = base   [(2 + n_hc) * n_hc]          f32
void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// y[d, t] = sum_{h=0..n_hc} x[d, h, t] * w[h, t]
//
// src[0] = x        [n_embd, n_hc, n_tokens] f32
// src[1] = weights  [n_hc, n_tokens]         f32
// dst    =          [n_embd, n_tokens]       f32
void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// y[d, h, t] = block_out[d, t] * post[h, t]
//            + sum_{s=0..n_hc} comb[h, s, t] * residual[d, s, t]
//
// src[0] = block_out  [n_embd, n_tokens]              f32
// src[1] = residual   [n_embd, n_hc, n_tokens]        f32
// src[2] = post       [n_hc, n_tokens]                f32
// src[3] = comb       [n_hc, n_hc, n_tokens]          f32
// dst    =            [n_embd, n_hc, n_tokens]        f32
void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

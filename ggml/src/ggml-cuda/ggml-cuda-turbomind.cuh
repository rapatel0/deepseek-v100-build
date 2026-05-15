// SPRINT-023 P3 — CUDA_TURBOMIND buffer type.
//
// A specialized cuda buffer type that intercepts set_tensor for
// MXFP4 / F8_E4M3_B128 weights and runs them through turbomind's
// pack_weight pipeline (libggml-turbomind.so via dlopen). For any
// other tensor type the buffer acts like a plain CUDA device buffer.
//
// Surface:
//   ggml_backend_cuda_turbomind_buffer_type(device)        // the buft
//   ggml_backend_cuda_turbomind_get_extra_bufts(device)    // for get_proc_address
//
// Tensor extra (populated by set_tensor for packed types):
//   struct ggml_turbomind_tensor_extra {
//       int    k_pack;       // encoded {b_pack[0:12], v_pack[12:24]}
//       void*  scales_dev;   // packed scale buffer on device
//       size_t scales_bytes; // for accounting / free
//       int    group_size;   // 128 for FP8, 32 for MXFP4
//   };

#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// For MoE expert-weight tensors (ne[2] = num_experts), set_tensor packs each
// expert independently into the same tensor->data buffer using stride
// src0->nb[2] (the per-expert byte stride from the original GGML layout —
// the packed format is ≤ GGML format for our types, so it fits in that
// slot). Scales are allocated as one contiguous buffer of size
// num_experts * scales_per_expert; per-expert offset = i02 *
// scales_per_expert. For a non-MoE tensor (ne[2]==1) this degenerates.
struct ggml_turbomind_tensor_extra {
    int    k_pack;
    void * scales_dev;
    size_t scales_bytes;         // total across all experts
    size_t scales_per_expert;    // step between experts in scales_dev
    int    group_size;
    int    n_experts;

    // SPRINT-024 P1.3 — cached for grouped MoE dispatch.
    // Populated lazily on first grouped call (not in set_tensor: at upload
    // time we don't yet know the tensor's final ggml_type, only the buft
    // does. Resolved + cached the first time the grouped dispatch sees this
    // tensor). Freed in free_buffer.
    void * weight_ptrs_dev;      // device StridedPtr[n_experts]
    void * scale_ptrs_dev;       // device StridedPtr[n_experts]
    int    packed_b_ld;          // converter-derived B leading dim
    int    packed_v_ld;          // converter-derived V leading dim
};

// Buft for a specific CUDA device. The buft holds device-local state
// (a libggml-turbomind.so handle + function pointers, lazily loaded).
ggml_backend_buffer_type_t ggml_backend_cuda_turbomind_buffer_type(int device);

// NULL-terminated array of extra bufts for this device — wired into the
// ggml_backend_dev_get_extra_bufts proc address surface so llama-model
// and -ot can find us.
ggml_backend_buffer_type_t * ggml_backend_cuda_turbomind_get_extra_bufts(int device);

// True iff buft is one of ours (a CUDA_TURBOMIND buft). Cheap pointer
// comparison against the singletons returned by the function above.
bool ggml_backend_buft_is_cuda_turbomind(ggml_backend_buffer_type_t buft);

#ifdef __cplusplus
}

// ---------------------------------------------------------------------------
// P4 dispatch helper. Called from ggml_cuda_mul_mat when src0 is in a
// CUDA_TURBOMIND buffer. src1 must be FP32; dst must be FP32. Internally
// converts to/from FP16 around ggml_turbomind_mul_mat.
struct ggml_backend_cuda_context;
void ggml_cuda_mul_mat_turbomind(ggml_backend_cuda_context & ctx,
                                 const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 struct ggml_tensor * dst);

// SPRINT-024 P1.5 — grouped MoE dispatch. Replaces the per-expert slicing
// path in ggml_cuda_mul_mat_id when src0 is on a CUDA_TURBOMIND buffer.
// One ggml_turbomind_mul_mat_grouped launch per MoE-linear, amortizing
// the per-expert launch cost across the active experts for the layer.
//
// Layout: src0 = expert weights [K, N, n_experts]; src1 = activations
// [K, n_tokens] (FP32 row-major); ids = routing [n_expert_used, n_tokens]
// (int32); dst = output [N, n_tokens, n_expert_used] (FP32 row-major).
void ggml_cuda_mul_mat_grouped_turbomind(ggml_backend_cuda_context & ctx,
                                          const struct ggml_tensor * src0,
                                          const struct ggml_tensor * src1,
                                          const struct ggml_tensor * ids,
                                          struct ggml_tensor * dst);
#endif

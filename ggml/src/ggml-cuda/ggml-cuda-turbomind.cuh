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

struct ggml_turbomind_tensor_extra {
    int    k_pack;
    void * scales_dev;
    size_t scales_bytes;
    int    group_size;
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
#endif

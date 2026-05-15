// SPRINT-023 P1.2 — Public C ABI for libggml-turbomind.so
//
// This header is the ONLY public surface of libggml-turbomind.so. It exposes
// a stable C ABI suitable for dlopen by libggml-cuda.so. All turbomind
// internals (CUTLASS, fmt, gemm2 / core / kernels) are hidden behind these
// entry points.
//
// Usage pattern from libggml-cuda.so:
//
//   void* h = dlopen("libggml-turbomind.so", RTLD_LAZY | RTLD_LOCAL);
//   auto init     = (decltype(&ggml_turbomind_init))     dlsym(h, "ggml_turbomind_init");
//   auto packed_b = (decltype(&ggml_turbomind_packed_bytes)) dlsym(h, "ggml_turbomind_packed_bytes");
//   auto pack_e   = (decltype(&ggml_turbomind_pack_weight_expert)) dlsym(h, "...");
//   auto mul_mat  = (decltype(&ggml_turbomind_mul_mat_grouped))    dlsym(h, "...");
//   auto shutdown = (decltype(&ggml_turbomind_shutdown)) dlsym(h, "...");
//
// All entry points return int rc: 0 = success, non-zero = error code.
// The C ABI is grouped-MoE-aware per SPRINT-023 P0.3 finding: turbomind's
// LlamaLinear::Forward dispatches one launch per layer covering all top-k
// active experts via the offsets[]/indices[] ragged-batch contract.

#ifndef GGML_TURBOMIND_API_H
#define GGML_TURBOMIND_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type enum mirroring GGML types we support. Stable values across versions.
// ============================================================================

enum ggml_turbomind_dtype {
    // Activation / output dtypes
    GGML_TM_DTYPE_FP16          = 0,

    // Weight dtypes supported on sm70
    GGML_TM_DTYPE_F8_E4M3_B128  = 1,  // 1 byte/wt, E8M0 scale per 128 K
    GGML_TM_DTYPE_MXFP4         = 2,  // 0.5 byte/wt, E8M0 scale per 32 K
    GGML_TM_DTYPE_U4_G          = 3,  // U4 grouped — sm70 has a V-operand
                                      // convert bug as of this writing; do
                                      // not enable in production paths
};

// ============================================================================
// Versioning
// ============================================================================

#define GGML_TURBOMIND_API_VERSION 1

/**
 * Returns the ABI version this library implements.
 * libggml-cuda.so checks this matches GGML_TURBOMIND_API_VERSION before
 * using any other entry point.
 */
int ggml_turbomind_api_version(void);

// ============================================================================
// Library lifecycle
// ============================================================================

/**
 * Initialize the turbomind runtime on the specified CUDA device.
 * Must be called once before any other entry point. Idempotent if called
 * multiple times with the same device.
 *
 * @param cuda_device  CUDA device ordinal (typically 0)
 * @return 0 on success, non-zero on error
 */
int ggml_turbomind_init(int cuda_device);

/**
 * Tear down the turbomind runtime. After this returns, the only valid
 * call is ggml_turbomind_init() again.
 */
void ggml_turbomind_shutdown(void);

// ============================================================================
// Weight packing (called once per expert tensor at upload time)
// ============================================================================

/**
 * Return the number of bytes needed to store the packed weight + scales for
 * one expert tensor in turbomind's expected layout.
 *
 * @param ggml_type   one of GGML_TM_DTYPE_*
 * @param N           output_dim of the expert weight (e.g. 7168 for w1)
 * @param K           input_dim of the expert weight
 * @param group_size  scale group size (32 for MXFP4, 128 for F8_E4M3_B128)
 * @param weight_out_bytes  OUT: bytes needed for packed weight buffer
 * @param scales_out_bytes  OUT: bytes needed for packed scales buffer
 * @return 0 on success, non-zero if (ggml_type, N, K, group_size) is unsupported
 */
int ggml_turbomind_packed_bytes(
    int       ggml_type,
    int       N,
    int       K,
    int       group_size,
    size_t*   weight_out_bytes,
    size_t*   scales_out_bytes);

/**
 * Pack one expert tensor from its GGML layout into turbomind's packed
 * device layout. Called per-tensor at model load.
 *
 * Memory layout of src follows GGML conventions for the specified type:
 *   F8_E4M3_B128: blocks of {uint8 scale, uint8 qs[128]} in row-major order
 *   MXFP4:        blocks of {uint8 scale, uint8 qs[16]} where each qs byte
 *                 packs two E2M1 fp4 values along K
 *
 * @param src             device pointer to GGML-formatted weight blocks
 * @param ggml_type       one of GGML_TM_DTYPE_*
 * @param N, K            logical dims
 * @param group_size      32 (MXFP4) or 128 (F8_E4M3_B128)
 * @param weight_out      device pointer to pre-allocated packed weight buffer
 * @param scales_out      device pointer to pre-allocated packed scales buffer
 * @param k_pack_out      OUT: opaque value to pass to mul_mat (encodes the
 *                        Pack flag for MatrixLayout reconstruction)
 * @param stream          CUDA stream
 * @return 0 on success, non-zero on error
 */
int ggml_turbomind_pack_weight_expert(
    const void*   src,
    int           ggml_type,
    int           N,
    int           K,
    int           group_size,
    void*         weight_out,
    void*         scales_out,
    int*          k_pack_out,
    void*         stream  /* cudaStream_t */);

// ============================================================================
// Grouped MoE matmul (one call per MoE layer; processes all top-k experts)
// ============================================================================

/**
 * Execute a grouped GEMM that covers all active experts for one MoE layer.
 * Internally this wraps turbomind's LlamaLinear-style grouped dispatch:
 * one kernel launch handles the ragged-batch of tokens routed to each
 * expert, using offsets[] for the per-expert sub-tile boundaries.
 *
 * The input A is in standard row-major FP16 with shape [total_tokens, K].
 * Token-to-expert routing is encoded by the offsets[] array: tokens
 * [offsets[i], offsets[i+1]) are routed to expert i. Pass num_experts+1
 * elements (last is total_tokens).
 *
 * weights_packed[] and scales_packed[] are arrays of num_experts device
 * pointers, each produced by ggml_turbomind_pack_weight_expert(). The
 * k_pack_value must match what packing emitted.
 *
 * @param A                device fp16 activations [total_tokens, K]
 * @param token_indices    optional device int* — original token positions
 *                         for gathering activations. Pass NULL if A is
 *                         already pre-gathered (one row per active route).
 * @param expert_offsets   device int* of length num_experts+1: prefix sum
 *                         of tokens per expert
 * @param num_experts      number of experts in this layer
 * @param weights_packed   device const void** of length num_experts
 * @param scales_packed    device const void** of length num_experts; may
 *                         be NULL if the type has no scales (FP16)
 * @param ggml_type        weight dtype (must match what was packed)
 * @param N, K             logical dims of each per-expert weight
 * @param group_size       scale group size
 * @param k_pack_value     opaque value from packing
 * @param D                device fp16 output [total_tokens, N]
 * @param stream           CUDA stream
 * @return 0 on success, non-zero on error
 */
int ggml_turbomind_mul_mat_grouped(
    const void*        A,
    const int*         token_indices,    // may be NULL
    const int*         expert_offsets,
    int                num_experts,
    const void* const* weights_packed,
    const void* const* scales_packed,    // may be NULL for FP16
    int                ggml_type,
    int                N,
    int                K,
    int                group_size,
    int                k_pack_value,
    void*              D,
    void*              stream  /* cudaStream_t */);

// ============================================================================
// Single-expert mul-mat (convenience for non-grouped cases like dense layers)
// ============================================================================

/**
 * Execute a single mul-mat for a non-MoE tensor (e.g. dense projection,
 * attention QKV, output head). Convenience wrapper around the same
 * Gemm::Run as the grouped path, without the routing machinery.
 *
 * @param A             device fp16 [M, K]
 * @param B_packed      device packed weight (single expert / tensor)
 * @param V_packed      device packed scales; NULL for FP16
 * @param ggml_type     weight dtype
 * @param M, N, K       dims
 * @param group_size    scale group size
 * @param k_pack_value  opaque value from packing
 * @param D             device fp16 [M, N]
 * @param stream        CUDA stream
 * @return 0 on success, non-zero on error
 */
int ggml_turbomind_mul_mat(
    const void* A,
    const void* B_packed,
    const void* V_packed,
    int         ggml_type,
    int         M,
    int         N,
    int         K,
    int         group_size,
    int         k_pack_value,
    void*       D,
    void*       stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GGML_TURBOMIND_API_H

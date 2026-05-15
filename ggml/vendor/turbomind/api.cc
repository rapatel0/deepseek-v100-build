// SPRINT-023 P1.3 — Implementation of ggml-turbomind C ABI.
//
// Wraps turbomind's gemm2 + core libraries (gemm::Gemm, gemm::Convert) to
// expose the 6-function C ABI declared in ggml-turbomind-api.h.
//
// All entry points have C linkage and __attribute__((visibility("default"))).
// Everything else in this TU is internal to the .so.
//
// Conventions / contracts:
// - We require the GGML weight to already have been UPLOADED to the device
//   in its native block-quantized layout. Packing reads device→device.
// - Output device buffer sizing must match what ggml_turbomind_packed_bytes
//   reported. We don't allocate.
// - The opaque "k_pack_value" returned by pack_weight_expert encodes the
//   turbomind Pack flag so mul_mat can reconstruct the MatrixLayout.
//
// Internal layout / mapping decisions:
// - F8_E4M3_B128 → turbomind Config_E4M3 (Operand_B_Pack<fp8_e4m3_t>,
//   Operand_V_Pack<uint16_t> for FP16-acted scales)
// - MXFP4        → turbomind Config_MXF4 (Operand_B_Pack<fp4_e2m1_t>,
//   Operand_V_Pack<uint8_t>)
// - U4_G         → turbomind Config_U4_g; included for completeness, has a
//   known V-operand convert bug on sm70 — caller should avoid

#define GGML_TURBOMIND_API_INTERNAL
#include "ggml-turbomind-api.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <mutex>

#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/core/data_type.h"

namespace tmg = turbomind::gemm;

// ============================================================================
// Visibility helper
// ============================================================================
#define GGML_TM_EXPORT __attribute__((visibility("default")))

// ============================================================================
// Module-level state
// ============================================================================
namespace {

struct State {
    bool                 initialized = false;
    int                  device      = -1;
    tmg::Gemm*           gemm        = nullptr;
    void*                d_barriers  = nullptr;
    void*                d_partials  = nullptr;
    int*                 d_flags     = nullptr;
    size_t               partials_size = 0;
    std::mutex           mtx;
};

State g_state;

inline turbomind::DataType to_tm_wdtype(int ggml_type) {
    switch (ggml_type) {
        case GGML_TM_DTYPE_FP16:         return turbomind::kHalf;
        case GGML_TM_DTYPE_F8_E4M3_B128: return turbomind::kFloat8_e4m3;
        case GGML_TM_DTYPE_MXFP4:        return turbomind::kFloat4_e2m1;
        case GGML_TM_DTYPE_U4_G:         return turbomind::kUint4;
        default:                         return turbomind::kHalf;  // sentinel
    }
}

inline turbomind::DataType to_tm_sdtype(int ggml_type) {
    // Mirrors core/data_format.cc::ResolveLinearWeightFormat scale dtypes.
    switch (ggml_type) {
        case GGML_TM_DTYPE_F8_E4M3_B128: return turbomind::kFloat;  // FP32 scale
        case GGML_TM_DTYPE_MXFP4:        return turbomind::kUint8;  // E8M0 byte
        case GGML_TM_DTYPE_U4_G:         return turbomind::kHalf;
        default:                         return turbomind::kHalf;
    }
}

inline int bits_per_weight(int ggml_type) {
    switch (ggml_type) {
        case GGML_TM_DTYPE_FP16:         return 16;
        case GGML_TM_DTYPE_F8_E4M3_B128: return 8;
        case GGML_TM_DTYPE_MXFP4:        return 4;
        case GGML_TM_DTYPE_U4_G:         return 4;
        default:                         return 0;
    }
}

// Encode the relevant turbomind Pack value into a single int we hand back
// to the caller. We round-trip it through the int* k_pack_out parameter so
// callers can pass it back at mul-mat time without storing turbomind types.
inline int encode_pack(uint32_t p) { return (int)p; }
inline tmg::Pack decode_pack(int p) { return (tmg::Pack)(uint32_t)p; }

}  // namespace

// ============================================================================
// API: versioning
// ============================================================================
extern "C" GGML_TM_EXPORT int ggml_turbomind_api_version(void) {
    return GGML_TURBOMIND_API_VERSION;
}

// ============================================================================
// API: lifecycle
// ============================================================================
extern "C" GGML_TM_EXPORT int ggml_turbomind_init(int cuda_device) {
    std::lock_guard<std::mutex> lk(g_state.mtx);
    if (g_state.initialized && g_state.device == cuda_device) return 0;
    if (g_state.initialized) {
        // Re-init on different device — tear down first.
        // Reset fields individually (std::mutex is non-movable).
        delete g_state.gemm;
        g_state.gemm = nullptr;
        cudaFree(g_state.d_barriers); g_state.d_barriers = nullptr;
        cudaFree(g_state.d_partials); g_state.d_partials = nullptr;
        cudaFree(g_state.d_flags);    g_state.d_flags    = nullptr;
        g_state.partials_size = 0;
        g_state.initialized   = false;
        g_state.device        = -1;
    }
    cudaError_t err = cudaSetDevice(cuda_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ggml-turbomind] cudaSetDevice(%d) failed: %s\n",
                cuda_device, cudaGetErrorString(err));
        return 1;
    }
    g_state.gemm = new tmg::Gemm();
    // Allocate scratch buffers used by all dispatched kernels.
    g_state.partials_size = (size_t) 4096 * 4096 * sizeof(float) * 4;
    if (cudaMalloc(&g_state.d_barriers, tmg::Gemm::kBarriersSize) != cudaSuccess ||
        cudaMalloc(&g_state.d_partials, g_state.partials_size)    != cudaSuccess ||
        cudaMalloc(&g_state.d_flags,    sizeof(int) * 1024)       != cudaSuccess) {
        fprintf(stderr, "[ggml-turbomind] failed to allocate workspace buffers\n");
        return 2;
    }
    g_state.device      = cuda_device;
    g_state.initialized = true;
    return 0;
}

extern "C" GGML_TM_EXPORT void ggml_turbomind_shutdown(void) {
    std::lock_guard<std::mutex> lk(g_state.mtx);
    if (!g_state.initialized) return;
    delete g_state.gemm;
    g_state.gemm = nullptr;
    cudaFree(g_state.d_barriers); g_state.d_barriers = nullptr;
    cudaFree(g_state.d_partials); g_state.d_partials = nullptr;
    cudaFree(g_state.d_flags);    g_state.d_flags    = nullptr;
    g_state.partials_size = 0;
    g_state.initialized   = false;
    g_state.device        = -1;
}

// ============================================================================
// API: packed bytes
// ============================================================================
extern "C" GGML_TM_EXPORT int ggml_turbomind_packed_bytes(
    int       ggml_type,
    int       N,
    int       K,
    int       group_size,
    size_t*   weight_out_bytes,
    size_t*   scales_out_bytes)
{
    if (!weight_out_bytes || !scales_out_bytes) return 1;
    if (N <= 0 || K <= 0 || group_size <= 0)    return 2;

    const int bits = bits_per_weight(ggml_type);
    if (bits == 0) return 3;

    const size_t n_elem = (size_t)N * (size_t)K;
    *weight_out_bytes = turbomind::byte_size(to_tm_wdtype(ggml_type), n_elem);

    if (ggml_type == GGML_TM_DTYPE_FP16) {
        *scales_out_bytes = 0;  // no scales for FP16
    } else {
        if (K % group_size != 0) return 4;
        const int n_scales = N * (K / group_size);
        *scales_out_bytes = turbomind::byte_size(to_tm_sdtype(ggml_type), n_scales);
    }
    return 0;
}

// ============================================================================
// API: pack_weight_expert
// ============================================================================
//
// This mirrors the build_packed_weight() helper in
// tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu, which is the only
// proven-working invocation pattern for sm70 packed-weight kernels.
//
// Important: the GGML weight source is in its NATIVE block layout. We need
// to first "expand" sub-byte values to u16, then transpose if conv_w wants
// row-major source, then call conv_w->Convert into the packed output.

extern "C" GGML_TM_EXPORT int ggml_turbomind_pack_weight_expert(
    const void*   src,
    int           ggml_type,
    int           N,
    int           K,
    int           group_size,
    void*         weight_out,
    void*         scales_out,
    int*          k_pack_out,
    void*         stream_v)
{
    if (!g_state.initialized) return 100;
    if (!src || !weight_out)  return 1;
    if (!k_pack_out)          return 2;
    cudaStream_t stream = (cudaStream_t) stream_v;

    if (ggml_type == GGML_TM_DTYPE_FP16) {
        // FP16 path: no packing; turbomind dispatches to cuBLAS. Caller
        // should not hit this in production (FP16 isn't a quant type), but
        // we support it for completeness.
        cudaMemcpyAsync(weight_out, src, (size_t)N * K * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
        *k_pack_out = 0;
        return 0;
    }

    // ---- Get converters (B-weight + V-scales) for this (dtype, sm) ----
    auto convs = tmg::GetConverters(
        /*data_type=*/turbomind::kHalf,
        /*weight_type=*/to_tm_wdtype(ggml_type),
        /*input_type=*/turbomind::kHalf,
        /*grouped=*/false,
        /*sm=*/70);
    const tmg::LayoutConverter* conv_w = convs[0];
    const tmg::LayoutConverter* conv_s = convs[1];
    if (!conv_w) {
        fprintf(stderr, "[ggml-turbomind] no weight converter for type=%d on sm70\n",
                ggml_type);
        return 3;
    }

    const int bits       = bits_per_weight(ggml_type);
    const int output_dim = N;
    const int input_dim  = K;
    const size_t n_elem  = (size_t)output_dim * input_dim;

    // ---- Step 1: extend src to u16 tmp ----
    uint16_t* tmp = nullptr;
    if (cudaMalloc(&tmp, n_elem * sizeof(uint16_t)) != cudaSuccess) return 4;

    if (bits == 4) {
        turbomind::extend_to_u16(tmp, (const turbomind::uint4_t*)src, n_elem, stream);
    } else if (bits == 8) {
        turbomind::extend_to_u16(tmp, (const uint8_t*)src, n_elem, stream);
    } else {
        cudaFree(tmp);
        return 5;
    }

    // ---- Step 2: transpose if conv expects row-major source ----
    if (conv_w->order == tmg::kRowMajor) {
        uint16_t* trans = nullptr;
        if (cudaMalloc(&trans, n_elem * sizeof(uint16_t)) != cudaSuccess) {
            cudaFree(tmp);
            return 6;
        }
        turbomind::invokeTransposeAxis01(trans, tmp, input_dim, output_dim, 1, stream);
        cudaFree(tmp);
        tmp = trans;
    }

    // ---- Step 3: build w_desc + kd (matches linear_weight.cc dance) ----
    tmg::MatrixLayout w_desc{
        turbomind::kHalf,
        conv_w->order,
        output_dim,
        input_dim,
        conv_w->order == tmg::kRowMajor ? input_dim : output_dim,
    };
    const bool is_A = tmg::get_operand_tag(conv_w->pack) == tmg::OPERAND_A;
    if (!is_A) {
        std::swap(w_desc.rows, w_desc.cols);
        w_desc.order = ~w_desc.order;
    }

    tmg::MatrixLayout kd = w_desc;
    kd.type = (bits == 4) ? turbomind::data_type_v<turbomind::uint4_t>
                          : turbomind::data_type_v<uint8_t>;
    kd.pack = conv_w->pack;

    // Pre-zero output buffer.
    const size_t raw_bytes = turbomind::byte_size(to_tm_wdtype(ggml_type), n_elem);
    cudaMemsetAsync(weight_out, 0, raw_bytes, stream);

    int rc = conv_w->Convert(tmp, w_desc, weight_out, kd, stream);
    cudaFree(tmp);
    if (rc != 0) {
        fprintf(stderr, "[ggml-turbomind] conv_w->Convert rc=%d\n", rc);
        return 7;
    }

    kd.type = to_tm_wdtype(ggml_type);  // restore final type
    kd.num = 1;
    *k_pack_out = encode_pack(kd.pack);

    // ---- Step 4: scales path ----
    if (conv_s && scales_out) {
        // For sub-byte / fp8 quant, the source scales are stored in the
        // GGML block alongside the weight values. The caller is expected
        // to have packed them somewhere — but for the initial smoke-test
        // implementation we'll require scales as a separate buffer the
        // caller pre-extracts. The C ABI accepts both via the `src`
        // pointer if the caller passes a struct... but for now we expect
        // the scales to be EXTRACTED from `src` already and the caller
        // passes a pointer to them via scales_out's INITIAL CONTENT.
        //
        // SIMPLIFICATION for P1: this path requires the caller to extract
        // scales separately. For the dlopen test we'll skip this path and
        // just zero scales_out. Real integration in P2 will spec the
        // src tensor layout precisely.
        cudaMemsetAsync(scales_out, 0,
            turbomind::byte_size(to_tm_sdtype(ggml_type), N * (K / group_size)),
            stream);
        // Note: caller must call ggml_turbomind_pack_scales() (TODO P2) to
        // actually populate scales_out. For now this is a stub.
    }

    cudaStreamSynchronize(stream);
    return 0;
}

// ============================================================================
// API: mul_mat (single, non-grouped)
// ============================================================================
extern "C" GGML_TM_EXPORT int ggml_turbomind_mul_mat(
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
    void*       stream_v)
{
    if (!g_state.initialized) return 100;
    if (!A || !B_packed || !D) return 1;
    cudaStream_t stream = (cudaStream_t) stream_v;

    tmg::Workspace workspace{};
    workspace.barriers        = g_state.d_barriers;
    workspace.barriers_size   = tmg::Gemm::kBarriersSize;
    workspace.partials        = g_state.d_partials;
    workspace.partials_size   = g_state.partials_size;
    workspace.tensormaps      = nullptr;
    workspace.tensormaps_size = 0;
    workspace.flags           = g_state.d_flags;

    tmg::Operation op{};
    op.dispatch  = tmg::DispatchPolicy::kDefault;
    op.epilogue  = tmg::Epilogue::kNone;
    op.quant_a   = tmg::QuantDesc{tmg::QuantType::kNone, 0};
    if (ggml_type == GGML_TM_DTYPE_FP16) {
        op.quant_b = tmg::QuantDesc{tmg::QuantType::kNone, 0};
    } else {
        op.quant_b = tmg::QuantDesc{tmg::QuantType::kK, group_size};
    }
    op.batch_dim = 0;

    tmg::MatrixLayout Adesc{turbomind::kHalf, tmg::Order::kRowMajor,
                            M, K, K, 0, 1, nullptr, nullptr};

    // Bdesc: reconstruct from k_pack_value.
    // For non-FP16 quant, B is in column-major with the encoded pack.
    tmg::MatrixLayout Bdesc;
    if (ggml_type == GGML_TM_DTYPE_FP16) {
        Bdesc = tmg::MatrixLayout{turbomind::kHalf, tmg::Order::kRowMajor,
                                  K, N, N, 0, 1, nullptr, nullptr};
    } else {
        Bdesc.type   = to_tm_wdtype(ggml_type);
        Bdesc.order  = tmg::Order::kColMajor;
        Bdesc.rows   = K;
        Bdesc.cols   = N;
        Bdesc.ld     = K;
        Bdesc.pack   = decode_pack(k_pack_value);
        Bdesc.num    = 1;
        Bdesc.offsets = nullptr;
        Bdesc.idxs   = nullptr;
    }

    tmg::MatrixLayout Vdesc{};
    if (V_packed) {
        Vdesc.type   = to_tm_sdtype(ggml_type);
        Vdesc.order  = tmg::Order::kColMajor;
        Vdesc.rows   = K / group_size;
        Vdesc.cols   = N;
        Vdesc.ld     = K / group_size;
        Vdesc.pack   = decode_pack(k_pack_value & 0xFFF0F0u) | tmg::OPERAND_V;
        // ^ approximation; real Pack reconstruction needs the V converter
        //   metadata too. For dispatcher correctness we may need a
        //   separate v_pack value — flagged as P2 follow-up.
        Vdesc.num    = 1;
    }

    tmg::MatrixLayout Cdesc, Ddesc;
    Ddesc = tmg::MatrixLayout{turbomind::kHalf, tmg::Order::kColMajor,
                              M, N, M, 0, 1, nullptr, nullptr};
    Cdesc = Ddesc;
    tmg::MatrixLayout Udesc{};

    int rc = g_state.gemm->Run(
        op, 1.0f, A, Adesc, nullptr, Udesc,
        B_packed, Bdesc, V_packed, Vdesc, 0.0f, nullptr, Cdesc,
        D, Ddesc, workspace, stream);
    return rc;
}

// ============================================================================
// API: mul_mat_grouped (the MoE primitive)
// ============================================================================
extern "C" GGML_TM_EXPORT int ggml_turbomind_mul_mat_grouped(
    const void*        A,
    const int*         token_indices,
    const int*         expert_offsets,
    int                num_experts,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                ggml_type,
    int                N,
    int                K,
    int                group_size,
    int                k_pack_value,
    void*              D,
    void*              stream_v)
{
    if (!g_state.initialized)                return 100;
    if (!A || !expert_offsets || !weights_packed || !D) return 1;
    if (num_experts <= 0)                    return 2;
    cudaStream_t stream = (cudaStream_t) stream_v;

    tmg::Workspace workspace{};
    workspace.barriers        = g_state.d_barriers;
    workspace.barriers_size   = tmg::Gemm::kBarriersSize;
    workspace.partials        = g_state.d_partials;
    workspace.partials_size   = g_state.partials_size;
    workspace.tensormaps      = nullptr;
    workspace.tensormaps_size = 0;
    workspace.flags           = g_state.d_flags;

    tmg::Operation op{};
    op.dispatch  = tmg::DispatchPolicy::kDefault;
    op.epilogue  = tmg::Epilogue::kNone;
    op.quant_a   = tmg::QuantDesc{tmg::QuantType::kNone, 0};
    op.quant_b   = (ggml_type == GGML_TM_DTYPE_FP16)
                       ? tmg::QuantDesc{tmg::QuantType::kNone, 0}
                       : tmg::QuantDesc{tmg::QuantType::kK, group_size};
    op.batch_dim = 0;

    // Total tokens = expert_offsets[num_experts]. We need to read this from
    // device memory — but it's a single int, so synchronous copy is fine.
    int total_tokens = 0;
    cudaMemcpy(&total_tokens, &expert_offsets[num_experts], sizeof(int),
               cudaMemcpyDeviceToHost);

    // Adesc: ragged-batch input. num=num_experts, offsets=expert_offsets,
    // idxs=token_indices. ld = K (per-row stride).
    tmg::MatrixLayout Adesc;
    Adesc.type    = turbomind::kHalf;
    Adesc.order   = tmg::Order::kRowMajor;
    Adesc.rows    = total_tokens;
    Adesc.cols    = K;
    Adesc.ld      = K;
    Adesc.pack    = 0;
    Adesc.num     = num_experts;
    Adesc.offsets = const_cast<int*>(expert_offsets);
    Adesc.idxs    = const_cast<int*>(token_indices);

    // Bdesc: per-expert strided pointers. num=num_experts, ld=0 (signal that
    // the offsets are pointer-array, not row offset).
    // Following turbomind convention: when num>1 and Bdesc.ld=0, weights are
    // passed as a device array of per-expert pointers (B parameter is the
    // device pointer array itself, not a single tensor).
    tmg::MatrixLayout Bdesc{};
    Bdesc.type    = to_tm_wdtype(ggml_type);
    Bdesc.order   = tmg::Order::kColMajor;
    Bdesc.rows    = K;
    Bdesc.cols    = N;
    Bdesc.ld      = 0;   // signal per-expert pointers
    Bdesc.pack    = decode_pack(k_pack_value);
    Bdesc.num     = num_experts;
    Bdesc.offsets = nullptr;
    Bdesc.idxs    = nullptr;

    tmg::MatrixLayout Vdesc{};
    if (scales_packed) {
        Vdesc.type    = to_tm_sdtype(ggml_type);
        Vdesc.order   = tmg::Order::kColMajor;
        Vdesc.rows    = K / group_size;
        Vdesc.cols    = N;
        Vdesc.ld      = 0;
        Vdesc.pack    = decode_pack(k_pack_value & 0xFFF0F0u) | tmg::OPERAND_V;
        Vdesc.num     = num_experts;
    }

    tmg::MatrixLayout Ddesc;
    Ddesc.type    = turbomind::kHalf;
    Ddesc.order   = tmg::Order::kRowMajor;
    Ddesc.rows    = total_tokens;
    Ddesc.cols    = N;
    Ddesc.ld      = N;
    Ddesc.pack    = 0;
    Ddesc.num     = num_experts;
    Ddesc.offsets = const_cast<int*>(expert_offsets);
    Ddesc.idxs    = nullptr;
    tmg::MatrixLayout Cdesc = Ddesc;
    tmg::MatrixLayout Udesc{};

    int rc = g_state.gemm->Run(
        op, 1.0f, A, Adesc, nullptr, Udesc,
        weights_packed, Bdesc, scales_packed, Vdesc, 0.0f, nullptr, Cdesc,
        D, Ddesc, workspace, stream);
    return rc;
}

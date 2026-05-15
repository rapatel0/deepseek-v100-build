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

#include "ggml-turbomind-deinterleave.h"
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
// Module-level state — SPRINT-025 P2: PER-DEVICE state[cuda_device].
//
// Prior to SPRINT-025 a single global `g_state` held workspace pointers for
// one device at a time, and ggml_turbomind_init(cuda_device) called
// cudaFree(...) on the previous device's pointers when the cuda_device
// changed. That serialized cross-device dispatch and could free pointers
// concurrent dispatches were still using. Now each device has its own
// State entry, indexed by cuda_device ordinal. Idempotent init per device.
// No C ABI change; the entry points still take an int cuda_device.
// ============================================================================
namespace {

constexpr int TM_MAX_DEVICES = 32;

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

// Default-constructed array of per-device State entries. Indexed by cuda
// device ordinal. The std::mutex inside each State is independently held
// per device.
State g_states[TM_MAX_DEVICES];

// Resolve a device ordinal to its State entry. Range-checks; returns
// nullptr if cuda_device is out of bounds.
inline State * get_state(int cuda_device) {
    if (cuda_device < 0 || cuda_device >= TM_MAX_DEVICES) return nullptr;
    return &g_states[cuda_device];
}

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
    // Output scale dtype produced by conv_s for each weight type on sm70.
    // Per convert_v3.cu: F8_E4M3 uses Cvt<uint16_t, uint16_t> for scales,
    // MXFP4 uses Cvt<uint8_t, uint8_t>.
    switch (ggml_type) {
        case GGML_TM_DTYPE_F8_E4M3_B128: return turbomind::kHalf;   // 2-byte FP16
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
// SPRINT-025 P2: per-device idempotent init. No tear-down on "different
// device" — each device has its own State entry.
extern "C" GGML_TM_EXPORT int ggml_turbomind_init(int cuda_device) {
    State * s = get_state(cuda_device);
    if (!s) {
        fprintf(stderr, "[ggml-turbomind] cuda_device=%d out of range [0,%d)\n",
                cuda_device, TM_MAX_DEVICES);
        return 3;
    }
    std::lock_guard<std::mutex> lk(s->mtx);
    if (s->initialized) return 0;  // idempotent per device
    cudaError_t err = cudaSetDevice(cuda_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ggml-turbomind] cudaSetDevice(%d) failed: %s\n",
                cuda_device, cudaGetErrorString(err));
        return 1;
    }
    s->gemm = new tmg::Gemm();
    s->partials_size = (size_t) 4096 * 4096 * sizeof(float) * 4;
    if (cudaMalloc(&s->d_barriers, tmg::Gemm::kBarriersSize) != cudaSuccess ||
        cudaMalloc(&s->d_partials, s->partials_size)         != cudaSuccess ||
        cudaMalloc(&s->d_flags,    sizeof(int) * 1024)       != cudaSuccess) {
        fprintf(stderr, "[ggml-turbomind] failed to allocate workspace on dev %d\n",
                cuda_device);
        return 2;
    }
    s->device      = cuda_device;
    s->initialized = true;
    return 0;
}

// SPRINT-025 P2: tear down ALL device entries. The C ABI has no
// per-device shutdown variant; on dlclose we walk all initialized devices.
extern "C" GGML_TM_EXPORT void ggml_turbomind_shutdown(void) {
    for (int d = 0; d < TM_MAX_DEVICES; ++d) {
        State * s = &g_states[d];
        std::lock_guard<std::mutex> lk(s->mtx);
        if (!s->initialized) continue;
        cudaSetDevice(s->device);
        delete s->gemm;
        s->gemm = nullptr;
        cudaFree(s->d_barriers); s->d_barriers = nullptr;
        cudaFree(s->d_partials); s->d_partials = nullptr;
        cudaFree(s->d_flags);    s->d_flags    = nullptr;
        s->partials_size = 0;
        s->initialized   = false;
        s->device        = -1;
    }
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
    // SPRINT-025 P2: resolve State by current CUDA device (caller already
    // called cudaSetDevice). Each device has its own State entry.
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    State * s = get_state(cur_dev);
    if (!s || !s->initialized) return 100;
    if (!src || !weight_out)  return 1;
    if (!k_pack_out)          return 2;
    cudaStream_t stream = (cudaStream_t) stream_v;

    if (ggml_type == GGML_TM_DTYPE_FP16) {
        // FP16 path: no packing; turbomind dispatches to cuBLAS.
        cudaMemcpyAsync(weight_out, src, (size_t)N * K * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
        *k_pack_out = 0;
        return 0;
    }

    // The GGML source is in interleaved block layout:
    //   F8_E4M3_B128: [uint8 scale, uint8 qs[128]] per block
    //   MXFP4:        [uint8 scale, uint8 qs[16]] per block
    //
    // Turbomind's Convert API wants the weight bytes and scales as TWO
    // SEPARATE flat buffers. P2.1 introduced deinterleave kernels for that.
    //
    // After deinterleave we follow the same dance as
    // tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu (the proven-working
    // sm70 packed-weight invocation pattern).

    // ---- Step 0: deinterleave GGML blocks straight into uint16 [K, N] tmp ----
    //
    // This replaces the bench's "raw bytes → extend_to_u16" two-step. We
    // produce the u16 tmp directly because two-fp4-per-byte packing is hard
    // to do row-major in [K, N] across N strides.
    const int    bits      = bits_per_weight(ggml_type);
    const int    n_scales  = N * (K / group_size);
    const int    output_dim = N;
    const int    input_dim  = K;
    const size_t n_elem    = (size_t)output_dim * input_dim;

    uint16_t* tmp = nullptr;
    if (cudaMalloc(&tmp, n_elem * sizeof(uint16_t)) != cudaSuccess) return 8;

    const size_t scale_byte_size_per_value =
        (ggml_type == GGML_TM_DTYPE_F8_E4M3_B128) ? 2 : 1;
    void* raw_scales = nullptr;
    if (scales_out) {
        if (cudaMalloc(&raw_scales, (size_t)n_scales * scale_byte_size_per_value) != cudaSuccess) {
            cudaFree(tmp);
            return 9;
        }
    }

    cudaError_t derr = cudaSuccess;
    if (ggml_type == GGML_TM_DTYPE_F8_E4M3_B128) {
        derr = ggml_turbomind::launch_deinterleave_f8_e4m3_b128(
            src, tmp, raw_scales, N, K, stream);
    } else if (ggml_type == GGML_TM_DTYPE_MXFP4) {
        derr = ggml_turbomind::launch_deinterleave_mxfp4(
            src, tmp, raw_scales, N, K, stream);
    } else {
        fprintf(stderr, "[ggml-turbomind] unsupported ggml_type=%d\n", ggml_type);
        cudaFree(tmp);
        if (raw_scales) cudaFree(raw_scales);
        return 10;
    }
    if (derr != cudaSuccess) {
        fprintf(stderr, "[ggml-turbomind] deinterleave launch failed: %s\n",
                cudaGetErrorString(derr));
        cudaFree(tmp);
        if (raw_scales) cudaFree(raw_scales);
        return 11;
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
        cudaFree(tmp);
        if (raw_scales) cudaFree(raw_scales);
        return 3;
    }

    // tmp is [K, N] u16 row-major. Transpose to [N, K] u16 row-major before
    // passing to conv_w — matches lmdeploy/models/linear_weight.cc convention.
    if (conv_w->order == tmg::kRowMajor) {
        uint16_t* trans = nullptr;
        if (cudaMalloc(&trans, n_elem * sizeof(uint16_t)) != cudaSuccess) {
            cudaFree(tmp);
            if (raw_scales) cudaFree(raw_scales);
            return 6;
        }
        turbomind::invokeTransposeAxis01(trans, tmp, input_dim, output_dim, 1, stream);
        cudaFree(tmp);
        tmp = trans;
    }

    // ---- Step 3: build w_desc + kd ----
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

    const size_t raw_bytes_out = turbomind::byte_size(to_tm_wdtype(ggml_type), n_elem);
    cudaMemsetAsync(weight_out, 0, raw_bytes_out, stream);

    int rc = conv_w->Convert(tmp, w_desc, weight_out, kd, stream);
    cudaFree(tmp);
    if (rc != 0) {
        fprintf(stderr, "[ggml-turbomind] conv_w->Convert rc=%d\n", rc);
        if (raw_scales) cudaFree(raw_scales);
        return 7;
    }

    kd.type = to_tm_wdtype(ggml_type);
    kd.num  = 1;
    *k_pack_out = encode_pack(kd.pack);

    // ---- Step 3.5: adjust UE8M0 scales for half-precision dispatch ----
    // MXFP4 scales are stored as raw E8M0 bytes. The sm70 transform pipeline
    // reads them directly as FP16 exponent fields, so we need to re-bias from
    // E8M0 bias=127 to FP16 bias=15. See linear_weight.cc:241.
    if (ggml_type == GGML_TM_DTYPE_MXFP4 && raw_scales) {
        turbomind::AdjustUe8m0ScaleForHalf((uint8_t*)raw_scales, n_scales, stream);
    }

    // ---- Step 4: scales path ----
    if (conv_s && scales_out && raw_scales) {
        // Source scale type matches what the deinterleave produced.
        // F8: FP16 (kUint16 to Convert per linear_weight.cc convention)
        // MXFP4: uint8 (E8M0 byte)
        turbomind::DataType src_scale_type =
            (ggml_type == GGML_TM_DTYPE_F8_E4M3_B128)
                ? turbomind::kUint16
                : turbomind::kUint8;

        tmg::MatrixLayout s_desc{
            src_scale_type,
            conv_s->order,
            output_dim,                   // rows
            input_dim / group_size,       // cols
            output_dim,
        };
        const bool s_is_A = tmg::get_operand_tag(conv_s->pack) == tmg::OPERAND_U;
        if (!s_is_A) {
            std::swap(s_desc.rows, s_desc.cols);
            s_desc.order = ~s_desc.order;
        }

        tmg::MatrixLayout qd = s_desc;
        qd.pack = conv_s->pack;

        const size_t scale_out_bytes =
            turbomind::byte_size(to_tm_sdtype(ggml_type), n_scales);
        cudaMemsetAsync(scales_out, 0, scale_out_bytes, stream);

        int src_rc = conv_s->Convert(raw_scales, s_desc, scales_out, qd, stream);
        cudaFree(raw_scales);
        if (src_rc != 0) {
            fprintf(stderr, "[ggml-turbomind] conv_s->Convert rc=%d\n", src_rc);
            return 12;
        }
        qd.num = 1;
        // Encode v_pack in upper 12 bits of k_pack_out so mul_mat can recover.
        *k_pack_out = encode_pack(kd.pack) | (encode_pack(qd.pack) << 12);
    } else if (raw_scales) {
        cudaFree(raw_scales);
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
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    State * s = get_state(cur_dev);
    if (!s || !s->initialized) return 100;
    if (!A || !B_packed || !D) return 1;
    cudaStream_t stream = (cudaStream_t) stream_v;

    tmg::Workspace workspace{};
    workspace.barriers        = s->d_barriers;
    workspace.barriers_size   = tmg::Gemm::kBarriersSize;
    workspace.partials        = s->d_partials;
    workspace.partials_size   = s->partials_size;
    workspace.tensormaps      = nullptr;
    workspace.tensormaps_size = 0;
    workspace.flags           = s->d_flags;

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

    // Bdesc / Vdesc: orders must match what GetConverters() returned during
    // pack — which the registered turbomind kernels were built around. The
    // pack stage built w_desc/s_desc with conv_*->order then flipped order
    // when the operand tag is B (or V). Replicate that here.
    tmg::MatrixLayout Bdesc;
    tmg::MatrixLayout Vdesc{};
    if (ggml_type == GGML_TM_DTYPE_FP16) {
        Bdesc = tmg::MatrixLayout{turbomind::kHalf, tmg::Order::kRowMajor,
                                  K, N, N, 0, 1, nullptr, nullptr};
    } else {
        auto convs = tmg::GetConverters(
            /*data_type=*/turbomind::kHalf,
            /*weight_type=*/to_tm_wdtype(ggml_type),
            /*input_type=*/turbomind::kHalf,
            /*grouped=*/false,
            /*sm=*/70);
        const tmg::LayoutConverter* conv_w = convs[0];
        const tmg::LayoutConverter* conv_s = convs[1];
        if (!conv_w) {
            fprintf(stderr, "[ggml-turbomind] mul_mat: no conv_w for type=%d\n", ggml_type);
            return 3;
        }

        // B layout — start with conv_w->order, then apply OPERAND_B swap
        // (same as pack stage).
        // Build w_desc the same way pack_weight_expert did, then apply the
        // Packing_v2 transform to get the ld of the PACKED output (= K*32 for
        // sm70 HMMA_884 OPERAND_B Pack_M=1, not the bare K of the unpacked form).
        tmg::MatrixLayout w_desc{
            turbomind::kHalf,
            conv_w->order,
            N,
            K,
            (conv_w->order == tmg::kRowMajor) ? K : N,
        };
        if (tmg::get_operand_tag(conv_w->pack) != tmg::OPERAND_A) {
            std::swap(w_desc.rows, w_desc.cols);
            w_desc.order = ~w_desc.order;
        }
        // Replicate convert_v3.cu's Ddesc.ld update:
        //   trans Sdesc, then ld = mk2cs<order>(Packing_v2::apply({rows,cols})).x
        // For OPERAND_B sm70 HMMA_884 Pack_M=1, Packing_v2::apply({m,k})={m/32, k*32}.
        const bool b_trans = tmg::get_operand_tag(conv_w->pack) == tmg::OPERAND_B
                          || tmg::get_operand_tag(conv_w->pack) == tmg::OPERAND_V;
        const int b_pack_num = (int)(conv_w->pack & 0xF);
        int b_packed_ld;
        {
            int rows_t = b_trans ? w_desc.cols : w_desc.rows;
            int cols_t = b_trans ? w_desc.rows : w_desc.cols;
            // For HMMA_884 OPERAND_B Pack_M=num: apply({rows, cols}) = {rows/(32*num), cols*32*num}.
            int packed_rows = rows_t / (32 * b_pack_num);
            int packed_cols = cols_t * 32 * b_pack_num;
            b_packed_ld = (conv_w->order == tmg::kRowMajor) ? packed_cols : packed_rows;
            (void)packed_rows; (void)packed_cols;
        }

        Bdesc.type    = to_tm_wdtype(ggml_type);
        Bdesc.order   = w_desc.order;
        Bdesc.rows    = w_desc.rows;
        Bdesc.cols    = w_desc.cols;
        Bdesc.ld      = b_packed_ld;
        Bdesc.pack    = decode_pack(k_pack_value & 0xFFFu);
        Bdesc.num     = 1;
        Bdesc.offsets = nullptr;
        Bdesc.idxs    = nullptr;

        if (V_packed && conv_s) {
            uint32_t v_pack_raw = ((uint32_t)k_pack_value >> 12) & 0xFFFu;
            if (v_pack_raw == 0) {
                v_pack_raw = ((uint32_t)k_pack_value & 0xF0Fu) | (uint32_t)tmg::OPERAND_V;
            }
            // For OPERAND_V sm70 HMMA_884 there's no PackingImpl specialization,
            // so apply returns mk unchanged. ld = original ld after swap.
            tmg::MatrixLayout s_desc{
                to_tm_sdtype(ggml_type),
                conv_s->order,
                N,
                K / group_size,
                N,  // ld pre-swap
            };
            if (tmg::get_operand_tag(conv_s->pack) != tmg::OPERAND_U) {
                std::swap(s_desc.rows, s_desc.cols);
                s_desc.order = ~s_desc.order;
            }
            Vdesc.type    = s_desc.type;
            Vdesc.order   = s_desc.order;
            Vdesc.rows    = s_desc.rows;
            Vdesc.cols    = s_desc.cols;
            Vdesc.ld      = s_desc.ld;
            Vdesc.pack    = (tmg::Pack)v_pack_raw;
            Vdesc.num     = 1;
        }
    }

    // D: row-major [M, N]. Matches the proven sm70 invocation in
    // tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu.
    tmg::MatrixLayout Cdesc, Ddesc;
    Ddesc = tmg::MatrixLayout{turbomind::kHalf, tmg::Order::kRowMajor,
                              M, N, N, 0, 1, nullptr, nullptr};
    Cdesc = Ddesc;
    tmg::MatrixLayout Udesc{};

    int rc = s->gemm->Run(
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
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    State * s = get_state(cur_dev);
    if (!s || !s->initialized)               return 100;
    if (!A || !expert_offsets || !weights_packed || !D) return 1;
    if (num_experts <= 0)                    return 2;
    cudaStream_t stream = (cudaStream_t) stream_v;

    tmg::Workspace workspace{};
    workspace.barriers        = s->d_barriers;
    workspace.barriers_size   = tmg::Gemm::kBarriersSize;
    workspace.partials        = s->d_partials;
    workspace.partials_size   = s->partials_size;
    workspace.tensormaps      = nullptr;
    workspace.tensormaps_size = 0;
    workspace.flags           = s->d_flags;

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

    // Bdesc / Vdesc: derive orders from the converter that was used to pack
    // these tensors (same flow as ggml_turbomind_mul_mat).
    tmg::MatrixLayout Bdesc{};
    tmg::MatrixLayout Vdesc{};
    if (ggml_type != GGML_TM_DTYPE_FP16) {
        auto convs = tmg::GetConverters(
            /*data_type=*/turbomind::kHalf,
            /*weight_type=*/to_tm_wdtype(ggml_type),
            /*input_type=*/turbomind::kHalf,
            /*grouped=*/false,  // matches pack_weight_expert
            /*sm=*/70);
        const tmg::LayoutConverter* conv_w = convs[0];
        const tmg::LayoutConverter* conv_s = convs[1];
        if (!conv_w) {
            fprintf(stderr, "[ggml-turbomind] mul_mat_grouped: no conv_w for type=%d\n", ggml_type);
            return 3;
        }

        Bdesc.type    = to_tm_wdtype(ggml_type);
        Bdesc.order   = conv_w->order;
        Bdesc.rows    = N;
        Bdesc.cols    = K;
        Bdesc.ld      = 0;  // per-expert pointer array
        Bdesc.pack    = decode_pack(k_pack_value & 0xFFFu);
        Bdesc.num     = num_experts;
        Bdesc.offsets = nullptr;
        Bdesc.idxs    = nullptr;
        if (tmg::get_operand_tag(conv_w->pack) != tmg::OPERAND_A) {
            std::swap(Bdesc.rows, Bdesc.cols);
            Bdesc.order = ~Bdesc.order;
        }

        if (scales_packed && conv_s) {
            uint32_t v_pack_raw = ((uint32_t)k_pack_value >> 12) & 0xFFFu;
            if (v_pack_raw == 0) {
                v_pack_raw = ((uint32_t)k_pack_value & 0xF0Fu) | (uint32_t)tmg::OPERAND_V;
            }
            Vdesc.type    = to_tm_sdtype(ggml_type);
            Vdesc.order   = conv_s->order;
            Vdesc.rows    = N;
            Vdesc.cols    = K / group_size;
            Vdesc.ld      = 0;
            Vdesc.pack    = (tmg::Pack)v_pack_raw;
            Vdesc.num     = num_experts;
            if (tmg::get_operand_tag(conv_s->pack) != tmg::OPERAND_U) {
                std::swap(Vdesc.rows, Vdesc.cols);
                Vdesc.order = ~Vdesc.order;
            }
        }
    } else {
        Bdesc.type    = turbomind::kHalf;
        Bdesc.order   = tmg::Order::kRowMajor;
        Bdesc.rows    = K;
        Bdesc.cols    = N;
        Bdesc.ld      = 0;
        Bdesc.pack    = 0;
        Bdesc.num     = num_experts;
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

    int rc = s->gemm->Run(
        op, 1.0f, A, Adesc, nullptr, Udesc,
        weights_packed, Bdesc, scales_packed, Vdesc, 0.0f, nullptr, Cdesc,
        D, Ddesc, workspace, stream);
    return rc;
}

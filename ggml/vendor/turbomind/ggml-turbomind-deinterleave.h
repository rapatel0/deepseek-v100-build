// SPRINT-023 P2.1 — Host-side declarations for GGML block-interleaved →
// turbomind flat (weight, scale) deinterleave kernels.
//
// The kernel definitions live in ggml-turbomind-deinterleave.cu (nvcc).
// This header is C++ compatible so api.cc (compiled as CXX) can call them.

#pragma once

#include <cuda_runtime.h>

namespace ggml_turbomind {

// F8_E4M3_B128: GGML block = [uint8 e8m0 scale, uint8 qs[128]]
//   weight_u16_out:  uint16 buffer [K, N] row-major (fp8 byte zero-extended)
//   scale_fp16_out:  half buffer   [K/128, N] row-major (E8M0 → FP16)
cudaError_t launch_deinterleave_f8_e4m3_b128(
    const void*  src,
    void*        weight_u16_out,
    void*        scale_fp16_out,
    int          N,
    int          K,
    cudaStream_t stream);

// MXFP4: GGML block = [uint8 e8m0 scale, uint8 qs[16] packed 2 fp4 per byte]
//   weight_u16_out: uint16 buffer [K, N] row-major (fp4 nibble zero-extended)
//   scale_u8_out:   byte buffer   [K/32, N] row-major (E8M0 copied directly)
cudaError_t launch_deinterleave_mxfp4(
    const void*  src,
    void*        weight_u16_out,
    void*        scale_u8_out,
    int          N,
    int          K,
    cudaStream_t stream);

}  // namespace ggml_turbomind

// SPRINT-023 P2.1 — GGML block-interleaved → turbomind flat (weight, scale)
// deinterleave kernels.
//
// Input  (GGML block layouts from ggml/src/ggml-common.h):
//   block_f8_e4m3_b128: [ uint8_t e8m0_scale; uint8_t qs[128]; ]  (129 B/block)
//   block_mxfp4:        [ uint8_t e8m0_scale; uint8_t qs[16];  ]  (17  B/block, 32 fp4 vals)
//
// Output (matches lmdeploy/models/linear_weight.cc tmp layout):
//   weight_out:  uint16_t [K, N] row-major (zero-extended fp8 byte / fp4 nibble)
//   scale_out:   For F8 → __half [K/128, N] row-major (E8M0 byte → FP16)
//                For MXFP4 → uint8_t [K/32, N] row-major (E8M0 byte direct)
//
// GGML source is in [N, K] block-row layout. We transpose to [K, N] during
// deinterleave so the downstream convert sees the lmdeploy convention.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ggml_turbomind {

constexpr int QK_F8_E4M3_B128 = 128;
constexpr int QK_MXFP4        = 32;

struct __align__(1) block_f8_e4m3_b128 {
    uint8_t e;
    uint8_t qs[128];
};
static_assert(sizeof(block_f8_e4m3_b128) == 129, "wrong f8 block size");

struct __align__(1) block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];
};
static_assert(sizeof(block_mxfp4) == 17, "wrong mxfp4 block size");

// ============================================================================
// F8_E4M3_B128 deinterleave → [K, N] uint16 weight + [K/128, N] fp16 scale
// ============================================================================
__global__ void k_deinterleave_f8_e4m3_b128(
    const block_f8_e4m3_b128* __restrict__ src,
    uint16_t* __restrict__ weight_u16_out,
    __half*   __restrict__ scale_fp16_out,
    int N, int K)
{
    const int blocks_per_row = K / QK_F8_E4M3_B128;
    const int row = blockIdx.y;  // n
    const int bk  = blockIdx.x;
    if (row >= N || bk >= blocks_per_row) return;

    const block_f8_e4m3_b128& b = src[row * blocks_per_row + bk];

    const int t = threadIdx.x;
    if (t < QK_F8_E4M3_B128) {
        const int k = bk * QK_F8_E4M3_B128 + t;
        // [K, N] row-major: (k, n) → offset k*N + n. Zero-extend the fp8 byte.
        weight_u16_out[k * N + row] = (uint16_t)b.qs[t];
    }

    if (t == 0) {
        int e_fp16 = (int)b.e - 112;
        uint16_t fp16_bits;
        if (e_fp16 <= 0) {
            fp16_bits = 0;
        } else if (e_fp16 >= 0x1F) {
            fp16_bits = 0x7BFF;
        } else {
            fp16_bits = (uint16_t)(e_fp16 << 10);
        }
        scale_fp16_out[bk * N + row] = *reinterpret_cast<__half*>(&fp16_bits);
    }
}

// ============================================================================
// MXFP4 deinterleave → [K, N] uint16 weight + [K/32, N] uint8 scale
// ============================================================================
__global__ void k_deinterleave_mxfp4(
    const block_mxfp4* __restrict__ src,
    uint16_t* __restrict__ weight_u16_out,
    uint8_t*  __restrict__ scale_u8_out,
    int N, int K)
{
    const int blocks_per_row = K / QK_MXFP4;
    const int row = blockIdx.y;
    const int bk  = blockIdx.x;
    if (row >= N || bk >= blocks_per_row) return;

    const block_mxfp4& b = src[row * blocks_per_row + bk];
    const int t = threadIdx.x;
    if (t < 16) {
        // GGML stores the low nibble for k=t and the high nibble for k=t+16.
        uint8_t byte = b.qs[t];
        uint16_t lo = (uint16_t)(byte & 0x0F);
        uint16_t hi = (uint16_t)(byte >> 4);
        const int k0 = bk * QK_MXFP4 + t;
        weight_u16_out[k0                  * N + row] = lo;
        weight_u16_out[(k0 + QK_MXFP4 / 2) * N + row] = hi;
    }
    if (t == 0) {
        scale_u8_out[bk * N + row] = b.e;
    }
}

// ============================================================================
// Host launchers
// ============================================================================
cudaError_t launch_deinterleave_f8_e4m3_b128(
    const void* src,
    void* weight_u16_out,
    void* scale_fp16_out,
    int N, int K,
    cudaStream_t stream)
{
    const int blocks_per_row = K / QK_F8_E4M3_B128;
    dim3 grid(blocks_per_row, N, 1);
    dim3 block(128, 1, 1);
    k_deinterleave_f8_e4m3_b128<<<grid, block, 0, stream>>>(
        (const block_f8_e4m3_b128*)src,
        (uint16_t*)weight_u16_out,
        (__half*)scale_fp16_out,
        N, K);
    return cudaGetLastError();
}

cudaError_t launch_deinterleave_mxfp4(
    const void* src,
    void* weight_u16_out,
    void* scale_u8_out,
    int N, int K,
    cudaStream_t stream)
{
    const int blocks_per_row = K / QK_MXFP4;
    dim3 grid(blocks_per_row, N, 1);
    dim3 block(16, 1, 1);
    k_deinterleave_mxfp4<<<grid, block, 0, stream>>>(
        (const block_mxfp4*)src,
        (uint16_t*)weight_u16_out,
        (uint8_t*)scale_u8_out,
        N, K);
    return cudaGetLastError();
}

}  // namespace ggml_turbomind

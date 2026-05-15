// MXFP4 tensor-core matmul kernels for V100 (sm_70).
// FP4 = E2M1 nibble. Per-32-block shared E8M0 exponent (uint8).
// Reconstructed value: kvals[nibble & 7] * (nibble & 8 ? -1 : 1) * 2^(e-127) * 0.5
// where kvals = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
//
// Path-A (LUT): table lookup + FP cast in registers.
// Path-B (PRMT/parallel): pending follow-up pass; not in this file yet.

#pragma once

#include "tc_grid.h"

#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::fp4 {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float e8m0_to_f32_dev(uint8_t e) {
    return __int_as_float(((int)e) << 23);  // 2^(e-127); works for e in [1, 254]
}

template <int BM_, int BN_, int BK_, int WARPS_>
__global__ void mm_mxfp4_lut(
        const uint8_t * __restrict__ W_blocks,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    static_assert(BN / WARPS == 16, "BN/WARPS must equal 16");
    static_assert(BK % 16 == 0 && BK % QK_MXFP4 == 0, "BK constraints");

    static const float kvals_pos[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    constexpr int kBlkBytes = 1 + (QK_MXFP4 / 2);

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);
    __half * sB = sA + BM * BK;
    float  * sC = reinterpret_cast<float *>(sB + BK * BN);

    FragC c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int blocks_per_row = K / QK_MXFP4;
    const int W_row_bytes    = blocks_per_row * kBlkBytes;
    const int k_tiles = K / BK;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k0 = kt * BK;
        for (int idx = tid; idx < BM * BK; idx += threads) {
            int mm = idx / BK;
            int kk = idx % BK;
            int gm = tile_m + mm;
            int gk = k0     + kk;
            float v = 0.0f;
            if (gm < M && gk < K) v = A[(size_t) gm * K + gk];
            sA[mm * BK + kk] = __float2half(v);
        }
        for (int idx = tid; idx < BN * BK; idx += threads) {
            int nn = idx / BK;
            int kk = idx % BK;
            int gn = tile_n + nn;
            int gk = k0     + kk;
            half v = __float2half(0.0f);
            if (gn < N && gk < K) {
                int blk = gk / QK_MXFP4;
                int in_blk = gk - blk * QK_MXFP4;
                const uint8_t * row = W_blocks + (size_t) gn * W_row_bytes;
                const uint8_t * bptr = row + blk * kBlkBytes;
                uint8_t e = bptr[0];
                uint8_t code = bptr[1 + (in_blk >> 1)];
                uint8_t nibble = ((in_blk & 1) == 0) ? (code & 0xF) : (code >> 4);
                float d = e8m0_to_f32_dev(e);
                float val = kvals_pos[nibble & 7] * ((nibble & 8) ? -1.0f : 1.0f) * d * 0.5f;
                v = __float2half(val);
            }
            sB[kk + nn * BK] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a;
            wmma::load_matrix_sync(a, sA + kk, BK);
            int n_off = warp * 16;
            FragB b;
            wmma::load_matrix_sync(b, sB + kk + n_off * BK, BK);
            wmma::mma_sync(c_frag, a, b, c_frag);
        }
        __syncthreads();
    }

    float * tile_c = sC + warp * BM * 16;
    wmma::store_matrix_sync(tile_c, c_frag, 16, wmma::mem_row_major);
    __syncwarp();
    int n_off = warp * 16;
    for (int idx = lane; idx < BM * 16; idx += 32) {
        int mm = idx / 16;
        int nn = idx % 16;
        int gm = tile_m + mm;
        int gn = tile_n + n_off + nn;
        if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
    }
}

}  // namespace tc_grid::kernels::fp4

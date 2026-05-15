// INT4 tensor-core matmul kernels for V100 (sm_70).
// Path-A (LUT): unpack nibble in registers, sign-extend, FP cast, apply scale.
// Path-B will live in int4_bitshift_kernels.cuh once path-A is validated.

#pragma once

#include "tc_grid.h"

#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int4 {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// W is packed: 2 nibbles per byte. Layout: rows-major, K/2 bytes per row.
// Scales: [N, K/QK_INT4] FP16.
template <int BM_, int BN_, int BK_, int WARPS_>
__global__ void mm_int4_lut(
        const uint8_t * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    static_assert(BN % 16 == 0 && BN / WARPS == 16, "BN/WARPS must equal 16");
    static_assert(BK % 16 == 0, "BK must be multiple of 16");
    static_assert(BK % QK_INT4 == 0, "BK must be multiple of QK_INT4");

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

    const int K_pad_bytes = K / 2;
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
                int blocks_per_row = K / QK_INT4;
                int blk = gk / QK_INT4;
                float s = __half2float(W_scales[(size_t) gn * blocks_per_row + blk]);
                int byte_idx = gk / 2;
                uint8_t b = W_qs[(size_t) gn * K_pad_bytes + byte_idx];
                int8_t n;
                if ((gk & 1) == 0) n = (int8_t)((b & 0xF) << 4) >> 4;
                else               n = (int8_t)(b & 0xF0)        >> 4;
                v = __float2half((float) n * s);
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

}  // namespace tc_grid::kernels::int4

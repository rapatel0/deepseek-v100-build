// INT8 LUT with SMEM padding to avoid 2-way bank conflicts on B accesses.
//
// Standard sB layout `B[k + n*BK]` for col-major: when 32 lanes within a warp
// load WMMA fragment from sB along the (k, n) plane with n varying, addresses
// stride by BK = 32 elements = 64 bytes = 16 banks. Two lanes hit the same
// bank → 2-way conflict. Padding the per-column stride to BK+8 elements shifts
// the bank index and breaks the conflict pattern.
//
// Cost: 8 extra halves per column = 8*BN extra SMEM (e.g. 16 halves * 64 cols
// = 1024 halves = 2KB).

#pragma once

#include "tc_grid.h"

#include <mma.h>
#include <cuda_fp16.h>

namespace tc_grid::kernels::int8_pad {

using namespace nvcuda;

using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <int BM_, int BN_, int BK_, int WARPS_>
__global__ void mm_int8_lut_padded(
        const int8_t  * __restrict__ W_qs,
        const __half  * __restrict__ W_scales,
        const float   * __restrict__ A,
        float         * __restrict__ C,
        int M, int N, int K) {
    constexpr int BM = BM_, BN = BN_, BK = BK_;
    constexpr int WARPS = WARPS_;
    constexpr int BK_PAD = BK + 8;  // pad columns of B to break bank conflicts
    static_assert(BN / WARPS == 16, "BN/WARPS == 16");
    static_assert(BK % 16 == 0 && BK % QK_INT8 == 0, "BK constraints");

    const int tile_m = blockIdx.y * BM;
    const int tile_n = blockIdx.x * BN;
    const int warp   = threadIdx.x / 32;
    const int lane   = threadIdx.x & 31;
    const int tid    = (int) threadIdx.x;
    const int threads = WARPS * 32;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * sA = reinterpret_cast<__half *>(smem_raw);            // [BM, BK]
    __half * sB = sA + BM * BK;                                    // [BK_PAD * BN] col-major w/ pad
    float  * sC = reinterpret_cast<float *>(sB + BK_PAD * BN);

    FragC c_frag; wmma::fill_fragment(c_frag, 0.0f);
    const int k_tiles = K / BK;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k0 = kt * BK;
        for (int idx = tid; idx < BM * BK; idx += threads) {
            int mm = idx / BK, kk = idx % BK;
            int gm = tile_m + mm, gk = k0 + kk;
            float v = (gm < M && gk < K) ? A[(size_t) gm * K + gk] : 0.0f;
            sA[mm * BK + kk] = __float2half(v);
        }
        for (int idx = tid; idx < BN * BK; idx += threads) {
            int nn = idx / BK, kk = idx % BK;
            int gn = tile_n + nn, gk = k0 + kk;
            half v = __float2half(0.0f);
            if (gn < N && gk < K) {
                int blocks_per_row = K / QK_INT8;
                int blk = gk / QK_INT8;
                float s = __half2float(W_scales[(size_t) gn * blocks_per_row + blk]);
                int8_t q = W_qs[(size_t) gn * K + gk];
                v = __float2half((float) q * s);
            }
            // Store with PADDED stride BK_PAD
            sB[kk + nn * BK_PAD] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            FragA a; wmma::load_matrix_sync(a, sA + kk, BK);
            int n_off = warp * 16;
            FragB b; wmma::load_matrix_sync(b, sB + kk + n_off * BK_PAD, BK_PAD);
            wmma::mma_sync(c_frag, a, b, c_frag);
        }
        __syncthreads();
    }

    float * tile_c = sC + (size_t) warp * BM * 16;
    wmma::store_matrix_sync(tile_c, c_frag, 16, wmma::mem_row_major);
    __syncwarp();
    int n_off = warp * 16;
    for (int idx = lane; idx < BM * 16; idx += 32) {
        int mm = idx / 16, nn = idx % 16;
        int gm = tile_m + mm, gn = tile_n + n_off + nn;
        if (gm < M && gn < N) C[(size_t) gm * N + gn] = tile_c[idx];
    }
}

}  // namespace tc_grid::kernels::int8_pad

// SPRINT-018 P1: CUTLASS-based INT8 GEMM candidate kernel (sm_70).
//
// P1 contract: pre-dequant INT8 W_qs + half scales into a FP16 buffer ONCE
// outside the timing loop, then call CUTLASS's tuned FP16 Gemm in the timing
// loop. This gives the CUTLASS UPPER BOUND for our shape — it's not a fair
// apples-to-apples vs v10 (which does inline dequant), but it tells us the
// ceiling we can chase in P2 (fused dequant) and P3 (tile sweep).
//
// In P2 the dequant moves inside the timing loop / into the kernel.

#pragma once

#include "tc_grid.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

namespace tc_grid::kernels::int8_cutlass {

// ============================================================ Cast helpers
// Pre-dequant pass: W_qs (int8_t) + W_scales (half) -> W_fp16 (half).
// W is laid out as W[n][k] in gmem; output W_fp16 mirrors that (row-major n outer).
// Scales are per QK_INT8-block along K.
__global__ void cast_int8_to_fp16(
        const int8_t * __restrict__ W_qs,
        const __half * __restrict__ W_scales,
        __half       * __restrict__ W_fp16,
        int N, int K) {
    const int n = blockIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || k >= K) return;
    const int blocks_per_row = K / QK_INT8;
    int8_t q = W_qs[(size_t) n * K + k];
    half   s = W_scales[(size_t) n * blocks_per_row + (k / QK_INT8)];
    W_fp16[(size_t) n * K + k] = __hmul(__short2half_rn((short) q), s);
}

// Activation cast: float A -> half A. A is laid out [m][k] row-major.
__global__ void cast_f32_to_fp16(
        const float * __restrict__ A,
        __half      * __restrict__ A_fp16,
        int M, int K) {
    const int m = blockIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || k >= K) return;
    A_fp16[(size_t) m * K + k] = __float2half(A[(size_t) m * K + k]);
}

// =========================================================== Gemm wrapper
// Layout notes:
//   tc-grid stores W as W[n][k]: physically n outer, k inner (row-major over n,k).
//   GEMM computes C[m,n] = sum_k A[m,k] * W[n,k] = A * W^T.
//   For CUTLASS device::Gemm<A_layout, B_layout, C_layout>, take:
//     A = A_fp16 with LayoutA = RowMajor, shape (M, K)
//     B = W_fp16 with LayoutB = ColumnMajor, shape (K, N)  — because our row-major
//         (N, K) buffer IS col-major (K, N) when viewed transposed
//     C = C_f32  with LayoutC = RowMajor, shape (M, N)
//
// ThreadblockShape / WarpShape / InstructionShape are tuneable in P3.
// For P1 we pick one canonical sm_70 INT8 GEMM tile from CUTLASS examples.

template <
    int CTA_M, int CTA_N, int CTA_K,
    int W_M,   int W_N,   int W_K
>
struct Gemm70 {
    using Element = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Op = cutlass::gemm::device::Gemm<
        Element,  LayoutA,
        Element,  LayoutB,
        float,    LayoutC,
        float,    // accumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm70,
        cutlass::gemm::GemmShape<CTA_M, CTA_N, CTA_K>,
        cutlass::gemm::GemmShape<W_M,   W_N,   W_K>,
        cutlass::gemm::GemmShape<8, 8, 4>  // sm_70 m8n8k4
    >;

    static cutlass::Status run(
            const __half * A_fp16,
            const __half * W_fp16,
            float        * C,
            int M, int N, int K) {
        Op op;
        typename Op::Arguments args(
            {M, N, K},
            // A: row-major, ldA = K
            {(Element const *) A_fp16, K},
            // B (= W_fp16 viewed transposed): row-major buffer of shape (N, K) is
            // identical to column-major buffer of shape (K, N), ldB = K.
            {(Element const *) W_fp16, K},
            // C: row-major, ldC = N
            {C, N},
            // D = C overwrite
            {C, N},
            {1.0f, 0.0f}
        );
        return op(args);
    }
};

}  // namespace int8_cutlass

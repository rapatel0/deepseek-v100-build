// V100 (sm_70) m8n8k4 HMMA inline PTX wrappers.
//
// Source: turbomind core/mma.h (BSD-3-clause, OpenMMLab), adapted to use raw
// pointer-extracted uint32 arrays instead of their Array<> template.
//
// Per-thread fragment shapes for `mma_m8n8k4_row_col`:
//   A: 4 halves (= 2 × uint32 packed pairs)
//   B: 4 halves
//   D, C: 8 floats (FP32 accumulator)
//
// Output tile per call: 8×8. Stacked 2×(m) × 2×(n) × 4×(k-iters) = 16 calls per
// m16n16k16. Caller must populate A/B fragments with the right lane→element
// layout (see turbomind smem_copy_sm70.h:21-65).

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace tc_grid::mma_sm70 {

// d[8], c[8] are float[8]; a[4], b[4] are half[4]. The two uint32 in each operand
// are reinterpretations of the half pairs.
__device__ __forceinline__ void mma_m8n8k4_row_col(
        float* d, const half* a, const half* b, const float* c)
{
    const uint32_t* A = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(b);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
          "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
}

// In-place variant: accumulates into c[]. Saves a temporary + 8-float copy in
// the inner loop. Uses "+f" so the same register holds both old and new c.
__device__ __forceinline__ void mma_m8n8k4_row_col_acc(
        float* c, const half* a, const half* b)
{
    const uint32_t* A = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(b);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]));
}

__device__ __forceinline__ void mma_m8n8k4_row_row(
        float* d, const half* a, const half* b, const float* c)
{
    const uint32_t* A = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(b);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
          "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
}

// FP16 accumulator variant: mma.m8n8k4.f16.f16.f16.f16
// Per-thread fragment shapes per Volta PTX (m8n8k4 f16-acc):
//   A: 4 halves (2 uint32 packed pairs) — same as f32 acc
//   B: 4 halves                          — same as f32 acc
//   D, C: 8 halves = 4 .f16x2 = 4 uint32 — same ELEMENT count as f32 acc but
//                                          half the storage (16 vs 32 bytes/atom).
//
// On V100 (sm_70) the f16-acc instruction has 2× the tensor-pipe throughput of
// the f32-acc instruction (125 TF vs 62 TF peak). Lane→element mapping for the
// C/D operand is implementation-defined and may DIFFER from the f32 acc mapping
// (see memory item v100_wmma_half_float_frag_layout_mismatch and SPRINT-019
// P1.1 probe test). Callers must keep all c_frag arithmetic in half end-to-end
// and convert via SMEM round-trip only at epilogue.
//
// In-place accumulation: same +r trick as the f32 variant's +f.
__device__ __forceinline__ void mma_m8n8k4_row_col_acc_f16(
        half* c, const half* a, const half* b)
{
    const uint32_t* A = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(b);
    uint32_t* C = reinterpret_cast<uint32_t*>(c);   // 4 uint32 = 8 halves
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6,  %7},"
        "{%0,  %1,  %2,  %3};"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]));
}

}  // namespace tc_grid::mma_sm70

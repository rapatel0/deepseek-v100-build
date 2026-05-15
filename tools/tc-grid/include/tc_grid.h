// tc-grid: Tensor-core kernel grid search harness for V100 (sm_70).
//
// Hosts independent CUDA kernels for FP4/FP8/INT4/INT8 matmul + matvec on
// FP16 tensor cores. Compares two unpack strategies per format and grid
// searches BM/BN/BK/M/N/K configurations to characterize FLOPS and bandwidth
// regimes.
//
// Reference path: per-element FP32 dequant followed by cuBLAS FP32 SGEMM.
// Correctness contract: tolerance band on max_abs + p99_abs vs FP32 ref.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace tc_grid {

// ============================================================ formats =====

enum class Format {
    INT8,           // 1 byte per value, per-tile FP16 scale
    INT4,           // 4 bits per value, per-tile FP16 scale
    MXFP4,          // E2M1 nibble + per-32-block E8M0 (uint8) shared exponent
    F8_E4M3_B128,   // E4M3 byte + per-128-block E8M0 shared exponent
};

enum class UnpackPath {
    LUT,            // Lookup-table / per-element FP cast in registers/SMEM
    BITSHIFT,       // Bit-pack into FP16 mantissa or PRMT-parallel decode
};

const char * format_name(Format f);
const char * unpack_name(UnpackPath p);

// Compile-time block sizes per format.
constexpr int QK_INT8  = 32;    // matches block_q8_1 in ggml-common.h
constexpr int QK_INT4  = 32;    // 32 values per scale block
constexpr int QK_MXFP4 = 32;
constexpr int QK_F8    = 128;

// ====================================================== data generation ===

// Distributions that stress different parts of the FP precision envelope.
// All values are produced as FP32; quantizers map to target format.
enum class DataDist {
    UNIFORM_SMALL,      // U(-1, 1) -- benign baseline
    UNIFORM_WIDE,       // U(-8, 8) -- stresses dynamic range of FP4/INT4 saturation
    LOGNORMAL,          // mixed magnitudes ~ exp(N(0, 1.5))
    SPARSE_SPIKES,      // 90% small values, 10% large spikes
    ADVERSARIAL,        // designed to maximize cancellation in dot products
};

const char * dist_name(DataDist d);

// Generate an [rows x cols] FP32 matrix into d_out (device pointer).
void gen_matrix_f32(float * d_out, int rows, int cols, DataDist dist, uint64_t seed);

// Quantize an FP32 matrix [rows x K] (K-contiguous) into the target format.
// d_q is the packed quantized output (device).
// d_scales is the per-block scales (device, FP16 or uint8 depending on format).
//
// Layout convention: rows-major, K-contiguous. For each row we have
// K/blocksize blocks, each holding `blocksize` values + 1 scale.
//
// For MXFP4 / F8_E4M3_B128 the scale is E8M0 (uint8).
// For INT8 / INT4 the scale is FP16.
void quantize(Format fmt, const float * d_in, void * d_q_blocks,
              int rows, int K, cudaStream_t stream = nullptr);

// Round-trip dequant (host-callable helper for diagnostics / reference build).
// Writes FP32 reconstructed values back to d_out.
void dequant_reference(Format fmt, const void * d_q_blocks, float * d_out,
                       int rows, int K, cudaStream_t stream = nullptr);

// ============================================================== kernels ==

struct LaunchSpec {
    Format       format;
    UnpackPath   path;
    int          M;            // tile/batch rows
    int          N;            // output columns
    int          K;            // contract dimension
    int          BM, BN, BK;   // CTA tile
    int          warps;        // warps per CTA (1..8)
    int          split_k;      // split-K factor
    int          frag_n;       // N-fragments per warp (1 = single, 2/4 = multi-frag kernel)
    int          version;      // 0 = baseline BM=16; 1 = BM>16 multi-A-frag; 2 = +double-buffer +uint4
};

struct LaunchResult {
    bool         ok;           // launch + sync clean
    double       ms_mean;      // mean kernel time over N iters
    double       ms_min;       // min
    double       ms_max;       // max
    double       tflops;       // 2*M*N*K / time
    double       gbytes_per_s; // total bytes / time (weights + acts + dst)
    double       max_abs_err;  // max |result - reference|
    double       p99_abs_err;  // 99th percentile |result - reference|
    double       rel_err;      // mean |result - ref| / mean |ref|
    std::string  note;
};

// Per-format launchers. Each returns timing + tolerance using the provided
// spec and reference output. d_W is the quantized weights for one tile bucket;
// d_act is FP32 activations [M, K]; d_dst is FP32 output [M, N].
LaunchResult launch_int8(const LaunchSpec & s, const void * d_W, const float * d_act, float * d_dst, const float * d_ref);
LaunchResult launch_int4(const LaunchSpec & s, const void * d_W, const float * d_act, float * d_dst, const float * d_ref);
LaunchResult launch_fp4 (const LaunchSpec & s, const void * d_W, const float * d_act, float * d_dst, const float * d_ref);
LaunchResult launch_fp8 (const LaunchSpec & s, const void * d_W, const float * d_act, float * d_dst, const float * d_ref);

// ==========================================================  reference ===

// Compute the reference C[M, N] = act[M, K] * dequant(W)[K, N] at FP32
// using cuBLAS SGEMM. d_W_dequant is the FP32-reconstructed weights.
void reference_gemm_f32(const float * d_W_dequant, const float * d_act,
                        float * d_dst, int M, int N, int K, cudaStream_t stream = nullptr);

// Time cuBLAS FP16 GEMM (tensor-core path) at shape [M, N, K]. Returns the
// mean kernel time and computed TFLOPS. This establishes the V100 TC peak
// ceiling that the custom dequant kernels are aiming for.
struct CublasFp16Result { bool ok; double ms_mean; double tflops; double gbytes_per_s; };
CublasFp16Result cublas_fp16_gemm_bench(int M, int N, int K);

// Tolerance evaluation: compare d_test vs d_ref (both [M, N] FP32 on device).
// Returns max_abs, p99_abs, rel_err.
struct ToleranceStats { double max_abs; double p99_abs; double rel_err; };
ToleranceStats evaluate_tolerance(const float * d_test, const float * d_ref, size_t n);

// ============================================================= utility ===

#define TCG_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                    \
                    cudaGetErrorString(e), __FILE__, __LINE__, #call);         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

}  // namespace tc_grid

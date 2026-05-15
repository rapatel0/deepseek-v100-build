// tc-grid reference path: FP32 GEMM via cuBLAS, and tolerance evaluation.

#include "tc_grid.h"

#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tc_grid {

static cublasHandle_t g_handle = nullptr;

static void ensure_handle() {
    if (g_handle == nullptr) {
        cublasStatus_t s = cublasCreate(&g_handle);
        if (s != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed: %d\n", (int) s);
            std::exit(1);
        }
    }
}

// Compute C[M, N] = A[M, K] @ B[K, N] in FP32 using cuBLAS SGEMM.
// All matrices are row-major in the caller's view. cuBLAS expects column-major,
// so we use the standard trick: C^T = B^T @ A^T, computed as
// SGEMM(N, N, n=N, m=M, k=K, alpha=1, B, ldb=N, A, lda=K, beta=0, C, ldc=N)
// effectively producing a column-major C^T with leading dim N, which is the
// same byte layout as our row-major C[M, N]. So no extra transpose needed.
void reference_gemm_f32(const float * d_W_dequant, const float * d_act,
                        float * d_dst, int M, int N, int K, cudaStream_t stream) {
    ensure_handle();
    cublasSetStream(g_handle, stream);
    // CRITICAL: must reset math mode to DEFAULT, otherwise cuBLAS uses FP16
    // tensor cores for SGEMM (via TENSOR_OP_MATH that the FP16 ceiling bench
    // sets). Without this reset, the "FP32 reference" silently downgrades to
    // FP16 precision at large shapes, masking correctness differences between
    // our FP16-dequant kernels and the supposed ground truth.
    cublasSetMathMode(g_handle, CUBLAS_DEFAULT_MATH);
    const float alpha = 1.0f, beta = 0.0f;
    // d_act is [M, K] row-major == [K, M] col-major (when transposed).
    // d_W_dequant is [N, K] row-major == [K, N] col-major (transposed).
    // Compute d_dst[M, N] = d_act * d_W_dequant^T  (row-major).
    // Equivalent column-major: dst^T[N, M] = d_W_dequant[N, K] @ d_act^T[K, M].
    cublasStatus_t s = cublasSgemm(
        g_handle,
        CUBLAS_OP_T,        // op for d_W_dequant viewed as [K, N] col-major -> we want [N, K]
        CUBLAS_OP_N,        // op for d_act viewed col-major
        N,                  // rows of resulting col-major dst^T
        M,                  // cols of resulting col-major dst^T
        K,
        &alpha,
        d_W_dequant, K,     // [K, N] col-major == [N, K] row-major, but we passed OP_T so input is [K,N] cm
        d_act,        K,    // [K, M] col-major == [M, K] row-major
        &beta,
        d_dst,        N);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed: %d\n", (int) s);
        std::exit(1);
    }
}

// =========================================================== cublas FP16 ===

CublasFp16Result cublas_fp16_gemm_bench(int M, int N, int K) {
    CublasFp16Result R; R.ok = false; R.ms_mean = R.tflops = R.gbytes_per_s = 0;
    ensure_handle();
    cublasSetStream(g_handle, nullptr);
    cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH);

    __half * dA = nullptr; __half * dB = nullptr; __half * dC = nullptr;
    TCG_CHECK(cudaMalloc(&dA, (size_t) M * K * sizeof(__half)));
    TCG_CHECK(cudaMalloc(&dB, (size_t) K * N * sizeof(__half)));
    TCG_CHECK(cudaMalloc(&dC, (size_t) M * N * sizeof(__half)));
    TCG_CHECK(cudaMemset(dA, 0, (size_t) M * K * sizeof(__half)));
    TCG_CHECK(cudaMemset(dB, 0, (size_t) K * N * sizeof(__half)));

    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    constexpr int WARM = 3, ITERS = 8;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    double total_ms = 0;
    for (int i = 0; i < WARM + ITERS; ++i) {
        cudaEventRecord(e0);
        cublasStatus_t s = cublasGemmEx(
            g_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, CUDA_R_16F, K,
            dA, CUDA_R_16F, K,
            &beta,
            dC, CUDA_R_16F, N,
            CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (s != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasGemmEx failed: %d\n", (int) s);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            return R;
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        if (i >= WARM) {
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    R.ms_mean = total_ms / ITERS;
    R.tflops = (2.0 * (double) M * N * K) / (R.ms_mean * 1e-3) / 1e12;
    size_t bytes = (size_t) M * K * 2 + (size_t) K * N * 2 + (size_t) M * N * 2;
    R.gbytes_per_s = (double) bytes / (R.ms_mean * 1e-3) / 1e9;
    R.ok = true;

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return R;
}

// =============================================================== tolerance ===
//
// NOTE: the prior implementation used thrust::sort on the full error array;
// for large n (≥ ~10M elements at M=2048 N=K=7168 ≈ 14.7M) the sort silently
// failed (likely scratch-allocation OOM in thrust internals) and produced
// max_abs / p99 readings of 0.0 with no error signal. This implementation
// uses a custom reduction kernel for max + sums and a sampled p99 to be
// robust at any n.

// Single-block reduction (no atomics, no inter-block sync). Robust at any n.
// Each thread does a grid-stride loop over n/256 elements; block reduction
// in shared memory gives the final values. One block writes once to globals.
__global__ void k_stats_single_block(const float * __restrict__ a,
                                     const float * __restrict__ b,
                                     float * out_max,
                                     double * out_sum_err,
                                     double * out_sum_abs_ref,
                                     size_t n) {
    constexpr int kBlock = 256;
    __shared__ float  s_max[kBlock];
    __shared__ double s_se [kBlock];
    __shared__ double s_ar [kBlock];

    const int tid = threadIdx.x;
    float  local_max = 0.0f;
    double local_se  = 0.0;
    double local_ar  = 0.0;
    for (size_t i = tid; i < n; i += kBlock) {
        float ai = a[i], bi = b[i];
        float e  = fabsf(ai - bi);
        float ar = fabsf(bi);
        if (e > local_max) local_max = e;
        local_se += (double) e;
        local_ar += (double) ar;
    }
    s_max[tid] = local_max;
    s_se [tid] = local_se;
    s_ar [tid] = local_ar;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
            s_se [tid] += s_se [tid + s];
            s_ar [tid] += s_ar [tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_max         = s_max[0];
        *out_sum_err     = s_se [0];
        *out_sum_abs_ref = s_ar [0];
    }
}

__global__ void k_sample_err(const float * __restrict__ a,
                             const float * __restrict__ b,
                             float * __restrict__ sample,
                             size_t n, int sample_n, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= sample_n) return;
    // Hash-based uniform stride sampling.
    uint64_t h = ((uint64_t) tid * 2654435761ull) ^ seed;
    size_t i = (size_t)(h % (uint64_t) n);
    sample[tid] = fabsf(a[i] - b[i]);
}

ToleranceStats evaluate_tolerance(const float * d_test, const float * d_ref, size_t n) {
    // DIAG: peek at the first 4 values of d_test and d_ref to confirm
    // the inputs to the reduction are actually non-zero.

    float  * d_max     = nullptr;
    double * d_sum_err = nullptr;
    double * d_sum_abs = nullptr;
    TCG_CHECK(cudaMalloc(&d_max,     sizeof(float)));
    TCG_CHECK(cudaMalloc(&d_sum_err, sizeof(double)));
    TCG_CHECK(cudaMalloc(&d_sum_abs, sizeof(double)));
    TCG_CHECK(cudaMemset(d_max,     0, sizeof(float)));
    TCG_CHECK(cudaMemset(d_sum_err, 0, sizeof(double)));
    TCG_CHECK(cudaMemset(d_sum_abs, 0, sizeof(double)));

    // Single-block reduction. 1 block × 256 threads, each thread iterates
    // n/256 elements via grid-stride loop. No inter-block synchronization,
    // no atomics, robust at any n.
    k_stats_single_block<<<1, 256>>>(d_test, d_ref, d_max, d_sum_err, d_sum_abs, n);
    TCG_CHECK(cudaGetLastError());

    float  h_max     = 0.0f;
    double h_sum_err = 0.0;
    double h_sum_abs = 0.0;
    TCG_CHECK(cudaMemcpy(&h_max,     d_max,     sizeof(float),  cudaMemcpyDeviceToHost));
    TCG_CHECK(cudaMemcpy(&h_sum_err, d_sum_err, sizeof(double), cudaMemcpyDeviceToHost));
    TCG_CHECK(cudaMemcpy(&h_sum_abs, d_sum_abs, sizeof(double), cudaMemcpyDeviceToHost));

    TCG_CHECK(cudaFree(d_max));
    TCG_CHECK(cudaFree(d_sum_err));
    TCG_CHECK(cudaFree(d_sum_abs));

    // Sampled p99 (uniform random sample of n elements, host-side sort)
    constexpr int kSample = 8192;
    float * d_sample = nullptr;
    TCG_CHECK(cudaMalloc(&d_sample, kSample * sizeof(float)));
    k_sample_err<<<(kSample + 255) / 256, 256>>>(d_test, d_ref, d_sample, n, kSample, 0xC0FFEEULL);
    TCG_CHECK(cudaGetLastError());
    std::vector<float> h_sample(kSample);
    TCG_CHECK(cudaMemcpy(h_sample.data(), d_sample, kSample * sizeof(float), cudaMemcpyDeviceToHost));
    TCG_CHECK(cudaFree(d_sample));
    std::sort(h_sample.begin(), h_sample.end());
    float h_p99 = h_sample[(int)(kSample * 0.99f)];

    ToleranceStats out;
    out.max_abs = (double) h_max;
    out.p99_abs = (double) h_p99;
    out.rel_err = (h_sum_abs > 0.0) ? (h_sum_err / h_sum_abs) : 0.0;
    return out;
}

}  // namespace tc_grid

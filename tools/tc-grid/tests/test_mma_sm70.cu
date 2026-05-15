// Standalone test for the m8n8k4 inline PTX wrapper on V100 (sm_70).
//
// Goal: validate that mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 compiles,
// links, and runs in our build environment. We do NOT yet verify per-element
// correctness (that requires nailing the lane→element mapping, which is the
// Step 2 work). We DO verify that:
//   1. Setting all of A=1, B=1, C=initial: D accumulates a known constant (every
//      m8n8k4 contributes 4 K-products of 1*1 = 4 → D = C_init + 4 per element).
//   2. Setting A=0: D == C (no contribution).
//
// These two tests together are a strong build/PTX sanity check.

#include "../kernels/mma_sm70.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

using tc_grid::mma_sm70::mma_m8n8k4_row_col;

// One warp executes one m8n8k4. Each thread provides:
//   A: 4 halves (2 uint32)
//   B: 4 halves
//   C: 8 floats
// Output: D = A * B^T + C (per the .row.col layout), 8x8 result spread across
// 32 lanes × 8 floats each.

__global__ void mma_test_kernel(
        float* d_out,     // [32 * 8] — per-lane D fragments
        const half* a_in, // [32 * 4]
        const half* b_in, // [32 * 4]
        const float* c_in // [32 * 8]
) {
    const int lane = threadIdx.x;
    half  a[4];
    half  b[4];
    float c[8];
    float d[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = a_in[lane * 4 + i];
    #pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = b_in[lane * 4 + i];
    #pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = c_in[lane * 8 + i];

    mma_m8n8k4_row_col(d, a, b, c);

    #pragma unroll
    for (int i = 0; i < 8; ++i) d_out[lane * 8 + i] = d[i];
}

static int run_case(const char* label,
                    float a_val, float b_val, float c_val,
                    float expected_min, float expected_max) {
    const int N_LANES = 32;
    const int A_SZ = N_LANES * 4;
    const int B_SZ = N_LANES * 4;
    const int CD_SZ = N_LANES * 8;

    half*  h_a = (half*) malloc(A_SZ * sizeof(half));
    half*  h_b = (half*) malloc(B_SZ * sizeof(half));
    float* h_c = (float*) malloc(CD_SZ * sizeof(float));
    float* h_d = (float*) malloc(CD_SZ * sizeof(float));
    for (int i = 0; i < A_SZ; ++i) h_a[i] = __float2half(a_val);
    for (int i = 0; i < B_SZ; ++i) h_b[i] = __float2half(b_val);
    for (int i = 0; i < CD_SZ; ++i) h_c[i] = c_val;

    half *d_a, *d_b; float *d_c, *d_d;
    cudaMalloc(&d_a, A_SZ * sizeof(half));
    cudaMalloc(&d_b, B_SZ * sizeof(half));
    cudaMalloc(&d_c, CD_SZ * sizeof(float));
    cudaMalloc(&d_d, CD_SZ * sizeof(float));
    cudaMemcpy(d_a, h_a, A_SZ * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, B_SZ * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, CD_SZ * sizeof(float), cudaMemcpyHostToDevice);

    mma_test_kernel<<<1, 32>>>(d_d, d_a, d_b, d_c);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[FAIL %s] cuda error: %s\n", label, cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_d, d_d, CD_SZ * sizeof(float), cudaMemcpyDeviceToHost);

    int n_in_range = 0, n_out = 0;
    float minv = h_d[0], maxv = h_d[0];
    for (int i = 0; i < CD_SZ; ++i) {
        if (h_d[i] < minv) minv = h_d[i];
        if (h_d[i] > maxv) maxv = h_d[i];
        if (h_d[i] >= expected_min - 1e-3 && h_d[i] <= expected_max + 1e-3) ++n_in_range;
        else ++n_out;
    }
    printf("[%s %s] D range [%.4f, %.4f], expected [%.4f, %.4f], %d/%d in range\n",
           (n_out == 0) ? " OK " : "FAIL",
           label, minv, maxv, expected_min, expected_max, n_in_range, CD_SZ);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    free(h_a); free(h_b); free(h_c); free(h_d);
    return (n_out == 0) ? 0 : 1;
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s, cc=%d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 7) {
        printf("[SKIP] sm_70 required for m8n8k4 mma.sync\n");
        return 0;
    }

    int rc = 0;
    // Case 1: A=1, B=1, C=0 → D should be 4 everywhere (each output is sum of 4 K-products).
    rc += run_case("A=1,B=1,C=0", 1.0f, 1.0f, 0.0f, 4.0f, 4.0f);
    // Case 2: A=1, B=1, C=10 → D should be 14 everywhere.
    rc += run_case("A=1,B=1,C=10", 1.0f, 1.0f, 10.0f, 14.0f, 14.0f);
    // Case 3: A=0, B=1, C=7 → D should be 7 everywhere (no contribution).
    rc += run_case("A=0,B=1,C=7", 0.0f, 1.0f, 7.0f, 7.0f, 7.0f);
    // Case 4: A=2, B=3, C=0 → D should be 24 (each = 4 * 2 * 3).
    rc += run_case("A=2,B=3,C=0", 2.0f, 3.0f, 0.0f, 24.0f, 24.0f);

    if (rc == 0) printf("\nALL PASS — m8n8k4 PTX wrapper functional on this device.\n");
    else         printf("\n%d FAIL — investigate.\n", rc);
    return rc;
}

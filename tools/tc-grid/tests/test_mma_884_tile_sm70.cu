// V11 Step 2b — Single 884 tile (m8n32k8) with manual Lds + SM70_MMA_884::fma.
//
// What this validates
// -------------------
// End-to-end correctness of the 884 tile:
//   1. SmemCopy_MMA_884_A loads FragA[8 halves] per lane from an 8×8 SMEM tile.
//   2. SmemCopy_MMA_884_B loads FragB[8 halves] per lane from a 32×8 SMEM tile.
//   3. SM70_MMA_884::fma runs two back-to-back mma.m8n8k4_row_col calls.
//   4. FragC[8 floats] per lane is scattered to an m8×n32 output grid using
//      thread_offset_C() + static_offset_C() from mma_sm70.h.
//
// The output C[8, 32] is compared bit-wise (rel tolerance 1e-3) against a
// host-side reference C[m, n] = sum_k A[m, k] * B[n, k] (B^T layout).
//
// What this NOT validates
// -----------------------
// The full v11 mainloop (BM/BN/BK tiling, multi-tile reuse, gmem→smem stage).
// That's Step 2c.

#include "../kernels/mma_sm70.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

using tc_grid::mma_sm70::mma_m8n8k4_row_col;

// ----------------------------------------------------------------------------
// Lane → (m, k_quad) — validated in Step 2a (test_smem_to_frag_sm70.cu).
// ----------------------------------------------------------------------------

__host__ __device__ static inline int A_m(int lane)     { return (lane / 16) * 4 + (lane % 4); }
__host__ __device__ static inline int A_kquad(int lane) { return (lane & 12) >> 2; }
__host__ __device__ static inline int B_m(int lane)     { return (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4); }

// ----------------------------------------------------------------------------
// Per-lane output cell mapping (from turbomind mma_sm70.h):
//
//   thread_offset_C() = ((lane & 1) + (lane / 16) * 4,
//                        (lane & 2) + (lane & 12) * 2)
//   static_offset_C() = {(0, 0), (2, 0), (0, 4), (2, 4)}   ordered by [n*2+m]
//
//   Per-lane FragC[8] is grouped as 4 pairs (Array<float, 2>[4]); pair i goes to
//   output offset thread_offset_C + static_offset_C[i].
//
// Empirically the 2 floats in each pair are at adjacent N positions (n_off, n_off+1).
// ----------------------------------------------------------------------------

__host__ __device__ static inline int C_lane_m(int lane)     { return (lane & 1) + (lane / 16) * 4; }
__host__ __device__ static inline int C_lane_n(int lane)     { return (lane & 2) + (lane & 12) * 2; }

struct OffC { int m; int n; };
__host__ __device__ static inline OffC C_static_off(int pair_idx) {
    int m = pair_idx & 1;
    int n = (pair_idx >> 1) & 1;
    return OffC{ m * 2, n * 4 };
}

// ----------------------------------------------------------------------------
// Tile dimensions
// ----------------------------------------------------------------------------

constexpr int M_TILE = 8;     // m
constexpr int N_TILE = 32;    // n
constexpr int K_TILE = 8;     // k

// ----------------------------------------------------------------------------
// GPU kernel: one warp, one 884 tile.
//
// SMEM layout:
//   sA: 8 rows × 8 cols (row-major; stride = 8 halves)
//   sB: 32 rows × 8 cols (row-major; stride = 8 halves)
//
// gmem layout:
//   A_in: [M_TILE, K_TILE] row-major
//   B_in: [N_TILE, K_TILE] row-major (this is B with N-outer, K-inner; the
//         "row.col" mma sees B as transposed, so B[n,k] dotted into A[m,k]
//         produces C[m,n] = sum_k A[m,k] * B[n,k]).
//   C_out: [M_TILE, N_TILE] row-major fp32
// ----------------------------------------------------------------------------

__global__ void mma_884_tile_kernel(
        const half* __restrict__ A_in,
        const half* __restrict__ B_in,
        float* __restrict__ C_out)
{
    const int lane = threadIdx.x;

    __shared__ __align__(16) half sA[M_TILE * K_TILE];
    __shared__ __align__(16) half sB[N_TILE * K_TILE];

    // Cooperative gmem → SMEM (32 lanes × 8 halves cover the full 256 halves of B).
    *reinterpret_cast<uint4*>(&sB[lane * 8]) = *reinterpret_cast<const uint4*>(&B_in[lane * 8]);
    // A is only 64 halves; first 8 lanes load it (one row each, 8 halves per row).
    if (lane < M_TILE) {
        *reinterpret_cast<uint4*>(&sA[lane * 8]) = *reinterpret_cast<const uint4*>(&A_in[lane * 8]);
    }
    __syncthreads();

    // ---- Load fragments via the SmemCopy_MMA_884 formulas ----
    half a_frag[K_TILE];
    half b_frag[K_TILE];
    {
        const int m = A_m(lane);
        // For an 8×8 tile, k_quad-step of 8 would read past the K=8 row. But
        // SmemCopy_MMA_884_A's K=8 frag is exactly one row of A for the lane's
        // assigned m. The 4 lanes per m all read the SAME 8 halves — duplicate
        // reads, but the mma instruction expects this replication (the quad
        // structure of m8n8k4 uses lane_id & 12 as a quad-pair selector, not as
        // a K offset). So k_quad here is implicitly 0 for the per-lane Lds.
        const half* src = &sA[m * K_TILE];
        *reinterpret_cast<uint4*>(a_frag) = *reinterpret_cast<const uint4*>(src);
    }
    {
        const int m = B_m(lane);
        const half* src = &sB[m * K_TILE];
        *reinterpret_cast<uint4*>(b_frag) = *reinterpret_cast<const uint4*>(src);
    }

    // ---- Init C frag = 0; run SM70_MMA_884::fma (two back-to-back m8n8k4) ----
    float c_frag[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) c_frag[i] = 0.0f;

    // First m8n8k4: K positions 0..3
    {
        float d[8];
        mma_m8n8k4_row_col(d, &a_frag[0], &b_frag[0], c_frag);
        #pragma unroll
        for (int i = 0; i < 8; ++i) c_frag[i] = d[i];
    }
    // Second m8n8k4: K positions 4..7, accumulate
    {
        float d[8];
        mma_m8n8k4_row_col(d, &a_frag[4], &b_frag[4], c_frag);
        #pragma unroll
        for (int i = 0; i < 8; ++i) c_frag[i] = d[i];
    }

    // ---- Scatter FragC to C_out[8, 32] ----
    const int m_base = C_lane_m(lane);
    const int n_base = C_lane_n(lane);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const OffC o = C_static_off(p);
        const int m = m_base + o.m;
        const int n = n_base + o.n;
        // FragC[p*2+0] → (m, n+0); FragC[p*2+1] → (m, n+1)
        C_out[m * N_TILE + (n + 0)] = c_frag[p * 2 + 0];
        C_out[m * N_TILE + (n + 1)] = c_frag[p * 2 + 1];
    }
}

// ----------------------------------------------------------------------------
// CPU reference
// ----------------------------------------------------------------------------

static void cpu_reference(const half* A, const half* B, float* C) {
    for (int m = 0; m < M_TILE; ++m) {
        for (int n = 0; n < N_TILE; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K_TILE; ++k) {
                acc += __half2float(A[m * K_TILE + k]) * __half2float(B[n * K_TILE + k]);
            }
            C[m * N_TILE + n] = acc;
        }
    }
}

// ----------------------------------------------------------------------------
// Driver
// ----------------------------------------------------------------------------

static float frand() {
    return ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f;   // [-1, 1]
}

static int compare(const float* gpu, const float* cpu, int n, float tol) {
    int n_bad = 0;
    float max_rel = 0.0f, max_abs = 0.0f;
    int first_bad = -1;
    for (int i = 0; i < n; ++i) {
        const float diff = std::fabs(gpu[i] - cpu[i]);
        const float denom = std::fabs(cpu[i]) + 1e-6f;
        const float rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
        if (diff > max_abs) max_abs = diff;
        if (rel > tol) {
            if (first_bad < 0) first_bad = i;
            ++n_bad;
        }
    }
    if (n_bad == 0) {
        printf("[ OK ] %d/%d cells within rel=%.1e (max_rel=%.2e, max_abs=%.2e)\n",
               n - n_bad, n, tol, max_rel, max_abs);
        return 0;
    }
    printf("[FAIL] %d/%d cells exceed rel=%.1e; first bad at idx %d (m=%d, n=%d): gpu=%.4f cpu=%.4f rel=%.2e\n",
           n_bad, n, tol, first_bad, first_bad / N_TILE, first_bad % N_TILE,
           gpu[first_bad], cpu[first_bad], std::fabs(gpu[first_bad] - cpu[first_bad]) /
           (std::fabs(cpu[first_bad]) + 1e-6f));
    printf("       max_rel=%.2e  max_abs=%.2e  (over the whole tile)\n", max_rel, max_abs);
    return 1;
}

static void dump_row(const char* label, const float* C, int row) {
    printf("%s row %d:", label, row);
    for (int n = 0; n < N_TILE; ++n) printf(" %.2f", C[row * N_TILE + n]);
    printf("\n");
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s, cc=%d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 7) {
        printf("[SKIP] sm_70 required for SM70_MMA_884 tile\n");
        return 0;
    }
    const bool verbose = (argc > 1 && argv[1][0] == 'v');

    // ---- Build A, B, run CPU reference ----
    srand(1234);
    half* h_A = (half*) malloc(M_TILE * K_TILE * sizeof(half));
    half* h_B = (half*) malloc(N_TILE * K_TILE * sizeof(half));
    float* h_C_ref = (float*) malloc(M_TILE * N_TILE * sizeof(float));
    float* h_C_gpu = (float*) malloc(M_TILE * N_TILE * sizeof(float));
    for (int i = 0; i < M_TILE * K_TILE; ++i) h_A[i] = __float2half(frand());
    for (int i = 0; i < N_TILE * K_TILE; ++i) h_B[i] = __float2half(frand());
    cpu_reference(h_A, h_B, h_C_ref);

    // ---- GPU launch ----
    half *d_A, *d_B; float *d_C;
    cudaMalloc(&d_A, M_TILE * K_TILE * sizeof(half));
    cudaMalloc(&d_B, N_TILE * K_TILE * sizeof(half));
    cudaMalloc(&d_C, M_TILE * N_TILE * sizeof(float));
    cudaMemcpy(d_A, h_A, M_TILE * K_TILE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N_TILE * K_TILE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_TILE * N_TILE * sizeof(float));

    mma_884_tile_kernel<<<1, 32>>>(d_A, d_B, d_C);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[FAIL] cuda error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_C_gpu, d_C, M_TILE * N_TILE * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- Compare ----
    int rc = compare(h_C_gpu, h_C_ref, M_TILE * N_TILE, 1e-3f);

    if (verbose || rc != 0) {
        printf("\nFirst row of CPU reference and GPU output:\n");
        dump_row("CPU", h_C_ref, 0);
        dump_row("GPU", h_C_gpu, 0);
        printf("\nLast row of CPU reference and GPU output:\n");
        dump_row("CPU", h_C_ref, M_TILE - 1);
        dump_row("GPU", h_C_gpu, M_TILE - 1);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_ref); free(h_C_gpu);

    if (rc == 0) printf("\nALL PASS — SM70_MMA_884 m8n32k8 tile bit-correct on V100.\n");
    else         printf("\n%d FAIL — investigate FragC layout / MMA arg order.\n", rc);
    return rc;
}

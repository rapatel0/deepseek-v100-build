// SPRINT-019 P1.1 — Tier 1 FP16-accumulator m8n8k4 atom test.
//
// Goal: empirically derive the lane→element mapping for the C/D operand of
// `mma.m8n8k4.row.col.f16.f16.f16.f16` on V100 (sm_70). FP32-acc mapping is
// validated in test_mma_884_tile_sm70.cu; FP16-acc mapping is implementation-
// defined per Volta PTX and may differ.
//
// Setup mirrors test_mma_884_tile_sm70: m=8, n=32, k=8 via two back-to-back
// m8n8k4 atoms (K=0..3, then K=4..7). The same A,B inputs feed two kernels:
//   - kern_fp32: ground truth via the validated f32-acc lane mapping.
//   - kern_fp16: f16-acc path. Each lane writes its 8 c_frag halves to gmem
//                indexed by lane; host derives (lane,slot)→(m,n) by matching
//                values against the f32-acc scatter result.
//
// Probe data: D[m, n] = m * 32 + n  (0..255), unique per cell, integer,
// fits in fp16 (max value 7*32+31 = 255).
//
// Strategy: a single set of inputs (A,B) is constructed so that
//   D[m,n] = m * N_TILE + n  with N_TILE = 32.
// We achieve this with A[m, k] depending on m only and B[n, k] depending on n
// only over different k positions. See fill_probe().

#include "../kernels/mma_sm70.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

using tc_grid::mma_sm70::mma_m8n8k4_row_col;
using tc_grid::mma_sm70::mma_m8n8k4_row_col_acc_f16;

// SmemCopy_MMA_884 lane→(m,k) for A and lane→(n,k) for B; validated for the
// 8×8 A and 32×8 B tiles in test_mma_884_tile_sm70.cu (Step 2a / 2b).
__host__ __device__ static inline int A_m_lane(int lane)  { return (lane / 16) * 4 + (lane % 4); }
__host__ __device__ static inline int B_n_lane(int lane)  { return (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4); }
// FP32-acc validated cell mapping for the m8n32k8 tile (4 atoms in n direction).
__host__ __device__ static inline int C_m_lane_f32(int lane) { return (lane & 1) + (lane / 16) * 4; }
__host__ __device__ static inline int C_n_lane_f32(int lane) { return (lane & 2) + (lane & 12) * 2; }
struct OffC { int m; int n; };
__host__ __device__ static inline OffC C_static_off(int p) {
    return OffC{ (p & 1) * 2, ((p >> 1) & 1) * 4 };
}

constexpr int M_TILE = 8, N_TILE = 32, K_TILE = 8;

// ----------------------------------------------------------------------------
// FP32-acc kernel: two back-to-back m8n8k4 calls; same scatter as
// test_mma_884_tile_sm70. Per-lane c_frag also dumped to gmem for inspection.
// ----------------------------------------------------------------------------
__global__ void kern_fp32(
        const half* __restrict__ A_in,
        const half* __restrict__ B_in,
        float* __restrict__ D_lane,    // [32, 8] per-lane c_frag
        float* __restrict__ D_mat)     // [M_TILE, N_TILE] scattered output
{
    const int lane = threadIdx.x;
    __shared__ __align__(16) half sA[M_TILE * K_TILE];
    __shared__ __align__(16) half sB[N_TILE * K_TILE];

    // 32 lanes cover B's 256 halves (uint4 = 8 halves each).
    *reinterpret_cast<uint4*>(&sB[lane * 8]) = *reinterpret_cast<const uint4*>(&B_in[lane * 8]);
    // 8 lanes cover A's 64 halves.
    if (lane < M_TILE) {
        *reinterpret_cast<uint4*>(&sA[lane * 8]) = *reinterpret_cast<const uint4*>(&A_in[lane * 8]);
    }
    __syncthreads();

    half a_frag[K_TILE], b_frag[K_TILE];
    {
        const int m = A_m_lane(lane);
        *reinterpret_cast<uint4*>(a_frag) = *reinterpret_cast<const uint4*>(&sA[m * K_TILE]);
    }
    {
        const int n = B_n_lane(lane);
        *reinterpret_cast<uint4*>(b_frag) = *reinterpret_cast<const uint4*>(&sB[n * K_TILE]);
    }
    float c_frag[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) c_frag[i] = 0.0f;

    {
        float d[8];
        mma_m8n8k4_row_col(d, &a_frag[0], &b_frag[0], c_frag);
        #pragma unroll
        for (int i = 0; i < 8; ++i) c_frag[i] = d[i];
    }
    {
        float d[8];
        mma_m8n8k4_row_col(d, &a_frag[4], &b_frag[4], c_frag);
        #pragma unroll
        for (int i = 0; i < 8; ++i) c_frag[i] = d[i];
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) D_lane[lane * 8 + i] = c_frag[i];

    const int m_base = C_m_lane_f32(lane);
    const int n_base = C_n_lane_f32(lane);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        OffC o = C_static_off(p);
        const int m = m_base + o.m;
        const int n = n_base + o.n;
        D_mat[m * N_TILE + (n + 0)] = c_frag[p * 2 + 0];
        D_mat[m * N_TILE + (n + 1)] = c_frag[p * 2 + 1];
    }
}

// ----------------------------------------------------------------------------
// FP16-acc kernel: same inputs, same two-atom flow, but c_frag is half[8] and
// uses mma_m8n8k4_row_col_acc_f16. Per-lane halves go to gmem for probe.
// ----------------------------------------------------------------------------
__global__ void kern_fp16(
        const half* __restrict__ A_in,
        const half* __restrict__ B_in,
        half* __restrict__ D_lane)    // [32, 8] per-lane c_frag in halves
{
    const int lane = threadIdx.x;
    __shared__ __align__(16) half sA[M_TILE * K_TILE];
    __shared__ __align__(16) half sB[N_TILE * K_TILE];

    *reinterpret_cast<uint4*>(&sB[lane * 8]) = *reinterpret_cast<const uint4*>(&B_in[lane * 8]);
    if (lane < M_TILE) {
        *reinterpret_cast<uint4*>(&sA[lane * 8]) = *reinterpret_cast<const uint4*>(&A_in[lane * 8]);
    }
    __syncthreads();

    half a_frag[K_TILE], b_frag[K_TILE];
    {
        const int m = A_m_lane(lane);
        *reinterpret_cast<uint4*>(a_frag) = *reinterpret_cast<const uint4*>(&sA[m * K_TILE]);
    }
    {
        const int n = B_n_lane(lane);
        *reinterpret_cast<uint4*>(b_frag) = *reinterpret_cast<const uint4*>(&sB[n * K_TILE]);
    }

    half c_frag[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) c_frag[i] = __float2half(0.0f);

    mma_m8n8k4_row_col_acc_f16(c_frag, &a_frag[0], &b_frag[0]);
    mma_m8n8k4_row_col_acc_f16(c_frag, &a_frag[4], &b_frag[4]);

    #pragma unroll
    for (int i = 0; i < 8; ++i) D_lane[lane * 8 + i] = c_frag[i];
}

// ----------------------------------------------------------------------------
// Probe: D[m, n] = m * N_TILE + n. Achieved with:
//   A[m, 0] = m,  A[m, 1] = 1,  A[m, k>=2] = 0
//   B[n, 0] = N_TILE,  B[n, 1] = n,  B[n, k>=2] = 0
//   D[m, n] = A[m,0]*B[n,0] + A[m,1]*B[n,1] = m * N_TILE + n  ✓
// Max value: 7 * 32 + 31 = 255 — comfortably under FP16's exact-int range.
// ----------------------------------------------------------------------------
static void fill_probe(half* A, half* B) {
    for (int m = 0; m < M_TILE; ++m) {
        for (int k = 0; k < K_TILE; ++k) {
            float v = 0;
            if (k == 0) v = (float) m;
            else if (k == 1) v = 1;
            A[m * K_TILE + k] = __float2half(v);
        }
    }
    for (int n = 0; n < N_TILE; ++n) {
        for (int k = 0; k < K_TILE; ++k) {
            float v = 0;
            if (k == 0) v = (float) N_TILE;
            else if (k == 1) v = (float) n;
            B[n * K_TILE + k] = __float2half(v);
        }
    }
}

static void cpu_reference(const half* A, const half* B, float* D) {
    for (int m = 0; m < M_TILE; ++m) {
        for (int n = 0; n < N_TILE; ++n) {
            float acc = 0;
            for (int k = 0; k < K_TILE; ++k)
                acc += __half2float(A[m * K_TILE + k]) * __half2float(B[n * K_TILE + k]);
            D[m * N_TILE + n] = acc;
        }
    }
}

// Given an fp16 per-lane dump, decode each slot's (m,n) by matching the
// half value against the probe target D[m,n] = m * N_TILE + n.
static int derive_fp16_mapping(const half* D_fp16_lane, int (*out_m)[8], int (*out_n)[8]) {
    int bad = 0;
    for (int lane = 0; lane < 32; ++lane) {
        for (int s = 0; s < 8; ++s) {
            const int v = (int) __half2float(D_fp16_lane[lane * 8 + s]);
            if (v < 0 || v >= M_TILE * N_TILE) {
                out_m[lane][s] = -1; out_n[lane][s] = -1; ++bad;
            } else {
                out_m[lane][s] = v / N_TILE;
                out_n[lane][s] = v % N_TILE;
            }
        }
    }
    return bad;
}

// Given the mapping, verify the fp16 kernel produces correct values for
// arbitrary inputs (CPU reference comparison). Production-style tolerance:
// fail only when BOTH rel > tol_rel AND abs > tol_abs (small-magnitude
// values get a pass on rel since the denominator is what blows it up; large
// values get a pass on abs since rel is the meaningful gate).
static int verify_with_mapping(const half* D_lane,
                               int (*map_m)[8], int (*map_n)[8],
                               const float* D_ref,
                               const char* tag, float tol_rel, float tol_abs) {
    int n_bad = 0;
    float max_rel = 0, max_abs = 0;
    bool covered[M_TILE * N_TILE] = {0};
    for (int lane = 0; lane < 32; ++lane) {
        for (int s = 0; s < 8; ++s) {
            const int m = map_m[lane][s], n = map_n[lane][s];
            if (m < 0) continue;
            const float gpu = __half2float(D_lane[lane * 8 + s]);
            const float cpu = D_ref[m * N_TILE + n];
            const float a = fabsf(gpu - cpu);
            const float r = a / (fabsf(cpu) + 1e-6f);
            if (r > max_rel) max_rel = r;
            if (a > max_abs) max_abs = a;
            if (r > tol_rel && a > tol_abs) ++n_bad;
            covered[m * N_TILE + n] = true;
        }
    }
    int uncovered = 0;
    for (int i = 0; i < M_TILE * N_TILE; ++i) if (!covered[i]) ++uncovered;
    printf("  [%s] rel_max=%.2e abs_max=%.2e bad=%d uncovered=%d\n",
           tag, max_rel, max_abs, n_bad, uncovered);
    return (n_bad > 0 || uncovered > 0) ? 1 : 0;
}

static void run_fp16(const half* hA, const half* hB, half* host_lane) {
    half *dA, *dB, *dLane;
    cudaMalloc(&dA, M_TILE * K_TILE * sizeof(half));
    cudaMalloc(&dB, N_TILE * K_TILE * sizeof(half));
    cudaMalloc(&dLane, 32 * 8 * sizeof(half));
    cudaMemcpy(dA, hA, M_TILE * K_TILE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N_TILE * K_TILE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dLane, 0, 32 * 8 * sizeof(half));
    kern_fp16<<<1, 32>>>(dA, dB, dLane);
    cudaDeviceSynchronize();
    cudaMemcpy(host_lane, dLane, 32 * 8 * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dLane);
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s, cc=%d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 7) { printf("[SKIP] sm_70 required\n"); return 0; }
    const bool verbose = (argc > 1 && argv[1][0] == 'v');

    half hA[M_TILE * K_TILE], hB[N_TILE * K_TILE];
    fill_probe(hA, hB);

    // Run FP32-acc and capture per-lane dump + scattered matrix.
    float hD32_lane[32 * 8], hD32_mat[M_TILE * N_TILE];
    {
        half *dA, *dB; float *dLane, *dMat;
        cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB));
        cudaMalloc(&dLane, sizeof(hD32_lane)); cudaMalloc(&dMat, sizeof(hD32_mat));
        cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
        cudaMemset(dMat, 0, sizeof(hD32_mat));
        kern_fp32<<<1, 32>>>(dA, dB, dLane, dMat);
        cudaDeviceSynchronize();
        cudaMemcpy(hD32_lane, dLane, sizeof(hD32_lane), cudaMemcpyDeviceToHost);
        cudaMemcpy(hD32_mat, dMat, sizeof(hD32_mat), cudaMemcpyDeviceToHost);
        cudaFree(dA); cudaFree(dB); cudaFree(dLane); cudaFree(dMat);
    }

    // FP32 sanity: the scattered matrix should equal the probe target.
    int sane = 0;
    for (int m = 0; m < M_TILE; ++m) {
        for (int n = 0; n < N_TILE; ++n) {
            float exp = (float)(m * N_TILE + n);
            float got = hD32_mat[m * N_TILE + n];
            if (fabsf(got - exp) > 1e-3f) {
                if (sane < 3)
                    printf("  [FP32 sanity FAIL] D[%d,%d] = %.2f (expected %.2f)\n", m, n, got, exp);
                ++sane;
            }
        }
    }
    if (sane > 0) {
        printf("[FAIL] FP32-acc kernel did not reproduce probe matrix (%d/%d cells bad)\n",
               sane, M_TILE * N_TILE);
        return 1;
    }
    printf("[ OK ] FP32-acc kernel reproduces probe D[m,n] = m*%d + n across all %d cells.\n",
           N_TILE, M_TILE * N_TILE);

    // FP16 probe.
    half hD16_lane[32 * 8];
    run_fp16(hA, hB, hD16_lane);

    // Derive mapping by value lookup.
    int map_m[32][8], map_n[32][8];
    int bad = derive_fp16_mapping(hD16_lane, map_m, map_n);
    if (bad) {
        printf("[FAIL] FP16-acc probe: %d/256 slots produced out-of-range values.\n", bad);
        if (verbose) {
            for (int lane = 0; lane < 32; ++lane) {
                printf("  lane %2d:", lane);
                for (int s = 0; s < 8; ++s) printf(" %.0f", __half2float(hD16_lane[lane*8+s]));
                printf("\n");
            }
        }
        return 1;
    }
    printf("[ OK ] FP16-acc probe: all 256 slot values match a valid D[m,n] cell.\n");

    if (verbose) {
        printf("\nDerived FP16-acc (lane, slot) → (m, n):\n");
        for (int lane = 0; lane < 32; ++lane) {
            printf("  lane %2d:", lane);
            for (int s = 0; s < 8; ++s)
                printf("  s%d=(%d,%d)", s, map_m[lane][s], map_n[lane][s]);
            printf("\n");
        }
    }

    // Adversarial verification. Production tolerance is `rel ≤ 1e-3 ∨ abs ≤ 0.1`
    // (sprint §1.2). For integer-valued inputs we require bit-exact (tol=0).
    auto run_test = [&](const char* tag, const half* A, const half* B,
                        float tol_rel, float tol_abs) -> int {
        half D_lane[32 * 8];
        run_fp16(A, B, D_lane);
        float D_ref[M_TILE * N_TILE];
        cpu_reference(A, B, D_ref);
        return verify_with_mapping(D_lane, map_m, map_n, D_ref, tag, tol_rel, tol_abs);
    };

    int rc = 0;
    // 1) zeros — bit-exact required
    {
        half A[M_TILE * K_TILE] = {}, B[N_TILE * K_TILE] = {};
        rc |= run_test("zeros", A, B, 0.0f, 0.0f);
    }
    // 2) identity-ish along K — bit-exact required
    {
        half A[M_TILE * K_TILE] = {}, B[N_TILE * K_TILE] = {};
        for (int m = 0; m < M_TILE; ++m) for (int k = 0; k < K_TILE; ++k)
            A[m*K_TILE+k] = __float2half(m == k ? 1.0f : 0.0f);
        for (int n = 0; n < N_TILE; ++n) for (int k = 0; k < K_TILE; ++k)
            B[n*K_TILE+k] = __float2half(n == k ? 1.0f : 0.0f);
        rc |= run_test("identity-ish", A, B, 0.0f, 0.0f);
    }
    // 3) sign-heavy: ±1 random — integer values, bit-exact required
    {
        half A[M_TILE * K_TILE], B[N_TILE * K_TILE];
        srand(0xCAFE);
        for (int i = 0; i < M_TILE * K_TILE; ++i) A[i] = __float2half((rand() & 1) ? 1.0f : -1.0f);
        for (int i = 0; i < N_TILE * K_TILE; ++i) B[i] = __float2half((rand() & 1) ? 1.0f : -1.0f);
        rc |= run_test("sign-heavy", A, B, 0.0f, 0.0f);
    }
    // 4) saturation-adjacent: |x|~200, |y|~80 → accumulator overflows fp16.
    //    This is the FP16-acc dynamic-range CEILING; we record it as
    //    informational (not a pass/fail gate). Document in V12-DESIGN.md.
    {
        half A[M_TILE * K_TILE], B[N_TILE * K_TILE];
        srand(0xBEEF);
        for (int i = 0; i < M_TILE * K_TILE; ++i) A[i] = __float2half((rand() & 1) ? 200.0f : -200.0f);
        for (int i = 0; i < N_TILE * K_TILE; ++i) B[i] = __float2half((rand() & 1) ? 80.0f  : -80.0f);
        printf("  [saturation-adjacent INFO] (intentionally over fp16 range; record only)\n");
        run_test("saturation-adjacent", A, B, 1e+9f, 1e+9f);  // sentinel: never fail
    }
    // 5) uniform_small: [-1, 1] like production inputs (rel ≤ 1e-3 ∨ abs ≤ 0.1)
    {
        half A[M_TILE * K_TILE], B[N_TILE * K_TILE];
        srand(1234);
        auto fr = []() { return ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f; };
        for (int i = 0; i < M_TILE * K_TILE; ++i) A[i] = __float2half(fr());
        for (int i = 0; i < N_TILE * K_TILE; ++i) B[i] = __float2half(fr());
        rc |= run_test("uniform_small", A, B, 1e-3f, 0.1f);
    }
    // 6) partial-m-zero: A row 4 all zero → D[4,*]=0; catches m-row confusion.
    {
        half A[M_TILE * K_TILE], B[N_TILE * K_TILE];
        srand(5678);
        auto fr = []() { return ((float) rand() / (float) RAND_MAX) * 2.0f - 1.0f; };
        for (int i = 0; i < M_TILE * K_TILE; ++i) A[i] = __float2half(fr());
        for (int k = 0; k < K_TILE; ++k) A[4 * K_TILE + k] = __float2half(0.0f);
        for (int i = 0; i < N_TILE * K_TILE; ++i) B[i] = __float2half(fr());
        rc |= run_test("partial-m4-zero", A, B, 1e-3f, 0.1f);
    }

    if (rc == 0) {
        printf("\n[PASS] FP16-acc lane mapping derived and verified across 6 adversarial cases.\n");
        printf("Mapping (32 lanes × 8 slots → (m,n)):\n");
        for (int lane = 0; lane < 32; ++lane) {
            printf("  lane %2d:", lane);
            for (int s = 0; s < 8; ++s)
                printf(" (%d,%d)", map_m[lane][s], map_n[lane][s]);
            printf("\n");
        }
    } else {
        printf("\n[FAIL] FP16-acc lane mapping verification failed (%d cases).\n", rc);
    }
    return rc;
}

// V11 Step 2a — SMEM → fragment round-trip unit test for the turbomind
// SmemCopy_MMA_884_A / _B lane mappings on V100 (sm_70).
//
// What this validates
// -------------------
// The lane → SMEM-offset formulas adapted from
//   research/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_sm70.h
//
//   A operand (SmemCopy_MMA_884_A, M=8):
//       m       = (lane / 16) * 4 + (lane % 4)
//       k_quad  = (lane & 12) >> 2
//       frag[8] = sA[m * K_A_STRIDE + k_quad * 8 .. + 7]
//
//   B operand (SmemCopy_MMA_884_B, M=32):
//       m       = (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4)
//       k_quad  = 0
//       frag[8] = sB[m * K_B_STRIDE + 0 .. + 7]
//
// The test fills SMEM with monotonically-increasing marker values, has every
// lane issue a 16-byte ld.shared (= 8 halves) at its derived offset, writes
// the fragment back to gmem, and compares against a CPU model that recomputes
// the same (m, k_quad) → offset mapping. Bit-exact match required.
//
// What this resolves
// ------------------
// Open question #1 from V11-STEP2-HANDOFF.md: what is the K-stride that makes
// the A formula valid for k_quad ∈ {0, 1, 2, 3}? Hypothesis: BK_A = 32 with
// 8-half steps per k_quad. The test fixes K_A_STRIDE = 32 (the smallest valid
// stride). If GPU and CPU match, the formula and stride are confirmed.
//
// What this does NOT validate
// ---------------------------
// - Whether these fragments are in the layout that mma.m8n8k4 actually
//   consumes (that's Step 2b).
// - The accumulator C lane→element mapping (open question #2; Step 2c).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

// ----------------------------------------------------------------------------
// Lane → (m, k_quad) formulas, shared between host and device.
// ----------------------------------------------------------------------------

__host__ __device__ static inline int A_m(int lane)      { return (lane / 16) * 4 + (lane % 4); }
__host__ __device__ static inline int A_kquad(int lane)  { return (lane & 12) >> 2; }
__host__ __device__ static inline int B_m(int lane)      { return (lane / 16) * 4 + (lane & 12) * 2 + (lane % 4); }
__host__ __device__ static inline int B_kquad(int lane)  { return 0; }

// SMEM tile dimensions chosen so the formula is fully exercised:
//   A: 8 rows × 32 cols (row-major in SMEM, stride 32). k_quad=3 reads cols 24-31.
//   B: 32 rows ×  8 cols (row-major in SMEM, stride 8).
constexpr int M_A = 8;
constexpr int K_A_STRIDE = 32;
constexpr int M_B = 32;
constexpr int K_B_STRIDE = 8;

constexpr int A_SMEM_HALVES = M_A * K_A_STRIDE;   // 256
constexpr int B_SMEM_HALVES = M_B * K_B_STRIDE;   // 256

// ----------------------------------------------------------------------------
// Device kernel: cooperative load gmem→SMEM, then per-lane fragment load.
// ----------------------------------------------------------------------------

__global__ void smem_to_frag_kernel(
        const half* __restrict__ A_init,
        const half* __restrict__ B_init,
        half* __restrict__ A_frags_out,   // [32 * 8]
        half* __restrict__ B_frags_out)   // [32 * 8]
{
    const int lane = threadIdx.x;

    __shared__ __align__(16) half sA[A_SMEM_HALVES];
    __shared__ __align__(16) half sB[B_SMEM_HALVES];

    // Cooperative gmem → SMEM (32 lanes × 8 halves = 256 halves per buffer).
    *reinterpret_cast<uint4*>(&sA[lane * 8]) = *reinterpret_cast<const uint4*>(&A_init[lane * 8]);
    *reinterpret_cast<uint4*>(&sB[lane * 8]) = *reinterpret_cast<const uint4*>(&B_init[lane * 8]);
    __syncthreads();

    // Per-lane fragment load using the turbomind formulas.
    half a_frag[8];
    half b_frag[8];
    {
        const int m  = A_m(lane);
        const int kq = A_kquad(lane);
        const half* src = &sA[m * K_A_STRIDE + kq * 8];
        *reinterpret_cast<uint4*>(a_frag) = *reinterpret_cast<const uint4*>(src);
    }
    {
        const int m  = B_m(lane);
        const int kq = B_kquad(lane);
        const half* src = &sB[m * K_B_STRIDE + kq * 8];
        *reinterpret_cast<uint4*>(b_frag) = *reinterpret_cast<const uint4*>(src);
    }

    // Write fragments back to gmem (one block of 8 halves per lane).
    *reinterpret_cast<uint4*>(&A_frags_out[lane * 8]) = *reinterpret_cast<const uint4*>(a_frag);
    *reinterpret_cast<uint4*>(&B_frags_out[lane * 8]) = *reinterpret_cast<const uint4*>(b_frag);
}

// ----------------------------------------------------------------------------
// Host CPU model: encode the same formula and predict each lane's 8 halves.
// ----------------------------------------------------------------------------

static void cpu_model_A(const half* sA, half* frags_out) {
    for (int lane = 0; lane < 32; ++lane) {
        const int m  = A_m(lane);
        const int kq = A_kquad(lane);
        const int off = m * K_A_STRIDE + kq * 8;
        for (int i = 0; i < 8; ++i) {
            frags_out[lane * 8 + i] = sA[off + i];
        }
    }
}

static void cpu_model_B(const half* sB, half* frags_out) {
    for (int lane = 0; lane < 32; ++lane) {
        const int m  = B_m(lane);
        const int kq = B_kquad(lane);
        const int off = m * K_B_STRIDE + kq * 8;
        for (int i = 0; i < 8; ++i) {
            frags_out[lane * 8 + i] = sB[off + i];
        }
    }
}

// ----------------------------------------------------------------------------
// Driver
// ----------------------------------------------------------------------------

static int compare_frags(const char* label, const half* gpu, const half* cpu, int n_halves) {
    int n_mismatch = 0;
    int first_bad = -1;
    for (int i = 0; i < n_halves; ++i) {
        const float g = __half2float(gpu[i]);
        const float c = __half2float(cpu[i]);
        if (g != c) {
            if (first_bad < 0) first_bad = i;
            ++n_mismatch;
        }
    }
    if (n_mismatch == 0) {
        printf("[ OK ] %s: %d/%d halves match\n", label, n_halves, n_halves);
        return 0;
    }
    printf("[FAIL] %s: %d/%d mismatches; first at frag idx %d (lane=%d, elem=%d): gpu=%.1f cpu=%.1f\n",
           label, n_mismatch, n_halves, first_bad, first_bad / 8, first_bad % 8,
           __half2float(gpu[first_bad]), __half2float(cpu[first_bad]));
    return 1;
}

static void print_lane_table_A() {
    printf("\nA-operand lane → (m, k_quad) table:\n");
    printf("  lane  m  k_quad  smem_off (stride=%d)\n", K_A_STRIDE);
    for (int lane = 0; lane < 32; ++lane) {
        const int m  = A_m(lane);
        const int kq = A_kquad(lane);
        printf("  %4d  %d  %6d  %d\n", lane, m, kq, m * K_A_STRIDE + kq * 8);
    }
}

static void print_lane_table_B() {
    printf("\nB-operand lane → m table (k_quad always 0):\n");
    printf("  lane  m  smem_off (stride=%d)\n", K_B_STRIDE);
    for (int lane = 0; lane < 32; ++lane) {
        const int m = B_m(lane);
        printf("  %4d  %2d  %d\n", lane, m, m * K_B_STRIDE);
    }
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s, cc=%d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 7) {
        printf("[SKIP] sm_70 required for SmemCopy_MMA_884 patterns\n");
        return 0;
    }

    const bool verbose = (argc > 1 && argv[1][0] == 'v');

    // ---- Build SMEM init data: sA[m][k] = m * K_A_STRIDE + k ----
    half* h_A_init = (half*) malloc(A_SMEM_HALVES * sizeof(half));
    half* h_B_init = (half*) malloc(B_SMEM_HALVES * sizeof(half));
    for (int i = 0; i < A_SMEM_HALVES; ++i) h_A_init[i] = __float2half((float) i);
    for (int i = 0; i < B_SMEM_HALVES; ++i) h_B_init[i] = __float2half((float) i);

    // ---- Allocate device buffers ----
    half *d_A_init, *d_B_init, *d_A_frags, *d_B_frags;
    cudaMalloc(&d_A_init,  A_SMEM_HALVES * sizeof(half));
    cudaMalloc(&d_B_init,  B_SMEM_HALVES * sizeof(half));
    cudaMalloc(&d_A_frags, 32 * 8 * sizeof(half));
    cudaMalloc(&d_B_frags, 32 * 8 * sizeof(half));
    cudaMemcpy(d_A_init, h_A_init, A_SMEM_HALVES * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_init, h_B_init, B_SMEM_HALVES * sizeof(half), cudaMemcpyHostToDevice);

    // ---- Launch ----
    smem_to_frag_kernel<<<1, 32>>>(d_A_init, d_B_init, d_A_frags, d_B_frags);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[FAIL] cuda error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    half* h_A_frags_gpu = (half*) malloc(32 * 8 * sizeof(half));
    half* h_B_frags_gpu = (half*) malloc(32 * 8 * sizeof(half));
    cudaMemcpy(h_A_frags_gpu, d_A_frags, 32 * 8 * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_frags_gpu, d_B_frags, 32 * 8 * sizeof(half), cudaMemcpyDeviceToHost);

    // ---- CPU model ----
    half* h_A_frags_cpu = (half*) malloc(32 * 8 * sizeof(half));
    half* h_B_frags_cpu = (half*) malloc(32 * 8 * sizeof(half));
    cpu_model_A(h_A_init, h_A_frags_cpu);
    cpu_model_B(h_B_init, h_B_frags_cpu);

    // ---- Compare ----
    int rc = 0;
    rc += compare_frags("A operand (8 × 32, k_quad ∈ {0..3})", h_A_frags_gpu, h_A_frags_cpu, 32 * 8);
    rc += compare_frags("B operand (32 × 8, k_quad = 0)",      h_B_frags_gpu, h_B_frags_cpu, 32 * 8);

    if (verbose) {
        print_lane_table_A();
        print_lane_table_B();
        printf("\nLane 0 A-frag (gpu):");
        for (int i = 0; i < 8; ++i) printf(" %.0f", __half2float(h_A_frags_gpu[i]));
        printf("\nLane 5 A-frag (gpu):");
        for (int i = 0; i < 8; ++i) printf(" %.0f", __half2float(h_A_frags_gpu[5*8 + i]));
        printf("\nLane 17 A-frag (gpu):");
        for (int i = 0; i < 8; ++i) printf(" %.0f", __half2float(h_A_frags_gpu[17*8 + i]));
        printf("\nLane 0 B-frag (gpu):");
        for (int i = 0; i < 8; ++i) printf(" %.0f", __half2float(h_B_frags_gpu[i]));
        printf("\nLane 7 B-frag (gpu):");
        for (int i = 0; i < 8; ++i) printf(" %.0f", __half2float(h_B_frags_gpu[7*8 + i]));
        printf("\n");
    }

    cudaFree(d_A_init); cudaFree(d_B_init);
    cudaFree(d_A_frags); cudaFree(d_B_frags);
    free(h_A_init); free(h_B_init);
    free(h_A_frags_gpu); free(h_B_frags_gpu);
    free(h_A_frags_cpu); free(h_B_frags_cpu);

    if (rc == 0) printf("\nALL PASS — SmemCopy_MMA_884 lane mappings match CPU model on V100.\n");
    else         printf("\n%d FAIL — lane mapping or SMEM stride hypothesis is wrong.\n", rc);
    return rc;
}

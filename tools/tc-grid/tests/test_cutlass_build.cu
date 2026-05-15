// SPRINT-018 P0: CUTLASS build verification on V100 (sm_70).
//
// Goal: prove that CUTLASS 2.11.0 headers resolve and a Gemm template
// instantiates for sm_70 with FP16 inputs + FP32 accumulator. We do NOT
// run a real GEMM here — that's P1's job. This test is the cheapest
// possible CUTLASS sanity check: if it compiles + links, we have a viable
// platform for SPRINT-018.
//
// What it does at runtime:
//   - Allocate tiny (M=N=K=128) FP16 A/B and FP32 C buffers.
//   - Instantiate the Gemm op (which forces template body compilation).
//   - Initialize args + call op() to execute one tiny GEMM.
//   - Verify it runs without error.
//
// If this passes -> CUTLASS works in our env -> proceed to SPRINT-018 P1.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s, cc=%d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 7) {
        printf("[SKIP] sm_70 required for this CUTLASS Gemm template\n");
        return 0;
    }

    // FP16-in, FP32-acc, FP32-out GEMM at the smallest tile that's
    // compatible with sm_70 INT8/FP16 TensorOps. This is intentionally
    // minimal — we just want template instantiation to succeed.
    using ElementInputA  = cutlass::half_t;
    using ElementInputB  = cutlass::half_t;
    using ElementOutput  = float;
    using ElementAcc     = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAcc,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm70,
        cutlass::gemm::GemmShape<128, 128, 32>,  // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape
        cutlass::gemm::GemmShape<8, 8, 4>        // InstructionShape (sm_70 m8n8k4)
    >;

    constexpr int M = 128, N = 128, K = 128;

    cutlass::HostTensor<ElementInputA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementInputB, LayoutB> B({K, N});
    cutlass::HostTensor<ElementOutput, LayoutC> C({M, N});

    // Trivial values: A all 1.0, B all 1.0 -> C should be all K (=128).
    for (int i = 0; i < M * K; ++i) A.host_data()[i] = cutlass::half_t(1.0f);
    for (int i = 0; i < K * N; ++i) B.host_data()[i] = cutlass::half_t(1.0f);
    for (int i = 0; i < M * N; ++i) C.host_data()[i] = 0.0f;
    A.sync_device();
    B.sync_device();
    C.sync_device();

    Gemm gemm_op;
    typename Gemm::Arguments args(
        {M, N, K},
        A.device_ref(),
        B.device_ref(),
        C.device_ref(),
        C.device_ref(),
        {1.0f, 0.0f}
    );

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        printf("[FAIL] CUTLASS Gemm returned status=%d\n", (int) status);
        return 1;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[FAIL] cuda error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    C.sync_host();
    float expected = (float) K;  // 128
    int n_ok = 0, n_bad = 0;
    float minv = C.host_data()[0], maxv = C.host_data()[0];
    for (int i = 0; i < M * N; ++i) {
        float v = C.host_data()[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        if (v == expected) ++n_ok;
        else ++n_bad;
    }
    printf("[%s] C range [%.1f, %.1f], expected %.1f, %d/%d match\n",
           (n_bad == 0) ? " OK " : "FAIL",
           minv, maxv, expected, n_ok, M * N);

    if (n_bad == 0) {
        printf("\nALL PASS - CUTLASS 2.11.0 Gemm sm_70 FP16-in/FP32-out works.\n");
        printf("Proceed to SPRINT-018 P1 (CUTLASS as candidate kernel).\n");
        return 0;
    }
    return 1;
}

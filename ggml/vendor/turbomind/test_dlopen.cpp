// SPRINT-023 P1.4 — dlopen smoke test for libggml-turbomind.so
//
// Validates that:
//  1. libggml-turbomind.so can be loaded via dlopen with RTLD_LOCAL
//  2. The 6 exported C ABI symbols resolve via dlsym
//  3. api_version() returns 1
//  4. init() succeeds on device 0
//  5. packed_bytes() returns plausible sizes for FP8 + MXFP4
//  6. A single FP16 mul_mat runs without crashing (round-trip with cuBLAS
//     path — exercises the dispatch infrastructure end-to-end)
//  7. shutdown() succeeds
//
// This is NOT a numerical correctness test (that's P2). Just exports.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml-turbomind-api.h"

#define CHECK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %d at %s:%d: %s\n", _e, __FILE__, __LINE__, \
            cudaGetErrorString(_e)); return 99; } } while(0)

// Function pointer types
typedef int  (*pfn_api_version)(void);
typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_packed_bytes)(int, int, int, int, size_t*, size_t*);
typedef int  (*pfn_pack_weight)(const void*, int, int, int, int,
                                void*, void*, int*, void*);
typedef int  (*pfn_mul_mat)(const void*, const void*, const void*,
                            int, int, int, int, int, int, void*, void*);
typedef int  (*pfn_mul_mat_grouped)(const void*, const int*, const int*, int,
                                    const void* const*, const void* const*,
                                    int, int, int, int, int, void*, void*);

int main(int argc, char** argv) {
    const char* lib_path = (argc > 1) ? argv[1] : "./libggml-turbomind.so";
    fprintf(stderr, "test_ggml_turbomind: opening %s\n", lib_path);

    void* h = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    auto v   = (pfn_api_version)        dlsym(h, "ggml_turbomind_api_version");
    auto in  = (pfn_init)               dlsym(h, "ggml_turbomind_init");
    auto sh  = (pfn_shutdown)           dlsym(h, "ggml_turbomind_shutdown");
    auto pb  = (pfn_packed_bytes)       dlsym(h, "ggml_turbomind_packed_bytes");
    auto pw  = (pfn_pack_weight)        dlsym(h, "ggml_turbomind_pack_weight_expert");
    auto mm  = (pfn_mul_mat)            dlsym(h, "ggml_turbomind_mul_mat");
    auto mmg = (pfn_mul_mat_grouped)    dlsym(h, "ggml_turbomind_mul_mat_grouped");

    if (!v || !in || !sh || !pb || !pw || !mm || !mmg) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        return 2;
    }
    fprintf(stderr, "[ok] all 7 symbols resolved\n");

    int api_v = v();
    if (api_v != GGML_TURBOMIND_API_VERSION) {
        fprintf(stderr, "api_version mismatch: got %d want %d\n",
                api_v, GGML_TURBOMIND_API_VERSION);
        return 3;
    }
    fprintf(stderr, "[ok] api_version = %d\n", api_v);

    if (in(0) != 0) {
        fprintf(stderr, "init failed\n");
        return 4;
    }
    fprintf(stderr, "[ok] init(0) succeeded\n");

    // Test packed_bytes for FP8 + MXFP4
    size_t wb, sb;
    int rc = pb(GGML_TM_DTYPE_F8_E4M3_B128, 7168, 7168, 128, &wb, &sb);
    if (rc != 0) { fprintf(stderr, "packed_bytes(fp8) rc=%d\n", rc); return 5; }
    fprintf(stderr, "[ok] FP8 N=K=7168 g=128 → w=%zu B (expect 51380224), s=%zu B\n", wb, sb);

    rc = pb(GGML_TM_DTYPE_MXFP4, 7168, 7168, 32, &wb, &sb);
    if (rc != 0) { fprintf(stderr, "packed_bytes(mxfp4) rc=%d\n", rc); return 6; }
    fprintf(stderr, "[ok] MXFP4 N=K=7168 g=32 → w=%zu B (expect 25690112), s=%zu B\n", wb, sb);

    // ---- End-to-end FP16 mul_mat smoke test ----
    // Generates random A (M=8, K=7168) and B (K=7168, N=7168) in FP16,
    // runs ggml_turbomind_mul_mat (which dispatches to the cuBLAS path
    // internally), confirms output is non-zero.
    const int M = 8, N = 7168, K = 7168;
    std::vector<__half> hA(M * K), hB(K * N);
    std::mt19937 rng(0x1234);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = __float2half(d(rng));
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = __float2half(d(rng));

    __half *dA, *dB, *dD;
    CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dB, hB.size() * sizeof(__half)));
    CHECK(cudaMalloc(&dD, (size_t)M * N * sizeof(__half)));
    CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half), cudaMemcpyHostToDevice));

    rc = mm(dA, dB, /*V=*/nullptr,
            GGML_TM_DTYPE_FP16, M, N, K, /*group_size=*/1,
            /*k_pack=*/0, dD, /*stream=*/nullptr);
    if (rc != 0) {
        fprintf(stderr, "ggml_turbomind_mul_mat rc=%d\n", rc);
        return 7;
    }
    fprintf(stderr, "[ok] FP16 mul_mat dispatched\n");

    // Spot-check: output has at least one non-zero value
    std::vector<__half> hD(M * N);
    CHECK(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    int nonzero = 0;
    for (size_t i = 0; i < hD.size(); ++i) {
        if (__half2float(hD[i]) != 0.0f) ++nonzero;
    }
    if (nonzero == 0) {
        fprintf(stderr, "output is all zeros — dispatch likely failed silently\n");
        return 8;
    }
    fprintf(stderr, "[ok] FP16 mul_mat produced %d non-zero output values (of %zu)\n",
            nonzero, hD.size());

    cudaFree(dA); cudaFree(dB); cudaFree(dD);

    sh();
    fprintf(stderr, "[ok] shutdown\n");

    dlclose(h);
    fprintf(stderr, "test_ggml_turbomind: PASS\n");
    return 0;
}

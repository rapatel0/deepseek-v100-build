// SPRINT-025 P2 — Multi-device CUDA_TURBOMIND smoke.
//
// Verifies that ggml_turbomind_init(0) and ggml_turbomind_init(1) can both
// be active in the same process, that workspace allocations on GPU 0 are
// not freed by a later ggml_turbomind_init(1), and that subsequent dispatch
// on GPU 0 still produces correct output.
//
// Pre-SPRINT-025 the global g_state in api.cc was a single device's
// workspace; init(1) after init(0) would call cudaFree on GPU 0's pointers,
// breaking any in-flight dispatch. P2 refactors to per-device State entries.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <cstdint>

#include "ggml-turbomind-api.h"

#define CHECK(x) do { auto _e=(x); if (_e!=cudaSuccess) { \
    fprintf(stderr,"CUDA err %d at %s:%d: %s\n",_e,__FILE__,__LINE__,\
            cudaGetErrorString(_e)); std::exit(1); } } while(0)

typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_packed_bytes)(int, int, int, int, size_t*, size_t*);
typedef int  (*pfn_pack_weight)(const void*, int, int, int, int,
                                void*, void*, int*, void*);
typedef int  (*pfn_mul_mat)(const void*, const void*, const void*,
                            int, int, int, int, int, int, void*, void*);

struct block_f8_e4m3_b128 { uint8_t e; uint8_t qs[128]; };

static void make_fixture(std::vector<block_f8_e4m3_b128>& blocks, int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(123, 130);
    auto rand_byte = [&]() {
        for (;;) { int b = rng() & 0xFF; if (b != 0x7F && b != 0xFF) return (uint8_t)b; }
    };
    const int blocks_per_row = K / 128;
    blocks.resize((size_t)N * blocks_per_row);
    for (auto& b : blocks) {
        b.e = (uint8_t)e_dist(rng);
        for (int j = 0; j < 128; ++j) b.qs[j] = rand_byte();
    }
}

// Pack a small weight, then run mul_mat. Returns sum of |D| as a sanity number.
static double pack_and_dispatch(
    pfn_pack_weight pw, pfn_mul_mat mm, pfn_packed_bytes pb,
    int N, int K, int group_size, uint32_t seed)
{
    std::vector<block_f8_e4m3_b128> blocks;
    make_fixture(blocks, N, K, seed);

    void * d_src = nullptr;
    CHECK(cudaMalloc(&d_src, blocks.size() * sizeof(block_f8_e4m3_b128)));
    CHECK(cudaMemcpy(d_src, blocks.data(), blocks.size() * sizeof(block_f8_e4m3_b128), cudaMemcpyHostToDevice));

    size_t wb = 0, sb = 0;
    if (pb(GGML_TM_DTYPE_F8_E4M3_B128, N, K, group_size, &wb, &sb) != 0) {
        fprintf(stderr, "packed_bytes failed\n"); return -1.0;
    }
    void *d_w=nullptr, *d_s=nullptr;
    CHECK(cudaMalloc(&d_w, wb));
    if (sb) CHECK(cudaMalloc(&d_s, sb));

    int k_pack = 0;
    if (pw(d_src, GGML_TM_DTYPE_F8_E4M3_B128, N, K, group_size, d_w, d_s, &k_pack, nullptr) != 0) {
        fprintf(stderr, "pack failed\n"); return -1.0;
    }

    const int M = 8;
    std::vector<__half> hA((size_t)M * K);
    std::mt19937 rng(seed ^ 0xCAFE);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    for (auto& v : hA) v = __float2half(ad(rng));

    __half * dA = nullptr;
    CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
    CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));
    __half * dD = nullptr;
    CHECK(cudaMalloc(&dD, (size_t) M * N * sizeof(__half)));

    int rc = mm(dA, d_w, d_s, GGML_TM_DTYPE_F8_E4M3_B128, M, N, K, group_size, k_pack, dD, nullptr);
    if (rc != 0) { fprintf(stderr, "mul_mat rc=%d\n", rc); return -1.0; }
    CHECK(cudaDeviceSynchronize());

    std::vector<__half> hD((size_t)M * N);
    CHECK(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    double sum_abs = 0;
    int nan_count = 0;
    for (auto v : hD) {
        float f = __half2float(v);
        if (std::isnan(f) || std::isinf(f)) nan_count++;
        else sum_abs += std::fabs(f);
    }
    cudaFree(dA); cudaFree(dD);
    cudaFree(d_w); if (d_s) cudaFree(d_s);
    cudaFree(d_src);
    if (nan_count > 0) {
        fprintf(stderr, "NaN/Inf count=%d\n", nan_count);
        return -1.0;
    }
    return sum_abs;
}

int main(int argc, char** argv) {
    const char* lib_path = (argc > 1) ? argv[1] : "./libggml-turbomind.so";
    void* h = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto in  = (pfn_init)         dlsym(h, "ggml_turbomind_init");
    auto sh  = (pfn_shutdown)     dlsym(h, "ggml_turbomind_shutdown");
    auto pb  = (pfn_packed_bytes) dlsym(h, "ggml_turbomind_packed_bytes");
    auto pw  = (pfn_pack_weight)  dlsym(h, "ggml_turbomind_pack_weight_expert");
    auto mm  = (pfn_mul_mat)      dlsym(h, "ggml_turbomind_mul_mat");
    if (!in || !sh || !pb || !pw || !mm) { fprintf(stderr, "dlsym\n"); return 1; }

    int dev_count = 0;
    CHECK(cudaGetDeviceCount(&dev_count));
    fprintf(stderr, "[multi-dev] %d CUDA devices visible\n", dev_count);
    if (dev_count < 2) {
        fprintf(stderr, "[multi-dev] SKIP: need >= 2 GPUs (have %d)\n", dev_count);
        return 0;
    }

    // Init both GPUs
    CHECK(cudaSetDevice(0));
    if (in(0) != 0) { fprintf(stderr, "init(0) failed\n"); return 2; }
    fprintf(stderr, "[multi-dev] init(0) ok\n");
    CHECK(cudaSetDevice(1));
    if (in(1) != 0) { fprintf(stderr, "init(1) failed\n"); return 2; }
    fprintf(stderr, "[multi-dev] init(1) ok\n");

    // Pack + dispatch on GPU 0
    CHECK(cudaSetDevice(0));
    double s0_a = pack_and_dispatch(pw, mm, pb, 256, 256, 128, 0xC0FFEE);
    fprintf(stderr, "[multi-dev] GPU 0 first dispatch: sum_abs=%.2f\n", s0_a);
    if (s0_a <= 0) return 3;

    // Pack + dispatch on GPU 1 (this used to teardown GPU 0's state)
    CHECK(cudaSetDevice(1));
    double s1 = pack_and_dispatch(pw, mm, pb, 256, 256, 128, 0xBEEFCAFE);
    fprintf(stderr, "[multi-dev] GPU 1 dispatch: sum_abs=%.2f\n", s1);
    if (s1 <= 0) return 4;

    // Second dispatch on GPU 0 — pre-P2 this would crash/corrupt because
    // init(1) freed GPU 0's d_barriers/d_partials/d_flags. Post-P2 it
    // must reuse the GPU 0 State entry that was never torn down.
    CHECK(cudaSetDevice(0));
    double s0_b = pack_and_dispatch(pw, mm, pb, 256, 256, 128, 0xC0FFEE);
    fprintf(stderr, "[multi-dev] GPU 0 second dispatch: sum_abs=%.2f\n", s0_b);
    if (s0_b <= 0) return 5;

    // Must match first dispatch (same input, same RNG seed).
    const double rel = std::fabs(s0_a - s0_b) / std::max(1e-6, s0_a);
    fprintf(stderr, "[multi-dev] GPU 0 first vs second: rel diff = %.6e\n", rel);
    if (rel > 1e-3) {
        fprintf(stderr, "[multi-dev] FAIL — second dispatch on GPU 0 doesn't match first\n");
        return 6;
    }

    sh();
    dlclose(h);
    fprintf(stderr, "[multi-dev] PASS — per-device state intact across cross-device init/dispatch\n");
    return 0;
}

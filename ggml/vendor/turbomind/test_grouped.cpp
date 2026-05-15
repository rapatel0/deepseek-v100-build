// SPRINT-024 P0.3 + P0.4 — Grouped MoE C ABI smoke + sync-memcpy microbench.
//
// Calls ggml_turbomind_mul_mat_grouped with N experts sharing the same
// packed weight (so the math degenerates to the single-expert case for any
// routing). Verifies:
//   (P0.3) num_experts > 1 produces finite, non-NaN output and rc == 0 on
//          FP8 and MXFP4 on sm70.
//   (P0.4) Timing 1000 back-to-back grouped calls quantifies the cost of
//          the synchronous cudaMemcpy at api.cc:607 (which reads
//          expert_offsets[num_experts] from device memory on every call).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>

#include "ggml-turbomind-api.h"

#define CHECK(x) do { auto _e=(x); if (_e!=cudaSuccess) { \
    fprintf(stderr,"CUDA err %d at %s:%d: %s\n",_e,__FILE__,__LINE__,\
            cudaGetErrorString(_e)); std::exit(1); } } while(0)

// ---- function pointer types ------------------------------------------------
typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_packed_bytes)(int, int, int, int, size_t*, size_t*);
typedef int  (*pfn_pack_weight)(const void*, int, int, int, int,
                                void*, void*, int*, void*);
typedef int  (*pfn_mul_mat_grouped)(const void*, const int*, const int*, int,
                                    const void* const*, const void* const*,
                                    int, int, int, int, int, void*, void*);

// ---- StridedPtr mirror — must match research/lmdeploy/.../matrix_ptr.h:9 ----
struct alignas(16) StridedPtrH {
    void* p;
    int   stride;
    // alignas(16) implicitly pads to 16 bytes total
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtr must be 16 bytes");

// ---- GGML block layouts (same as test_correctness.cpp) ---------------------
struct block_f8_e4m3_b128 { uint8_t e; uint8_t qs[128]; };
struct block_mxfp4        { uint8_t e; uint8_t qs[16];  };

static void make_fp8_fixture(std::vector<block_f8_e4m3_b128>& blocks, int N, int K, uint32_t seed) {
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

static void make_mxfp4_fixture(std::vector<block_mxfp4>& blocks, int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(123, 130);
    std::uniform_int_distribution<int> q_dist(0, 255);
    const int blocks_per_row = K / 32;
    blocks.resize((size_t)N * blocks_per_row);
    for (auto& b : blocks) {
        b.e = (uint8_t)e_dist(rng);
        for (int j = 0; j < 16; ++j) b.qs[j] = (uint8_t)q_dist(rng);
    }
}

// ---- One grouped-call smoke + timing pass --------------------------------
template <typename BlockT>
static int run_grouped_smoke(
    void* lib,
    int ggml_type,
    int n_experts,
    int tokens_per_expert,
    int N, int K, int group_size,
    const std::vector<BlockT>& blocks,
    const char* tag)
{
    auto in  = (pfn_init)            dlsym(lib, "ggml_turbomind_init");
    auto sh  = (pfn_shutdown)        dlsym(lib, "ggml_turbomind_shutdown");
    auto pb  = (pfn_packed_bytes)    dlsym(lib, "ggml_turbomind_packed_bytes");
    auto pw  = (pfn_pack_weight)     dlsym(lib, "ggml_turbomind_pack_weight_expert");
    auto mmg = (pfn_mul_mat_grouped) dlsym(lib, "ggml_turbomind_mul_mat_grouped");
    if (!in || !sh || !pb || !pw || !mmg) {
        fprintf(stderr, "[%s] dlsym failed (mmg=%p)\n", tag, (void*)mmg);
        return 1;
    }
    if (in(0) != 0) { fprintf(stderr, "[%s] init failed\n", tag); return 2; }

    // ---- pack one expert ----
    const size_t src_bytes = blocks.size() * sizeof(BlockT);
    void* d_src = nullptr;
    CHECK(cudaMalloc(&d_src, src_bytes));
    CHECK(cudaMemcpy(d_src, blocks.data(), src_bytes, cudaMemcpyHostToDevice));

    size_t wb = 0, sb = 0;
    if (pb(ggml_type, N, K, group_size, &wb, &sb) != 0) { fprintf(stderr,"[%s] pb fail\n",tag); return 3; }
    void *d_w = nullptr, *d_s = nullptr;
    CHECK(cudaMalloc(&d_w, wb));
    if (sb) CHECK(cudaMalloc(&d_s, sb));

    int k_pack = 0;
    if (pw(d_src, ggml_type, N, K, group_size, d_w, d_s, &k_pack, nullptr) != 0) {
        fprintf(stderr,"[%s] pack fail\n",tag); return 4;
    }
    cudaFree(d_src);

    // ---- StridedPtr arrays on device, all pointing to the same packed buffer
    // Per turbomind_packed_b_ld_factor.md and SPRINT-024 §3.2:
    //   packed_b_ld = K * 32 for HMMA_884 OPERAND_B Pack_M=1 on sm70
    //   packed_v_ld = N      (post-swap Vdesc.ld)
    const int packed_b_ld = K * 32;
    const int packed_v_ld = N;

    std::vector<StridedPtrH> h_w(n_experts), h_s(n_experts);
    for (int e = 0; e < n_experts; ++e) {
        h_w[e] = StridedPtrH{ d_w, packed_b_ld };
        h_s[e] = StridedPtrH{ d_s, packed_v_ld };
    }
    void *d_w_arr = nullptr, *d_s_arr = nullptr;
    CHECK(cudaMalloc(&d_w_arr, sizeof(StridedPtrH) * n_experts));
    CHECK(cudaMemcpy(d_w_arr, h_w.data(), sizeof(StridedPtrH) * n_experts, cudaMemcpyHostToDevice));
    if (sb) {
        CHECK(cudaMalloc(&d_s_arr, sizeof(StridedPtrH) * n_experts));
        CHECK(cudaMemcpy(d_s_arr, h_s.data(), sizeof(StridedPtrH) * n_experts, cudaMemcpyHostToDevice));
    }

    // ---- routing metadata: tokens_per_expert routes per expert ----
    const int total_tokens = n_experts * tokens_per_expert;
    std::vector<int> h_offsets(n_experts + 1);
    for (int e = 0; e <= n_experts; ++e) h_offsets[e] = e * tokens_per_expert;
    int* d_offsets = nullptr;
    CHECK(cudaMalloc(&d_offsets, sizeof(int) * (n_experts + 1)));
    CHECK(cudaMemcpy(d_offsets, h_offsets.data(), sizeof(int) * (n_experts + 1), cudaMemcpyHostToDevice));

    // ---- A activations + D output ----
    std::mt19937 rng(0xDEADBEEF);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    std::vector<__half> hA((size_t)total_tokens * K);
    for (auto& v : hA) v = __float2half(ad(rng));
    __half* dA = nullptr;
    CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
    CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));
    __half* dD = nullptr;
    CHECK(cudaMalloc(&dD, (size_t)total_tokens * N * sizeof(__half)));

    // ---- P0.3: functional smoke ----
    int rc = mmg(dA, /*token_indices=*/nullptr, d_offsets, n_experts,
                 (const void* const*)d_w_arr, (const void* const*)d_s_arr,
                 ggml_type, N, K, group_size, k_pack, dD, nullptr);
    if (rc != 0) {
        fprintf(stderr, "[%s] FAIL — mul_mat_grouped rc=%d on n_experts=%d\n", tag, rc, n_experts);
        return 5;
    }
    CHECK(cudaDeviceSynchronize());

    // Verify D has no NaN/Inf and is not all-zero.
    std::vector<__half> hD((size_t)total_tokens * N);
    CHECK(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    int n_nan = 0, n_inf = 0, n_finite_nz = 0;
    float sum_abs = 0;
    for (auto v : hD) {
        float f = __half2float(v);
        if (std::isnan(f)) n_nan++;
        else if (std::isinf(f)) n_inf++;
        else if (f != 0.0f) { n_finite_nz++; sum_abs += std::fabs(f); }
    }
    fprintf(stderr, "[%s n_experts=%d] D: nan=%d inf=%d finite_nz=%d sum_abs=%.2f\n",
            tag, n_experts, n_nan, n_inf, n_finite_nz, sum_abs);
    if (n_nan > 0 || n_inf > 0 || n_finite_nz == 0) {
        fprintf(stderr, "[%s n_experts=%d] FAIL — pathological output\n", tag, n_experts);
        return 6;
    }

    // ---- P0.4: time 1000 back-to-back calls. The synchronous cudaMemcpy at
    // api.cc:607 inside each grouped call dominates if launch overhead is
    // amortized.
    const int N_TIMED = 1000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N_TIMED; ++i) {
        rc = mmg(dA, nullptr, d_offsets, n_experts,
                 (const void* const*)d_w_arr, (const void* const*)d_s_arr,
                 ggml_type, N, K, group_size, k_pack, dD, nullptr);
        if (rc != 0) { fprintf(stderr,"[%s] grouped rc=%d at iter %d\n", tag, rc, i); break; }
    }
    CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_call_us = total_us / N_TIMED;
    fprintf(stderr, "[%s n_experts=%d] P0.4: %d calls in %.1f us → %.2f us/call (incl sync memcpy)\n",
            tag, n_experts, N_TIMED, total_us, per_call_us);

    cudaFree(d_offsets);
    if (d_s_arr) cudaFree(d_s_arr);
    cudaFree(d_w_arr);
    cudaFree(dA); cudaFree(dD);
    if (d_s) cudaFree(d_s);
    cudaFree(d_w);
    sh();

    fprintf(stderr, "[%s n_experts=%d] PASS\n", tag, n_experts);
    return 0;
}

int main(int argc, char** argv) {
    const char* lib_path = (argc > 1) ? argv[1] : "./libggml-turbomind.so";
    void* h = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

    // Per SPRINT-024 P0.3: N in {2, 6, 8}. tokens_per_expert = 8 keeps M small.
    const int N = 256, K = 256;
    const int tokens_per_expert = 8;
    const std::vector<int> n_experts_list = {2, 6, 8};

    int fails = 0;
    for (int ne : n_experts_list) {
        // FP8
        std::vector<block_f8_e4m3_b128> f8;
        make_fp8_fixture(f8, N, K, 0xC0FFEE);
        if (run_grouped_smoke<block_f8_e4m3_b128>(
                h, GGML_TM_DTYPE_F8_E4M3_B128, ne, tokens_per_expert,
                N, K, 128, f8, "F8_E4M3_B128") != 0) {
            fails++;
        }

        // MXFP4
        std::vector<block_mxfp4> fp4;
        make_mxfp4_fixture(fp4, N, K, 0xBEEFCAFE);
        if (run_grouped_smoke<block_mxfp4>(
                h, GGML_TM_DTYPE_MXFP4, ne, tokens_per_expert,
                N, K, 32, fp4, "MXFP4") != 0) {
            fails++;
        }
    }

    dlclose(h);
    return fails ? 1 : 0;
}

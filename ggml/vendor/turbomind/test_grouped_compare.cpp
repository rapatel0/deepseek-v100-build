// Compare grouped MXFP4 dispatch against independent single-expert dispatches
// at DSv4 MoE expert shapes. This catches grouped/ragged alignment mistakes
// without depending on a host-side MXFP4 reference for huge matrices.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ggml-turbomind-api.h"

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
} } while (0)

typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_packed_bytes)(int, int, int, int, size_t *, size_t *);
typedef int  (*pfn_pack_weight)(const void *, int, int, int, int, void *, void *, int *, void *);
typedef int  (*pfn_mul_mat)(const void *, const void *, const void *, int, int, int, int, int, int, void *, void *);
typedef int  (*pfn_mul_mat_grouped)(const void *, const int *, const int *, int,
                                    const void * const *, const void * const *,
                                    int, int, int, int, int, void *, void *);

struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];
};

struct alignas(16) StridedPtrH {
    void * p;
    int    stride;
};
static_assert(sizeof(StridedPtrH) == 16, "StridedPtrH must match turbomind StridedPtr");

struct Case {
    int N;
    int K;
    int tokens_per_active;
    const char * name;
};

static void make_mxfp4_fixture(std::vector<block_mxfp4> & blocks, int N, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(123, 130);
    std::uniform_int_distribution<int> q_dist(0, 255);
    blocks.resize((size_t) N * (K / 32));
    for (block_mxfp4 & b : blocks) {
        b.e = (uint8_t) e_dist(rng);
        for (uint8_t & q : b.qs) {
            q = (uint8_t) q_dist(rng);
        }
    }
}

static int run_case(void * lib, const Case & c) {
    auto in  = (pfn_init)             dlsym(lib, "ggml_turbomind_init");
    auto sh  = (pfn_shutdown)         dlsym(lib, "ggml_turbomind_shutdown");
    auto pb  = (pfn_packed_bytes)     dlsym(lib, "ggml_turbomind_packed_bytes");
    auto pw  = (pfn_pack_weight)      dlsym(lib, "ggml_turbomind_pack_weight_expert");
    auto mm  = (pfn_mul_mat)          dlsym(lib, "ggml_turbomind_mul_mat");
    auto mmg = (pfn_mul_mat_grouped)  dlsym(lib, "ggml_turbomind_mul_mat_grouped");
    if (!in || !sh || !pb || !pw || !mm || !mmg) {
        fprintf(stderr, "[%s] dlsym failed\n", c.name);
        return 1;
    }
    if (in(0) != 0) {
        fprintf(stderr, "[%s] init failed\n", c.name);
        return 2;
    }

    constexpr int ggml_type = GGML_TM_DTYPE_MXFP4;
    constexpr int group_size = 32;
    constexpr int num_experts = 256;
    const std::vector<int> active = {0, 17, 42, 87, 173, 255};
    const int total_tokens = (int) active.size() * c.tokens_per_active;

    size_t wb = 0, sb = 0;
    int rc = pb(ggml_type, c.N, c.K, group_size, &wb, &sb);
    if (rc != 0) {
        fprintf(stderr, "[%s] packed_bytes rc=%d\n", c.name, rc);
        return 3;
    }

    std::vector<void *> d_w_active(active.size(), nullptr);
    std::vector<void *> d_s_active(active.size(), nullptr);
    int k_pack = 0;

    for (size_t i = 0; i < active.size(); ++i) {
        std::vector<block_mxfp4> blocks;
        make_mxfp4_fixture(blocks, c.N, c.K, 0xC0010000u + (uint32_t)i * 97u + (uint32_t)c.N);

        void * d_src = nullptr;
        CHECK_CUDA(cudaMalloc(&d_src, blocks.size() * sizeof(block_mxfp4)));
        CHECK_CUDA(cudaMemcpy(d_src, blocks.data(), blocks.size() * sizeof(block_mxfp4), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc(&d_w_active[i], wb));
        CHECK_CUDA(cudaMalloc(&d_s_active[i], sb));

        int this_pack = 0;
        rc = pw(d_src, ggml_type, c.N, c.K, group_size, d_w_active[i], d_s_active[i], &this_pack, nullptr);
        CHECK_CUDA(cudaFree(d_src));
        if (rc != 0) {
            fprintf(stderr, "[%s] pack active[%zu]=%d rc=%d\n", c.name, i, active[i], rc);
            return 4;
        }
        if (i == 0) {
            k_pack = this_pack;
        } else if (this_pack != k_pack) {
            fprintf(stderr, "[%s] inconsistent k_pack 0x%x vs 0x%x\n", c.name, this_pack, k_pack);
            return 5;
        }
    }

    std::vector<StridedPtrH> h_w(num_experts);
    std::vector<StridedPtrH> h_s(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        h_w[e] = StridedPtrH{d_w_active[0], c.K * 32};
        h_s[e] = StridedPtrH{d_s_active[0], c.N};
    }
    for (size_t i = 0; i < active.size(); ++i) {
        h_w[active[i]] = StridedPtrH{d_w_active[i], c.K * 32};
        h_s[active[i]] = StridedPtrH{d_s_active[i], c.N};
    }

    void * d_w_table = nullptr;
    void * d_s_table = nullptr;
    CHECK_CUDA(cudaMalloc(&d_w_table, h_w.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMalloc(&d_s_table, h_s.size() * sizeof(StridedPtrH)));
    CHECK_CUDA(cudaMemcpy(d_w_table, h_w.data(), h_w.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_s_table, h_s.data(), h_s.size() * sizeof(StridedPtrH), cudaMemcpyHostToDevice));

    std::vector<int> h_offsets(num_experts + 1, 0);
    int running = 0;
    size_t active_pos = 0;
    for (int e = 0; e < num_experts; ++e) {
        h_offsets[e] = running;
        if (active_pos < active.size() && e == active[active_pos]) {
            running += c.tokens_per_active;
            active_pos++;
        }
    }
    h_offsets[num_experts] = running;
    if (running != total_tokens) {
        fprintf(stderr, "[%s] bad offsets total %d expected %d\n", c.name, running, total_tokens);
        return 6;
    }

    int * d_offsets = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

    std::mt19937 rng(0x5EED1234u + (uint32_t)c.K);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    std::vector<__half> h_A((size_t) total_tokens * c.K);
    for (__half & v : h_A) {
        v = __float2half(ad(rng));
    }

    __half * d_A = nullptr;
    __half * d_D_grouped = nullptr;
    __half * d_D_single = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_D_grouped, (size_t) total_tokens * c.N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_D_single,  (size_t) total_tokens * c.N * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_D_grouped, 0, (size_t) total_tokens * c.N * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_D_single,  0, (size_t) total_tokens * c.N * sizeof(__half)));

    rc = mmg(d_A, nullptr, d_offsets, num_experts,
             (const void * const *) d_w_table, (const void * const *) d_s_table,
             ggml_type, c.N, c.K, group_size, k_pack, d_D_grouped, nullptr);
    if (rc != 0) {
        fprintf(stderr, "[%s] grouped rc=%d\n", c.name, rc);
        return 7;
    }

    for (size_t i = 0; i < active.size(); ++i) {
        const int offset = h_offsets[active[i]];
        rc = mm(d_A + (size_t) offset * c.K,
                d_w_active[i],
                d_s_active[i],
                ggml_type, c.tokens_per_active, c.N, c.K, group_size, k_pack,
                d_D_single + (size_t) offset * c.N,
                nullptr);
        if (rc != 0) {
            fprintf(stderr, "[%s] single active[%zu]=%d rc=%d\n", c.name, i, active[i], rc);
            return 8;
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<__half> h_grouped((size_t) total_tokens * c.N);
    std::vector<__half> h_single ((size_t) total_tokens * c.N);
    CHECK_CUDA(cudaMemcpy(h_grouped.data(), d_D_grouped, h_grouped.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_single.data(),  d_D_single,  h_single.size()  * sizeof(__half), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float sum_ref = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < h_grouped.size(); ++i) {
        const float g = __half2float(h_grouped[i]);
        const float s = __half2float(h_single[i]);
        const float d = fabsf(g - s);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_ref += fabsf(s);
        if (!std::isfinite(g) || d > 0.25f) {
            bad++;
        }
    }
    const float rel = sum_ref > 0 ? sum_abs / sum_ref : 0.0f;
    fprintf(stderr, "[%s] N=%d K=%d active=%zu tpa=%d k_pack=0x%x max_abs=%.4e rel=%.4e bad=%d/%zu\n",
            c.name, c.N, c.K, active.size(), c.tokens_per_active, k_pack, max_abs, rel, bad, h_grouped.size());

    for (void * p : d_w_active) CHECK_CUDA(cudaFree(p));
    for (void * p : d_s_active) CHECK_CUDA(cudaFree(p));
    CHECK_CUDA(cudaFree(d_w_table));
    CHECK_CUDA(cudaFree(d_s_table));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_D_grouped));
    CHECK_CUDA(cudaFree(d_D_single));
    sh();

    if (bad != 0 || rel > 1e-3f || max_abs > 0.25f) {
        fprintf(stderr, "[%s] FAIL\n", c.name);
        return 9;
    }
    fprintf(stderr, "[%s] PASS\n", c.name);
    return 0;
}

int main(int argc, char ** argv) {
    const char * lib_path = argc > 1 ? argv[1] : "./libggml-turbomind.so";
    void * lib = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    const Case cases[] = {
        {4096, 2048, 1, "down_decode"},
        {2048, 4096, 1, "gate_up_decode"},
        {4096, 2048, 4, "down_prompt"},
        {2048, 4096, 4, "gate_up_prompt"},
    };

    int failures = 0;
    for (const Case & c : cases) {
        failures += run_case(lib, c) != 0;
    }
    dlclose(lib);
    return failures ? 1 : 0;
}

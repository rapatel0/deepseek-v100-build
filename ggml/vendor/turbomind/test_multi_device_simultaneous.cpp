// SPRINT-025-PATCH P0 — Simultaneous multi-device CUDA_TURBOMIND smoke.
//
// The earlier test_multi_device.cpp covers SEQUENTIAL dispatch on two
// devices (init 0 → dispatch 0 → init 1 → dispatch 1 → dispatch 0 again,
// each with cudaDeviceSynchronize between). That caught the singleton
// teardown bug fixed in SPRINT-025 P2.
//
// This test covers SIMULTANEOUS dispatch: launch GPU 0's mul_mat then
// switch to GPU 1 and launch its mul_mat with NO intervening sync. The
// kernels may overlap on the CUDA scheduler. If TURBOMIND's Gemm object,
// internal streams, or scratch buffers carry single-device assumptions
// past the workspace-pointer level, this is where it shows up.
//
// Pass criterion: each device's simultaneous-run output bit-matches its
// own single-device baseline. If multi-GPU 256e gibberish reproduces in
// this test, SPRINT-025-PATCH's bug is below the ggml-cuda integration
// layer (inside libggml-turbomind.so itself).

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

// Packed weight + input activations for one device. Caller owns the device
// allocations; this struct is just a bundle.
struct DeviceFixture {
    int      device;
    void   * d_w   = nullptr;   // packed weight
    void   * d_s   = nullptr;   // packed scales (or null)
    int      k_pack = 0;
    __half * d_A   = nullptr;   // input M×K
    __half * d_D   = nullptr;   // output M×N (per dispatch — caller may reuse)
    int      M = 0, N = 0, K = 0, group_size = 0;
};

static void prepare_fixture(
    DeviceFixture& f, int device,
    pfn_pack_weight pw, pfn_packed_bytes pb,
    int M, int N, int K, int group_size, uint32_t seed)
{
    f.device = device; f.M = M; f.N = N; f.K = K; f.group_size = group_size;
    CHECK(cudaSetDevice(device));

    std::vector<block_f8_e4m3_b128> blocks;
    make_fixture(blocks, N, K, seed);

    void * d_src = nullptr;
    CHECK(cudaMalloc(&d_src, blocks.size() * sizeof(block_f8_e4m3_b128)));
    CHECK(cudaMemcpy(d_src, blocks.data(), blocks.size() * sizeof(block_f8_e4m3_b128), cudaMemcpyHostToDevice));

    size_t wb = 0, sb = 0;
    if (pb(GGML_TM_DTYPE_F8_E4M3_B128, N, K, group_size, &wb, &sb) != 0) {
        fprintf(stderr, "packed_bytes failed on dev %d\n", device); std::exit(1);
    }
    CHECK(cudaMalloc(&f.d_w, wb));
    if (sb) CHECK(cudaMalloc(&f.d_s, sb));

    if (pw(d_src, GGML_TM_DTYPE_F8_E4M3_B128, N, K, group_size, f.d_w, f.d_s, &f.k_pack, nullptr) != 0) {
        fprintf(stderr, "pack failed on dev %d\n", device); std::exit(1);
    }
    cudaFree(d_src);

    // Input activations
    std::vector<__half> hA((size_t)M * K);
    std::mt19937 rng(seed ^ 0xCAFE);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    for (auto& v : hA) v = __float2half(ad(rng));
    CHECK(cudaMalloc(&f.d_A, hA.size() * sizeof(__half)));
    CHECK(cudaMemcpy(f.d_A, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Output buffer (one per fixture; will be reused across baseline + simul runs)
    CHECK(cudaMalloc(&f.d_D, (size_t)M * N * sizeof(__half)));
}

// Run a single dispatch + synchronize, copy output to host. Caller has
// already cudaSetDevice'd if needed.
static void run_and_readback(
    const DeviceFixture& f, pfn_mul_mat mm,
    std::vector<__half>& host_out)
{
    int rc = mm(f.d_A, f.d_w, f.d_s, GGML_TM_DTYPE_F8_E4M3_B128,
                f.M, f.N, f.K, f.group_size, f.k_pack, f.d_D, nullptr);
    if (rc != 0) { fprintf(stderr, "mul_mat rc=%d on dev %d\n", rc, f.device); std::exit(1); }
    CHECK(cudaDeviceSynchronize());
    host_out.resize((size_t)f.M * f.N);
    CHECK(cudaMemcpy(host_out.data(), f.d_D, host_out.size() * sizeof(__half), cudaMemcpyDeviceToHost));
}

// Bit-exact compare of two host fp16 arrays — for this test we expect
// identical bytes because the inputs are identical and the kernel is
// deterministic per device. If anything diverges, that's the bug.
static int compare(const std::vector<__half>& a, const std::vector<__half>& b, const char* label) {
    if (a.size() != b.size()) {
        fprintf(stderr, "[simul] %s: size mismatch %zu vs %zu\n", label, a.size(), b.size()); return 1;
    }
    int diff_count = 0;
    int first_diff = -1;
    float max_abs_diff = 0;
    int nan_count_a = 0, nan_count_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float fa = __half2float(a[i]);
        float fb = __half2float(b[i]);
        if (std::isnan(fa) || std::isinf(fa)) nan_count_a++;
        if (std::isnan(fb) || std::isinf(fb)) nan_count_b++;
        if (memcmp(&a[i], &b[i], sizeof(__half)) != 0) {
            if (first_diff < 0) first_diff = (int)i;
            diff_count++;
            float d = std::fabs(fa - fb);
            if (d > max_abs_diff) max_abs_diff = d;
        }
    }
    fprintf(stderr, "[simul] %s: %d/%zu bytes differ; first=%d; max|Δ|=%g; nan_a=%d nan_b=%d\n",
            label, diff_count, a.size(), first_diff, max_abs_diff, nan_count_a, nan_count_b);
    return diff_count == 0 ? 0 : 1;
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
    fprintf(stderr, "[simul] %d CUDA devices visible\n", dev_count);
    if (dev_count < 2) {
        fprintf(stderr, "[simul] SKIP: need >= 2 GPUs (have %d)\n", dev_count);
        return 0;
    }

    // Init both devices
    CHECK(cudaSetDevice(0));
    if (in(0) != 0) { fprintf(stderr, "init(0) failed\n"); return 2; }
    CHECK(cudaSetDevice(1));
    if (in(1) != 0) { fprintf(stderr, "init(1) failed\n"); return 2; }
    fprintf(stderr, "[simul] init(0) + init(1) ok\n");

    // SPRINT-025-PATCH P8: use the REAL DSv4-Flash-256e MoE-linear shape.
    // M=1 token (decode), N=K=2048 (typical DSv4 attn/ffn dim), F8_E4M3_B128.
    // If sequential Runs at this shape diverge, the kernel itself is shape-
    // dependent broken; if they remain bit-identical, the bug is integration-
    // level (gather/scatter, type conversion, or buft routing).
    const int M = 1, N = 2048, K = 2048, GS = 128;

    // Prepare per-device fixtures with DIFFERENT seeds so a cross-device
    // contamination produces a clear divergence (not just an exact-match
    // coincidence).
    DeviceFixture f0{}, f1{};
    prepare_fixture(f0, 0, pw, pb, M, N, K, GS, 0xC0FFEE);
    prepare_fixture(f1, 1, pw, pb, M, N, K, GS, 0xBEEFCAFE);
    fprintf(stderr, "[simul] fixtures prepared on GPU 0 + GPU 1\n");

    // -- Phase A: single-device baselines (deterministic, sync between) --
    std::vector<__half> base0, base1;
    CHECK(cudaSetDevice(0)); run_and_readback(f0, mm, base0);
    CHECK(cudaSetDevice(1)); run_and_readback(f1, mm, base1);
    fprintf(stderr, "[simul] baselines captured (each device sync'd alone)\n");

    // -- Phase B: simultaneous dispatch (NO sync between launches) --
    // Use FRESH output buffers so this phase cannot read leftover bytes
    // from Phase A.
    CHECK(cudaSetDevice(0)); CHECK(cudaFree(f0.d_D));
    CHECK(cudaMalloc(&f0.d_D, (size_t)M * N * sizeof(__half)));
    CHECK(cudaMemset(f0.d_D, 0, (size_t)M * N * sizeof(__half)));
    CHECK(cudaSetDevice(1)); CHECK(cudaFree(f1.d_D));
    CHECK(cudaMalloc(&f1.d_D, (size_t)M * N * sizeof(__half)));
    CHECK(cudaMemset(f1.d_D, 0, (size_t)M * N * sizeof(__half)));

    // Launch GPU 0 dispatch — DO NOT cudaDeviceSynchronize
    CHECK(cudaSetDevice(0));
    int rc0 = mm(f0.d_A, f0.d_w, f0.d_s, GGML_TM_DTYPE_F8_E4M3_B128,
                 M, N, K, GS, f0.k_pack, f0.d_D, nullptr);
    if (rc0 != 0) { fprintf(stderr, "[simul] GPU0 mul_mat rc=%d\n", rc0); return 3; }

    // Immediately switch to GPU 1 and launch — kernels may overlap on the
    // CUDA scheduler. If TURBOMIND's internal state (Gemm, streams, scratch)
    // carries single-device assumptions past the workspace-pointer level,
    // they will collide here.
    CHECK(cudaSetDevice(1));
    int rc1 = mm(f1.d_A, f1.d_w, f1.d_s, GGML_TM_DTYPE_F8_E4M3_B128,
                 M, N, K, GS, f1.k_pack, f1.d_D, nullptr);
    if (rc1 != 0) { fprintf(stderr, "[simul] GPU1 mul_mat rc=%d\n", rc1); return 3; }

    // Now sync both
    CHECK(cudaSetDevice(0)); CHECK(cudaDeviceSynchronize());
    CHECK(cudaSetDevice(1)); CHECK(cudaDeviceSynchronize());

    // Read back simultaneous-run outputs
    std::vector<__half> sim0((size_t)M * N), sim1((size_t)M * N);
    CHECK(cudaSetDevice(0)); CHECK(cudaMemcpy(sim0.data(), f0.d_D, sim0.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK(cudaSetDevice(1)); CHECK(cudaMemcpy(sim1.data(), f1.d_D, sim1.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    fprintf(stderr, "[simul] simultaneous-run outputs captured\n");

    // -- Compare --
    int r0 = compare(base0, sim0, "GPU0 baseline vs simul");
    int r1 = compare(base1, sim1, "GPU1 baseline vs simul");

    // -- Phase C: REPEATED sequential Runs on GPU 0's State. Tests whether
    //    State[0] accumulates drift across calls. SPRINT-025-PATCH bisection:
    //    P3.2 (6 sequential layers on one TURBOMIND device) produced gibberish
    //    in the full llama-server pipeline. This is a kernel-test repro check.
    fprintf(stderr, "[simul] Phase C: 16 sequential Runs on GPU 0\n");
    CHECK(cudaSetDevice(0));
    std::vector<__half> repeats[16];
    for (int i = 0; i < 16; ++i) {
        CHECK(cudaFree(f0.d_D));
        CHECK(cudaMalloc(&f0.d_D, (size_t)M * N * sizeof(__half)));
        CHECK(cudaMemset(f0.d_D, 0, (size_t)M * N * sizeof(__half)));
        run_and_readback(f0, mm, repeats[i]);
    }
    int rc_drift = 0;
    for (int i = 1; i < 16; ++i) {
        int diff = 0;
        for (size_t j = 0; j < base0.size(); ++j) {
            if (memcmp(&repeats[0][j], &repeats[i][j], sizeof(__half)) != 0) diff++;
        }
        if (diff != 0) {
            fprintf(stderr, "[simul] Phase C: Run %d differs from Run 0 in %d/%zu elements\n",
                    i, diff, base0.size());
            rc_drift = 1;
        }
    }
    if (rc_drift == 0) {
        fprintf(stderr, "[simul] Phase C: 16 sequential Runs all bit-identical (no State drift)\n");
    }

    sh();
    dlclose(h);

    if (r0 != 0 || r1 != 0 || rc_drift != 0) {
        fprintf(stderr,
            "[simul] FAIL — simultaneous dispatch diverges from single-device baseline.\n"
            "[simul] This indicates the multi-GPU CUDA_TURBOMIND regression is below\n"
            "[simul] the ggml-cuda integration layer (inside libggml-turbomind.so).\n");
        return 7;
    }
    fprintf(stderr, "[simul] PASS — simultaneous dispatch matches per-device baseline.\n");
    fprintf(stderr, "[simul] Bug is above libggml-turbomind.so; investigate ggml-cuda\n");
    fprintf(stderr, "[simul] integration / -ot routing / activation buft copy.\n");
    return 0;
}

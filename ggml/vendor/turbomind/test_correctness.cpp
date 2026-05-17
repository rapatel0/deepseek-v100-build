// SPRINT-023 P2.3 — Round-trip correctness gate for libggml-turbomind.so
//
// Generates a small synthetic GGML-formatted weight tensor (F8_E4M3_B128 or
// MXFP4), packs it through our C ABI, runs ggml_turbomind_mul_mat, and
// compares the output against a HOST-side reference computed by dequanting
// the same GGML blocks to FP32 and doing a plain matmul.
//
// Gate per SPRINT-015 P2 contract:
//   F8_E4M3_B128 col-parallel: max_abs ≤ 2e-2, p99_abs ≤ 1e-2
//   MXFP4        col-parallel: max_abs ≤ 2e-2, p99_abs ≤ 1e-2

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
#include <cerrno>
#include <string>

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
typedef int  (*pfn_mul_mat)(const void*, const void*, const void*,
                            int, int, int, int, int, int, void*, void*);

// ---- GGML block layouts (matches ggml-common.h) ----------------------------
struct block_f8_e4m3_b128 {
    uint8_t e;         // E8M0
    uint8_t qs[128];   // fp8 e4m3
};
struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[16];    // 32 fp4 packed
};

// E8M0 byte → FP32 scale. e=127 → 1.0, e=128 → 2.0, e=126 → 0.5.
static float e8m0_to_f32(uint8_t e) {
    if (e == 0) return 0.0f;
    if (e == 255) return INFINITY;  // NaN-like — won't appear in fixtures
    int exp_unbiased = (int)e - 127;
    return std::ldexp(1.0f, exp_unbiased);
}

// FP8 E4M3 byte → FP32 (manual decode; matches IEEE-like FN spec).
static float fp8_e4m3_to_f32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int exp  = (b >> 3) & 0xF;
    int mant = b & 0x7;
    float v;
    if (exp == 0) {
        if (mant == 0) v = 0.0f;
        else {
            v = std::ldexp((float)mant / 8.0f, -6);  // subnormal: 2^(-6) * mant/8
        }
    } else if (exp == 0xF && mant == 0x7) {
        v = NAN;  // E4M3FN reserves 0xFE/0xFF as NaN
    } else {
        v = std::ldexp(1.0f + (float)mant / 8.0f, exp - 7);
    }
    return sign ? -v : v;
}

// FP4 E2M1 nibble → FP32, official table from REPORT-15:
// {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign bit.
static float fp4_e2m1_to_f32(uint8_t nib) {
    static const float tbl[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int sign = (nib >> 3) & 1;
    float v = tbl[nib & 0x7];
    return sign ? -v : v;
}

// ---- Host reference matmul -------------------------------------------------
//
// Reference output: D[m, n] = sum over k of A[m, k] * dequant(B)[n, k] for
// the FP8 case. Both A and B are row-major; D is col-major to match the
// turbomind output layout.

static void ref_matmul_fp8(
    const __half* A, int M, int K,
    const block_f8_e4m3_b128* B, int N, int Kblk,
    float* D_rowmajor /* [M, N] row-major */)
{
    for (int n = 0; n < N; ++n) {
        const int blocks_per_row = K / 128;
        for (int m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                const block_f8_e4m3_b128& blk = B[n * blocks_per_row + b];
                float scale = e8m0_to_f32(blk.e);
                for (int i = 0; i < 128; ++i) {
                    int k = b * 128 + i;
                    float aval = __half2float(A[m * K + k]);
                    float bval = fp8_e4m3_to_f32(blk.qs[i]) * scale;
                    acc += aval * bval;
                }
            }
            D_rowmajor[(size_t)m * N + n] = acc;
        }
    }
    (void)Kblk;
}

static void ref_matmul_mxfp4(
    const __half* A, int M, int K,
    const block_mxfp4* B, int N,
    float* D_rowmajor)
{
    for (int n = 0; n < N; ++n) {
        const int blocks_per_row = K / 32;
        for (int m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                const block_mxfp4& blk = B[n * blocks_per_row + b];
                float scale = e8m0_to_f32(blk.e);
                for (int j = 0; j < 16; ++j) {
                    uint8_t byte = blk.qs[j];
                    uint8_t lo   = byte & 0x0F;
                    uint8_t hi   = byte >> 4;
                    int k0 = b * 32 + j;
                    int k1 = k0 + 16;
                    acc += __half2float(A[m * K + k0]) * fp4_e2m1_to_f32(lo) * scale;
                    acc += __half2float(A[m * K + k1]) * fp4_e2m1_to_f32(hi) * scale;
                }
            }
            D_rowmajor[(size_t)m * N + n] = acc;
        }
    }
}

// ---- Stats helper ----------------------------------------------------------
struct DiffStats { float max_abs; float p99_abs; float rel; };

static DiffStats compare(const std::vector<float>&  D_actual,
                         const std::vector<float>&  D_ref,
                         int M, int N)
{
    std::vector<float> abs_diff(M * N);
    float max_abs = 0.0f, max_ref = 0.0f, sum_abs = 0.0f, sum_ref = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float a = D_actual[i];
        float r = D_ref[i];
        float d = std::fabs(a - r);
        abs_diff[i] = d;
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_ref += std::fabs(r);
        max_ref = std::max(max_ref, std::fabs(r));
    }
    std::sort(abs_diff.begin(), abs_diff.end());
    int p99_idx = (int)((M * N) * 0.99);
    DiffStats s;
    s.max_abs = max_abs;
    s.p99_abs = abs_diff[p99_idx];
    s.rel = (sum_ref > 0) ? sum_abs / sum_ref : 0.0f;
    return s;
}

// ---- Fixtures --------------------------------------------------------------
static void make_fp8_fixture(std::vector<block_f8_e4m3_b128>& blocks, int N, int K,
                             uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(123, 130);    // scale ~ [0.0625, 8]
    // E4M3FN NaN encoding: exp=0xF AND mant=0x7 (i.e. 0x7F, 0xFF).
    auto rand_byte = [&]() {
        for (;;) {
            int b = rng() & 0xFF;
            if (b == 0x7F || b == 0xFF) continue;
            return (uint8_t)b;
        }
    };
    const int blocks_per_row = K / 128;
    blocks.resize((size_t)N * blocks_per_row);
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i].e = (uint8_t)e_dist(rng);
        for (int j = 0; j < 128; ++j) blocks[i].qs[j] = rand_byte();
    }
}

static void make_mxfp4_fixture(std::vector<block_mxfp4>& blocks, int N, int K,
                               uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> e_dist(123, 130);
    std::uniform_int_distribution<int> q_dist(0, 255);
    const int blocks_per_row = K / 32;
    blocks.resize((size_t)N * blocks_per_row);
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i].e = (uint8_t)e_dist(rng);
        for (int j = 0; j < 16; ++j) blocks[i].qs[j] = (uint8_t)q_dist(rng);
    }
}

template <typename BlockT>
static bool load_blocks_at(const char * path, long long offset, std::vector<BlockT>& blocks) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "fopen(%s): %s\n", path, strerror(errno));
        return false;
    }
    if (fseeko(f, (off_t) offset, SEEK_SET) != 0) {
        fprintf(stderr, "fseeko(%s, %lld): %s\n", path, offset, strerror(errno));
        fclose(f);
        return false;
    }
    const size_t want = blocks.size() * sizeof(BlockT);
    const size_t got  = fread(blocks.data(), 1, want, f);
    fclose(f);
    if (got != want) {
        fprintf(stderr, "short read from %s at %lld: got %zu want %zu\n", path, offset, got, want);
        return false;
    }
    return true;
}

static void scan_fp8_blocks(const std::vector<block_f8_e4m3_b128>& blocks, const char * tag) {
    size_t e_zero = 0, e_inf = 0, q_nan = 0;
    uint8_t e_min = 255, e_max = 0;
    for (const block_f8_e4m3_b128& blk : blocks) {
        e_zero += (blk.e == 0);
        e_inf  += (blk.e == 255);
        e_min = std::min(e_min, blk.e);
        e_max = std::max(e_max, blk.e);
        for (uint8_t q : blk.qs) {
            q_nan += (q == 0x7F || q == 0xFF);
        }
    }
    fprintf(stderr, "[%s] raw fp8 scan: blocks=%zu e_min=%u e_max=%u e_zero=%zu e_255=%zu q_nan=%zu\n",
            tag, blocks.size(), (unsigned)e_min, (unsigned)e_max, e_zero, e_inf, q_nan);
}

static void scan_mxfp4_blocks(const std::vector<block_mxfp4>& blocks, const char * tag) {
    size_t e_zero = 0, e_inf = 0;
    uint8_t e_min = 255, e_max = 0;
    for (const block_mxfp4& blk : blocks) {
        e_zero += (blk.e == 0);
        e_inf  += (blk.e == 255);
        e_min = std::min(e_min, blk.e);
        e_max = std::max(e_max, blk.e);
    }
    fprintf(stderr, "[%s] raw mxfp4 scan: blocks=%zu e_min=%u e_max=%u e_zero=%zu e_255=%zu\n",
            tag, blocks.size(), (unsigned)e_min, (unsigned)e_max, e_zero, e_inf);
}

// ---- Per-type test ----------------------------------------------------------
template <typename BlockT>
static int run_one_case(
    void* lib,
    int ggml_type,
    int M, int N, int K,
    int group_size,
    const std::vector<BlockT>& blocks,
    float gate_max_abs,
    float gate_p99,
    const char* tag)
{
    auto in  = (pfn_init)             dlsym(lib, "ggml_turbomind_init");
    auto sh  = (pfn_shutdown)         dlsym(lib, "ggml_turbomind_shutdown");
    auto pb  = (pfn_packed_bytes)     dlsym(lib, "ggml_turbomind_packed_bytes");
    auto pw  = (pfn_pack_weight)      dlsym(lib, "ggml_turbomind_pack_weight_expert");
    auto mm  = (pfn_mul_mat)          dlsym(lib, "ggml_turbomind_mul_mat");
    if (!in || !pb || !pw || !mm || !sh) { fprintf(stderr,"dlsym\n"); return 1; }

    if (in(0) != 0) { fprintf(stderr,"init\n"); return 2; }

    // ---- copy blocks H2D ----
    const size_t src_bytes = blocks.size() * sizeof(BlockT);
    void* d_src = nullptr;
    CHECK(cudaMalloc(&d_src, src_bytes));
    CHECK(cudaMemcpy(d_src, blocks.data(), src_bytes, cudaMemcpyHostToDevice));

    // ---- get packed sizes + alloc ----
    size_t wb, sb;
    int rc = pb(ggml_type, N, K, group_size, &wb, &sb);
    if (rc) { fprintf(stderr,"[%s] packed_bytes rc=%d\n", tag, rc); return 3; }

    void *d_wb = nullptr, *d_sb = nullptr;
    CHECK(cudaMalloc(&d_wb, wb));
    if (sb) CHECK(cudaMalloc(&d_sb, sb));

    // ---- pack ----
    int k_pack = 0;
    rc = pw(d_src, ggml_type, N, K, group_size, d_wb, d_sb, &k_pack, nullptr);
    if (rc) { fprintf(stderr,"[%s] pack rc=%d\n", tag, rc); return 4; }
    fprintf(stderr,"[%s] pack OK, k_pack=0x%x (b=0x%x, v=0x%x)\n",
            tag, k_pack, k_pack & 0xFFF, (k_pack >> 12) & 0xFFF);

    const int M_run = (M + 7) & ~7;

    // ---- generate FP16 activation ----
    std::mt19937 rng(0xDEADBEEF);
    std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
    std::vector<__half> hA((size_t)M_run * K, __float2half(0.0f));
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            hA[(size_t)m * K + k] = __float2half(ad(rng));
        }
    }

    __half* dA = nullptr;
    CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
    CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // ---- output ----
    __half* dD = nullptr;
    CHECK(cudaMalloc(&dD, (size_t)M_run * N * sizeof(__half)));

    // ---- run turbomind mul_mat ----
    rc = mm(dA, d_wb, d_sb, ggml_type, M_run, N, K, group_size, k_pack, dD, nullptr);
    if (rc) { fprintf(stderr,"[%s] mul_mat rc=%d\n", tag, rc); return 5; }

    // ---- copy result back ----
    std::vector<__half> hD_half((size_t)M * N);
    CHECK(cudaMemcpy(hD_half.data(), dD, hD_half.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> hD((size_t)M * N);
    for (size_t i = 0; i < hD.size(); ++i) {
        hD[i] = __half2float(hD_half[i]);
    }

    // ---- compute host reference ----
    std::vector<float> hD_ref((size_t)M * N, 0.0f);
    if (ggml_type == GGML_TM_DTYPE_F8_E4M3_B128) {
        ref_matmul_fp8(hA.data(), M, K,
                       reinterpret_cast<const block_f8_e4m3_b128*>(blocks.data()),
                       N, 128, hD_ref.data());
    } else {
        ref_matmul_mxfp4(hA.data(), M, K,
                        reinterpret_cast<const block_mxfp4*>(blocks.data()),
                        N, hD_ref.data());
    }

    // ---- compare ----
    DiffStats s = compare(hD, hD_ref, M, N);
    fprintf(stderr,"[%s] M=%d M_run=%d N=%d K=%d output=f16: max_abs=%.4e p99=%.4e rel=%.4e | gates max=%.0e p99=%.0e\n",
            tag, M, M_run, N, K, s.max_abs, s.p99_abs, s.rel, gate_max_abs, gate_p99);

    cudaFree(dA); cudaFree(dD); cudaFree(d_wb);
    if (d_sb) cudaFree(d_sb);
    cudaFree(d_src);
    sh();

    // Gates: relative-error focused since output magnitudes here are
    // O(100..1000), where the FP16 rounding step alone is up to 1.0.
    // - rel  (sum|diff| / sum|ref|): tight — should be FP16-precision-limited
    // - max_abs: looser absolute floor (1 ULP at the max output magnitude)
    float max_ref = 0;
    for (int i = 0; i < M*N; ++i) max_ref = std::max(max_ref, std::fabs(hD_ref[i]));
    float fp16_ulp_at_max = std::ldexp(1.0f, std::ilogb(std::max(max_ref, 1.0f)) - 10);
    float effective_max_abs_gate = std::max(gate_max_abs, 2.0f * fp16_ulp_at_max);
    if (s.rel > 1e-3f || s.max_abs > effective_max_abs_gate) {
        fprintf(stderr,"[%s] FAIL — rel=%.4e (gate ≤ 1e-3) or max_abs=%.4e > %.4f (2 ULP at %.0f)\n",
                tag, s.rel, s.max_abs, effective_max_abs_gate, max_ref);
        return 6;
    }
    fprintf(stderr,"[%s] PASS — rel=%.4e, max_abs=%.4e (gate %.4e, 2-ULP at %.0f)\n",
            tag, s.rel, s.max_abs, effective_max_abs_gate, max_ref);
    return 0;
}

int main(int argc, char** argv) {
    const char* lib_path = (argc > 1) ? argv[1] : "./libggml-turbomind.so";
    void* h = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!h) { fprintf(stderr,"dlopen failed: %s\n", dlerror()); return 1; }

    if (argc == 9 && std::string(argv[2]) == "--raw-f8") {
        const char * model_path = argv[3];
        const long long offset  = strtoll(argv[4], nullptr, 0);
        const int M             = atoi(argv[5]);
        const int N             = atoi(argv[6]);
        const int K             = atoi(argv[7]);
        const char * tag        = argv[8];
        std::vector<block_f8_e4m3_b128> blocks((size_t) N * (K / 128));
        if (!load_blocks_at(model_path, offset, blocks)) {
            dlclose(h);
            return 1;
        }
        scan_fp8_blocks(blocks, tag);
        const int rc = run_one_case<block_f8_e4m3_b128>(
            h, GGML_TM_DTYPE_F8_E4M3_B128, M, N, K, 128, blocks,
            /*max=*/2e-2f, /*p99=*/1e-2f, tag);
        dlclose(h);
        return rc;
    }

    if (argc == 9 && std::string(argv[2]) == "--raw-mxfp4") {
        const char * model_path = argv[3];
        const long long offset  = strtoll(argv[4], nullptr, 0);
        const int M             = atoi(argv[5]);
        const int N             = atoi(argv[6]);
        const int K             = atoi(argv[7]);
        const char * tag        = argv[8];
        std::vector<block_mxfp4> blocks((size_t) N * (K / 32));
        if (!load_blocks_at(model_path, offset, blocks)) {
            dlclose(h);
            return 1;
        }
        scan_mxfp4_blocks(blocks, tag);
        const int rc = run_one_case<block_mxfp4>(
            h, GGML_TM_DTYPE_MXFP4, M, N, K, 32, blocks,
            /*max=*/2e-2f, /*p99=*/1e-2f, tag);
        dlclose(h);
        return rc;
    }

    // Sized to engage smallest sm70_884 CTA tile (M=8, N=128, K=64+).
    const int M = 8, N = 256, K = 256;
    std::vector<block_f8_e4m3_b128> f8_blocks;
    make_fp8_fixture(f8_blocks, N, K, 0xC0FFEE);
    int rc1 = run_one_case<block_f8_e4m3_b128>(
        h, GGML_TM_DTYPE_F8_E4M3_B128, M, N, K, 128, f8_blocks,
        /*max=*/2e-2f, /*p99=*/1e-2f, "F8_E4M3_B128");

    // ---- MXFP4 fixture ----
    std::vector<block_mxfp4> fp4_blocks;
    make_mxfp4_fixture(fp4_blocks, N, K, 0xBEEFCAFE);
    int rc2 = run_one_case<block_mxfp4>(
        h, GGML_TM_DTYPE_MXFP4, M, N, K, 32, fp4_blocks,
        /*max=*/2e-2f, /*p99=*/1e-2f, "MXFP4");

    // Decode-time DSv4 shared-expert shapes. These exercise the single-tensor
    // path at M=1, which is not covered by the routed MoE grouped tests.
    std::vector<block_f8_e4m3_b128> f8_shexp_up;
    make_fp8_fixture(f8_shexp_up, 2048, 4096, 0x51EAD001);
    int rc3 = run_one_case<block_f8_e4m3_b128>(
        h, GGML_TM_DTYPE_F8_E4M3_B128, 1, 2048, 4096, 128, f8_shexp_up,
        /*max=*/2e-2f, /*p99=*/1e-2f, "F8_E4M3_B128_M1_SHEXP_UP");
    int rc3b = run_one_case<block_f8_e4m3_b128>(
        h, GGML_TM_DTYPE_F8_E4M3_B128, 4, 2048, 4096, 128, f8_shexp_up,
        /*max=*/2e-2f, /*p99=*/1e-2f, "F8_E4M3_B128_M4_SHEXP_UP");

    std::vector<block_f8_e4m3_b128> f8_shexp_down;
    make_fp8_fixture(f8_shexp_down, 4096, 2048, 0x51EAD002);
    int rc4 = run_one_case<block_f8_e4m3_b128>(
        h, GGML_TM_DTYPE_F8_E4M3_B128, 1, 4096, 2048, 128, f8_shexp_down,
        /*max=*/2e-2f, /*p99=*/1e-2f, "F8_E4M3_B128_M1_SHEXP_DOWN");
    int rc4b = run_one_case<block_f8_e4m3_b128>(
        h, GGML_TM_DTYPE_F8_E4M3_B128, 4, 4096, 2048, 128, f8_shexp_down,
        /*max=*/2e-2f, /*p99=*/1e-2f, "F8_E4M3_B128_M4_SHEXP_DOWN");

    std::vector<block_mxfp4> fp4_shexp_up;
    make_mxfp4_fixture(fp4_shexp_up, 2048, 4096, 0x51EAD003);
    int rc5 = run_one_case<block_mxfp4>(
        h, GGML_TM_DTYPE_MXFP4, 1, 2048, 4096, 32, fp4_shexp_up,
        /*max=*/2e-2f, /*p99=*/1e-2f, "MXFP4_M1_SHEXP_UP");
    int rc5b = run_one_case<block_mxfp4>(
        h, GGML_TM_DTYPE_MXFP4, 4, 2048, 4096, 32, fp4_shexp_up,
        /*max=*/2e-2f, /*p99=*/1e-2f, "MXFP4_M4_SHEXP_UP");

    dlclose(h);
    return (rc1 || rc2 || rc3 || rc3b || rc4 || rc4b || rc5 || rc5b) ? 1 : 0;
}

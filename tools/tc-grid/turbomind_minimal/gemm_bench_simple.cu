// SPRINT-020 P1.6 — turbomind ceiling reference via Gemm::Run.
//
// IMPORTANT FINDING (P1.6 first runtime attempt):
// Turbomind's sm70_s884 registry has NO INT8 (int8_t/uint8_t) weight
// kernel. The 8-bit-weight sm70 path is FP8 e4m3 (Config_E4M3); the
// 4-bit path is U4 (Config_U4_g). The original SPRINT-020 §1.1 plan
// to head-to-head v12_ms3 INT8 vs "turbomind INT8" is physically
// impossible on sm70 — turbomind's INT8 weight kernels are sm75+.
//
// LEGITIMATE sm70 ceiling proxies that turbomind DOES support:
//   1. cuBLAS FP16 GEMM (Config-less, no Convert needed) — captures
//      the raw HBM-bound peak with no quantization
//   2. Config_F16 (sm70_884 HMMA FP16, packed) — turbomind's own
//      HMMA-only ceiling, no dequant path
//   3. Config_E4M3 (sm70_884 FP8 weights) — same 1B/weight bandwidth
//      as INT8, different precision encoding
//
// This bench measures (1) cuBLAS FP16 path: simplest, no Convert.
// It produces the FP16 ceiling reference for sm70 on the asymmetric
// DSv4 shapes. v12_ms3's INT8 result vs this number tells us how
// much bandwidth amplification quantization is actually buying us.
//
// Once this works, follow-up: measure (2) and (3) for full triangulation.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/core/data_type.h"

namespace tmg = turbomind::gemm;
using turbomind::DataType;

namespace {

struct Args {
    std::vector<int> m_list = {64, 2048};
    std::vector<int> n_list = {7168};
    std::vector<int> k_list = {7168};
    int n_warmup = 2;
    int n_timed  = 5;
};

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    const char* p = s;
    while (*p) {
        char* end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out.push_back((int) v);
        p = end;
        if (*p == ',') ++p;
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--m-list") && i+1 < argc) a.m_list = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--n-list") && i+1 < argc) a.n_list = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--k-list") && i+1 < argc) a.k_list = parse_list(argv[++i]);
        else if (!strcmp(argv[i], "--nk") && i+1 < argc) {
            auto v = parse_list(argv[++i]);
            a.n_list = v; a.k_list = v;
        } else if (!strcmp(argv[i], "--warmup") && i+1 < argc) a.n_warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--timed") && i+1 < argc) a.n_timed = atoi(argv[++i]);
    }
    if (a.n_list.size() != a.k_list.size()) {
        fprintf(stderr, "[error] --n-list and --k-list must be same length OR use --nk\n");
        std::exit(2);
    }
    return a;
}

double median_ms(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

#define CUDA_CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %d at %s:%d: %s\n", e, __FILE__, __LINE__, \
            cudaGetErrorString(e)); std::exit(1); } } while(0)

void fill_fp16(__half* p, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) p[i] = __float2half(d(rng));
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "device: %s, cc=%d.%d, smem=%zuKiB/cta\n",
            prop.name, prop.major, prop.minor, prop.sharedMemPerBlock / 1024);
    fprintf(stderr, "format,path,dist,M,N,K,tile,status,detail\n");

    // Workspace shared across all sizes — sized for the largest M*N.
    int max_m = *std::max_element(a.m_list.begin(), a.m_list.end());
    int max_n = *std::max_element(a.n_list.begin(), a.n_list.end());
    size_t partials_bytes = (size_t) max_m * max_n * sizeof(float) * 4;
    void* d_barriers = nullptr; CUDA_CHECK(cudaMalloc(&d_barriers, tmg::Gemm::kBarriersSize));
    void* d_partials = nullptr; CUDA_CHECK(cudaMalloc(&d_partials, partials_bytes));
    int*  d_flags    = nullptr; CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * 1024));

    tmg::Workspace workspace{};
    workspace.barriers        = d_barriers;
    workspace.barriers_size   = tmg::Gemm::kBarriersSize;
    workspace.partials        = d_partials;
    workspace.partials_size   = partials_bytes;
    workspace.tensormaps      = nullptr;
    workspace.tensormaps_size = 0;
    workspace.flags           = d_flags;

    tmg::Gemm gemm;

    for (size_t i = 0; i < a.n_list.size(); ++i) {
        const int N = a.n_list[i];
        const int K = a.k_list[i];
        for (int M : a.m_list) {
            std::vector<__half> hA((size_t) M * K);
            std::vector<__half> hB((size_t) K * N);
            fill_fp16(hA.data(), hA.size(), 0xABCDE);
            fill_fp16(hB.data(), hB.size(), 0x12345);

            __half *dA, *dB, *dD;
            CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&dD, (size_t) M * N * sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half), cudaMemcpyHostToDevice));

            // FP16 GEMM, no quantization. cuBLAS feasibility (cublas.cu
            // is_feasible) requires:
            //   - pack_{a,b,u,v} = 0     (no packing — natural)
            //   - striding all kFlat     (nullptr offsets — natural)
            //   - order_c == kColMajor   (set on Ddesc below)
            //   - type_a == type_b == fp16 (set below)
            //   - no quant_a / quant_b   (op.quant_* = kNone)
            //   - num == 1, epilogue == kNone, group_axis < 0 (defaults)
            tmg::Operation op{};
            op.dispatch  = tmg::DispatchPolicy::kDefault;
            op.epilogue  = tmg::Epilogue::kNone;
            op.quant_a   = tmg::QuantDesc{tmg::QuantType::kNone, 0};
            op.quant_b   = tmg::QuantDesc{tmg::QuantType::kNone, 0};
            op.batch_dim = 0;

            // context.cu's get_gemm_desc reads:
            //   m0 = Adesc.rows, k0 = Adesc.cols       (A is M×K)
            //   k1 = Bdesc.rows, n0 = Bdesc.cols       (B is K×N)
            //   m1 = Ddesc.rows, n1 = Ddesc.cols       (D is M×N)
            tmg::MatrixLayout Adesc{turbomind::kHalf, tmg::Order::kRowMajor,
                                    M, K, K, 0, 1, nullptr, nullptr};
            tmg::MatrixLayout Bdesc{turbomind::kHalf, tmg::Order::kRowMajor,
                                    K, N, N, 0, 1, nullptr, nullptr};
            tmg::MatrixLayout Ddesc{turbomind::kHalf, tmg::Order::kColMajor,
                                    M, N, M, 0, 1, nullptr, nullptr};
            tmg::MatrixLayout Cdesc = Ddesc;
            tmg::MatrixLayout Udesc{};  // unused
            tmg::MatrixLayout Vdesc{};  // unused

            cudaStream_t stream = 0;
            int rc = 0;
            for (int w = 0; w < a.n_warmup; ++w) {
                rc = gemm.Run(op, 1.0f, dA, Adesc, nullptr, Udesc,
                              dB, Bdesc, nullptr, Vdesc, 0.0f, nullptr, Cdesc,
                              dD, Ddesc, workspace, stream);
                if (rc != 0) break;
            }
            if (rc != 0) {
                printf("FP16,turbomind,U(-1,1),%d,%d,%d,turbomind_default,SKIP,Gemm::Run rc=%d\n", M, N, K, rc);
                fflush(stdout);
                cudaFree(dA); cudaFree(dB); cudaFree(dD);
                continue;
            }

            cudaEvent_t e_start, e_stop;
            cudaEventCreate(&e_start);
            cudaEventCreate(&e_stop);
            std::vector<float> samples;
            samples.reserve(a.n_timed);
            for (int t = 0; t < a.n_timed; ++t) {
                cudaEventRecord(e_start, stream);
                gemm.Run(op, 1.0f, dA, Adesc, nullptr, Udesc,
                         dB, Bdesc, nullptr, Vdesc, 0.0f, nullptr, Cdesc,
                         dD, Ddesc, workspace, stream);
                cudaEventRecord(e_stop, stream);
                cudaEventSynchronize(e_stop);
                float ms = 0;
                cudaEventElapsedTime(&ms, e_start, e_stop);
                samples.push_back(ms);
            }
            const double ms_med = median_ms(samples);
            const double tflops = (2.0 * M * N * K) / (ms_med * 1e9);
            // FP16 BW model: A (2*M*K) + B (2*K*N) + D (2*M*N) bytes.
            const double gbps = (2.0 * M * K + 2.0 * (size_t) K * N + 2.0 * M * N) / (ms_med * 1e6);

            printf("FP16,turbomind,U(-1,1),%d,%d,%d,turbomind_default,OK,ms=%.3f,tflops=%.2f,gbps=%.1f\n",
                   M, N, K, ms_med, tflops, gbps);
            fflush(stdout);

            cudaEventDestroy(e_start);
            cudaEventDestroy(e_stop);
            cudaFree(dA); cudaFree(dB); cudaFree(dD);
        }
    }

    cudaFree(d_barriers);
    cudaFree(d_partials);
    cudaFree(d_flags);
    return 0;
}

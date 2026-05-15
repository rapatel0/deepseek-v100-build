// SPRINT-021 P0 — measure turbomind's packed-weight kernels on V100.
//
// Configs (sm70_s884 registry, from arch/config_sm70_s884.h):
//   - fp16  : Config_F16 (no quant, Operand_B_Pack<half>)
//   - fp8   : Config_E4M3 (Operand_B_Pack<fp8_e4m3_t> + Operand_V_Pack<uint16_t>)
//   - u4    : Config_U4_g (Operand_B_Pack<uint4_t> + Operand_V_Pack<uint32_t>)
//   - fp4   : Config_MXF4 (Operand_B_Pack<fp4_e2m1_t> + Operand_V_Pack<uint8_t>)
//
// Each non-FP16 config requires the weight to be packed into a kernel-
// specific layout via `gemm::GetConverters()` + `LayoutConverter::Convert()`.
// The packing dance mirrors `models/linear_weight.cc::prepare()` (the
// "General quantization format conversion path" at lines 148-211).
//
// Rationale: SPRINT-020 P3 left a hole — we measured FP16 ceiling (87 TF)
// but didn't measure the configs that have the same 1B/wt bandwidth as
// INT8 (FP8) or half the bandwidth (U4/FP4). DSv4 is an FP4/FP8 model;
// if those configs achieve high V100 perf, we should run DSv4 natively
// in its trained precision rather than dequant-to-INT8.

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
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/core/data_type.h"

namespace tmg = turbomind::gemm;
using turbomind::DataType;

namespace {

enum class Config { fp16, fp8, u4, fp4 };

const char* name(Config c) {
    switch (c) {
        case Config::fp16: return "fp16";
        case Config::fp8:  return "fp8";
        case Config::u4:   return "u4";
        case Config::fp4:  return "fp4";
    }
    return "unknown";
}

DataType weight_dtype(Config c) {
    switch (c) {
        case Config::fp16: return turbomind::kHalf;
        case Config::fp8:  return turbomind::kFloat8_e4m3;
        case Config::u4:   return turbomind::kUint4;
        case Config::fp4:  return turbomind::kFloat4_e2m1;
    }
    return turbomind::kHalf;
}

DataType scale_dtype(Config c) {
    // Per ResolveLinearWeightFormat in core/data_format.cc:
    //   fp8 → kFloat scales
    //   u4  → data_type scales (we use kHalf)
    //   fp4 → kUint8 scales (E8M0 exponent)
    //   fp16 → no scales
    switch (c) {
        case Config::fp16: return turbomind::kHalf;  // unused
        case Config::fp8:  return turbomind::kFloat;
        case Config::u4:   return turbomind::kHalf;
        case Config::fp4:  return turbomind::kUint8;
    }
    return turbomind::kHalf;
}

int bits_per_weight(Config c) {
    switch (c) {
        case Config::fp16: return 16;
        case Config::fp8:  return 8;
        case Config::u4:   return 4;
        case Config::fp4:  return 4;
    }
    return 0;
}

struct Args {
    std::vector<int> m_list = {64, 2048};
    std::vector<int> n_list = {7168};
    std::vector<int> k_list = {7168};
    int n_warmup = 2;
    int n_timed  = 5;
    int group_size = 128;
    std::vector<Config> configs = {Config::fp16, Config::fp8, Config::u4, Config::fp4};
};

std::vector<int> parse_int_list(const char* s) {
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

std::vector<Config> parse_configs(const char* s) {
    std::vector<Config> out;
    std::string s2(s);
    std::string token;
    for (char ch : s2 + ",") {
        if (ch == ',') {
            if (token == "fp16") out.push_back(Config::fp16);
            else if (token == "fp8") out.push_back(Config::fp8);
            else if (token == "u4")  out.push_back(Config::u4);
            else if (token == "fp4") out.push_back(Config::fp4);
            token.clear();
        } else {
            token.push_back(ch);
        }
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--m-list") && i+1 < argc) a.m_list = parse_int_list(argv[++i]);
        else if (!strcmp(argv[i], "--n-list") && i+1 < argc) a.n_list = parse_int_list(argv[++i]);
        else if (!strcmp(argv[i], "--k-list") && i+1 < argc) a.k_list = parse_int_list(argv[++i]);
        else if (!strcmp(argv[i], "--nk") && i+1 < argc) {
            auto v = parse_int_list(argv[++i]);
            a.n_list = v; a.k_list = v;
        } else if (!strcmp(argv[i], "--warmup") && i+1 < argc) a.n_warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--timed") && i+1 < argc) a.n_timed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--group-size") && i+1 < argc) a.group_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--configs") && i+1 < argc) a.configs = parse_configs(argv[++i]);
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

// Random byte fill for sub-byte / fp8 / u8 quantized weights. The exact
// value distribution doesn't matter for perf measurement.
void fill_bytes(void* p, size_t bytes, uint32_t seed) {
    std::mt19937 rng(seed);
    auto* b = (uint8_t*) p;
    for (size_t i = 0; i < bytes; ++i) b[i] = (uint8_t) (rng() & 0xff);
}

struct PackedWeight {
    void*        weight_data = nullptr;
    void*        scales_data = nullptr;
    tmg::MatrixLayout k_desc{};
    tmg::MatrixLayout q_desc{};
    bool         ok = false;
};

// Mirror of linear_weight.cc::prepare()'s "General quantization format
// conversion path". Builds packed weight + scale buffers ready for
// Gemm::Run. For fp16 (no quant), returns ok=false so the caller knows
// to skip the packing entirely.
PackedWeight build_packed_weight(Config c, int N, int K, int group_size,
                                  cudaStream_t stream) {
    PackedWeight pw{};
    if (c == Config::fp16) return pw;  // caller handles unpacked path

    const DataType wdt = weight_dtype(c);
    const DataType sdt = scale_dtype(c);
    const DataType act = turbomind::kHalf;  // model activation type

    auto convs = tmg::GetConverters(/*data_type=*/act,
                                    /*weight_type=*/wdt,
                                    /*input_type=*/act,
                                    /*grouped=*/false,
                                    /*sm=*/70);
    const tmg::LayoutConverter* conv_w = convs[0];
    const tmg::LayoutConverter* conv_s = convs[1];
    if (!conv_w) {
        fprintf(stderr, "[%s] GetConverters returned null conv_w; sm70 may not support this config\n", name(c));
        return pw;
    }

    const int bits = bits_per_weight(c);
    const int output_dim = N;
    const int input_dim  = K;
    const size_t n_elem  = (size_t) output_dim * input_dim;

    // Raw weight buffer (sub-byte for u4/fp4, byte for fp8). Allocate
    // at byte granularity using turbomind's byte_size helper.
    const size_t raw_bytes = turbomind::byte_size(wdt, n_elem);
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, raw_bytes));

    // Host fill, then upload. For sub-byte the host bytes pack two
    // 4-bit values per byte, which is fine for perf-only.
    std::vector<uint8_t> host_raw(raw_bytes);
    fill_bytes(host_raw.data(), host_raw.size(), 0x12345 ^ (uint32_t)wdt);
    CUDA_CHECK(cudaMemcpy(raw, host_raw.data(), raw_bytes, cudaMemcpyHostToDevice));

    // Expand to uint16 temporary for the Convert. extend_to_u16
    // handles 4-bit and 8-bit input widths.
    uint16_t* tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp, n_elem * sizeof(uint16_t)));
    if (bits == 4) {
        turbomind::extend_to_u16(tmp, (const turbomind::uint4_t*) raw, n_elem, stream);
    } else if (bits == 8) {
        turbomind::extend_to_u16(tmp, (const uint8_t*) raw, n_elem, stream);
    } else {
        fprintf(stderr, "[%s] unexpected bits=%d for packed config\n", name(c), bits);
        cudaFree(raw); cudaFree(tmp);
        return pw;
    }

    // If conv_w expects RowMajor, the upstream code transposes tmp
    // ((output_dim,input_dim) row → col swap). Do the same.
    if (conv_w->order == tmg::kRowMajor) {
        uint16_t* trans = nullptr;
        CUDA_CHECK(cudaMalloc(&trans, n_elem * sizeof(uint16_t)));
        turbomind::invokeTransposeAxis01(trans, tmp, input_dim, output_dim, 1, stream);
        cudaFree(tmp);
        tmp = trans;
    }

    // Build w_desc per linear_weight.cc line 180-186.
    tmg::MatrixLayout w_desc{
        act,  // tmp is uint16_t but the desc.type advertises the
              // upcast type per upstream convention
        conv_w->order,
        output_dim,
        input_dim,
        conv_w->order == tmg::kRowMajor ? input_dim : output_dim,
    };
    // For B operand: swap rows/cols, flip order (line 188-191).
    const bool is_A = tmg::get_operand_tag(conv_w->pack) == tmg::OPERAND_A;
    if (!is_A) {
        std::swap(w_desc.rows, w_desc.cols);
        w_desc.order = ~w_desc.order;
    }

    // kd is the OUTPUT layout — the packed weight format that the
    // kernel will read.
    tmg::MatrixLayout kd = w_desc;
    if (bits == 4) {
        kd.type = turbomind::data_type_v<turbomind::uint4_t>;
    } else if (bits == 8) {
        kd.type = turbomind::data_type_v<uint8_t>;
    }
    kd.pack = conv_w->pack;

    // Allocate packed output buffer.
    void* packed = nullptr;
    CUDA_CHECK(cudaMalloc(&packed, raw_bytes));
    CUDA_CHECK(cudaMemsetAsync(packed, 0, raw_bytes, stream));

    int rc = conv_w->Convert(tmp, w_desc, packed, kd, stream);
    cudaFree(tmp);
    if (rc != 0) {
        fprintf(stderr, "[%s] conv_w->Convert failed rc=%d\n", name(c), rc);
        cudaFree(raw); cudaFree(packed);
        return pw;
    }

    // Final k_desc carries the real weight dtype (line 206).
    kd.type = wdt;
    kd.num = 1;  // MatrixLayout default-init leaves num=0 which fails dim check
    pw.weight_data = packed;
    pw.k_desc = kd;
    cudaFree(raw);

    // ---- scales ----
    if (conv_s) {
        const int n_scales = output_dim * (input_dim / group_size);
        // Source scales tensor (post-conversion type expected by
        // convert; upstream uses uint8_t/uint16_t/uint32_t per branch).
        DataType src_type = sdt;  // raw scale storage type
        if (sdt == turbomind::kFloat) src_type = turbomind::kUint32;
        else if (sdt == turbomind::kHalf) src_type = turbomind::kUint16;
        else if (sdt == turbomind::kUint8) src_type = turbomind::kUint8;

        const size_t scale_raw_bytes = turbomind::byte_size(src_type, n_scales);
        void* scale_raw = nullptr;
        CUDA_CHECK(cudaMalloc(&scale_raw, scale_raw_bytes));
        std::vector<uint8_t> host_scale(scale_raw_bytes);
        fill_bytes(host_scale.data(), host_scale.size(), 0xCAFE ^ (uint32_t)wdt);
        CUDA_CHECK(cudaMemcpy(scale_raw, host_scale.data(), scale_raw_bytes, cudaMemcpyHostToDevice));

        tmg::MatrixLayout s_desc{
            src_type,
            conv_s->order,
            output_dim,
            input_dim / group_size,
            output_dim,
        };
        const bool s_is_A = tmg::get_operand_tag(conv_s->pack) == tmg::OPERAND_U;
        if (!s_is_A) {
            std::swap(s_desc.rows, s_desc.cols);
            s_desc.order = ~s_desc.order;
        }

        tmg::MatrixLayout qd = s_desc;
        qd.pack = conv_s->pack;

        void* packed_s = nullptr;
        CUDA_CHECK(cudaMalloc(&packed_s, scale_raw_bytes));
        CUDA_CHECK(cudaMemsetAsync(packed_s, 0, scale_raw_bytes, stream));
        int src_rc = conv_s->Convert(scale_raw, s_desc, packed_s, qd, stream);
        cudaFree(scale_raw);
        if (src_rc != 0) {
            fprintf(stderr, "[%s] conv_s->Convert failed rc=%d\n", name(c), src_rc);
            cudaFree(pw.weight_data); cudaFree(packed_s);
            pw.weight_data = nullptr;
            return pw;
        }

        qd.num = 1;  // same fix as for kd
        pw.scales_data = packed_s;
        pw.q_desc = qd;
    }

    pw.ok = true;
    return pw;
}

void free_packed(PackedWeight& pw) {
    if (pw.weight_data) cudaFree(pw.weight_data);
    if (pw.scales_data) cudaFree(pw.scales_data);
    pw.weight_data = nullptr;
    pw.scales_data = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "device: %s, cc=%d.%d, smem=%zuKiB/cta\n",
            prop.name, prop.major, prop.minor, prop.sharedMemPerBlock / 1024);
    fprintf(stderr, "format,path,dist,M,N,K,tile,status,detail\n");

    // Workspace sized once at startup.
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
    cudaStream_t stream = 0;

    for (Config c : a.configs) {
        for (size_t i = 0; i < a.n_list.size(); ++i) {
            const int N = a.n_list[i];
            const int K = a.k_list[i];
            if ((c != Config::fp16) && (K % a.group_size != 0)) {
                fprintf(stderr, "[%s] skip N=%d K=%d (K not divisible by group_size=%d)\n",
                        name(c), N, K, a.group_size);
                continue;
            }

            // Build packed weight + scales once per (config, N, K). The
            // weight matrix is reused across all M values for this shape.
            PackedWeight pw{};
            if (c != Config::fp16) {
                pw = build_packed_weight(c, N, K, a.group_size, stream);
                if (!pw.ok) {
                    for (int M : a.m_list) {
                        printf("%s,turbomind,U(-1,1),%d,%d,%d,turbomind_packed,SKIP,packing failed\n",
                               name(c), M, N, K);
                    }
                    fflush(stdout);
                    continue;
                }
            }

            for (int M : a.m_list) {
                std::vector<__half> hA((size_t) M * K);
                fill_fp16(hA.data(), hA.size(), 0xABCDE);
                __half *dA, *dD;
                CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
                CUDA_CHECK(cudaMalloc(&dD, (size_t) M * N * sizeof(__half)));
                CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice));

                tmg::MatrixLayout Adesc{turbomind::kHalf, tmg::Order::kRowMajor,
                                        M, K, K, 0, 1, nullptr, nullptr};
                tmg::MatrixLayout Ddesc;
                tmg::MatrixLayout Bdesc, Vdesc;
                tmg::Operation op{};
                op.dispatch  = tmg::DispatchPolicy::kDefault;
                op.epilogue  = tmg::Epilogue::kNone;
                op.quant_a   = tmg::QuantDesc{tmg::QuantType::kNone, 0};
                op.batch_dim = 0;

                if (c == Config::fp16) {
                    // FP16 cuBLAS path: B is fp16 K×N row-major,
                    // D is fp16 ColMajor (cuBLAS requirement).
                    std::vector<__half> hB((size_t) K * N);
                    fill_fp16(hB.data(), hB.size(), 0x12345);
                    __half* dB;
                    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(__half)));
                    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half), cudaMemcpyHostToDevice));
                    Bdesc = tmg::MatrixLayout{turbomind::kHalf, tmg::Order::kRowMajor, K, N, N, 0, 1, nullptr, nullptr};
                    Ddesc = tmg::MatrixLayout{turbomind::kHalf, tmg::Order::kColMajor, M, N, M, 0, 1, nullptr, nullptr};
                    op.quant_b = tmg::QuantDesc{tmg::QuantType::kNone, 0};
                    tmg::MatrixLayout Cdesc = Ddesc;
                    tmg::MatrixLayout Udesc{};

                    int rc = 0;
                    for (int w = 0; w < a.n_warmup; ++w) {
                        rc = gemm.Run(op, 1.0f, dA, Adesc, nullptr, Udesc, dB, Bdesc, nullptr, Vdesc, 0.0f, nullptr, Cdesc, dD, Ddesc, workspace, stream);
                        if (rc != 0) break;
                    }
                    if (rc != 0) {
                        printf("fp16,turbomind,U(-1,1),%d,%d,%d,turbomind_fp16,SKIP,Gemm::Run rc=%d\n", M, N, K, rc);
                    } else {
                        cudaEvent_t e_start, e_stop;
                        cudaEventCreate(&e_start);
                        cudaEventCreate(&e_stop);
                        std::vector<float> samples;
                        for (int t = 0; t < a.n_timed; ++t) {
                            cudaEventRecord(e_start, stream);
                            gemm.Run(op, 1.0f, dA, Adesc, nullptr, Udesc, dB, Bdesc, nullptr, Vdesc, 0.0f, nullptr, Cdesc, dD, Ddesc, workspace, stream);
                            cudaEventRecord(e_stop, stream);
                            cudaEventSynchronize(e_stop);
                            float ms = 0;
                            cudaEventElapsedTime(&ms, e_start, e_stop);
                            samples.push_back(ms);
                        }
                        cudaEventDestroy(e_start);
                        cudaEventDestroy(e_stop);
                        const double ms_med = median_ms(samples);
                        const double tflops = (2.0 * M * N * K) / (ms_med * 1e9);
                        printf("fp16,turbomind,U(-1,1),%d,%d,%d,turbomind_fp16,OK,ms=%.3f,tflops=%.2f\n",
                               M, N, K, ms_med, tflops);
                    }
                    fflush(stdout);
                    cudaFree(dB);
                } else {
                    // Packed path: use pw.weight_data / pw.k_desc / pw.scales_data / pw.q_desc.
                    Bdesc = pw.k_desc;
                    Vdesc = pw.q_desc;
                    Ddesc = tmg::MatrixLayout{turbomind::kHalf, tmg::Order::kRowMajor, M, N, N, 0, 1, nullptr, nullptr};
                    op.quant_b = tmg::QuantDesc{tmg::QuantType::kK, a.group_size};
                    tmg::MatrixLayout Cdesc = Ddesc;
                    tmg::MatrixLayout Udesc{};

                    int rc = 0;
                    for (int w = 0; w < a.n_warmup; ++w) {
                        rc = gemm.Run(op, 1.0f, dA, Adesc, nullptr, Udesc,
                                      pw.weight_data, Bdesc, pw.scales_data, Vdesc,
                                      0.0f, nullptr, Cdesc, dD, Ddesc, workspace, stream);
                        if (rc != 0) break;
                    }
                    if (rc != 0) {
                        printf("%s,turbomind,U(-1,1),%d,%d,%d,turbomind_%s,SKIP,Gemm::Run rc=%d\n",
                               name(c), M, N, K, name(c), rc);
                    } else {
                        cudaEvent_t e_start, e_stop;
                        cudaEventCreate(&e_start);
                        cudaEventCreate(&e_stop);
                        std::vector<float> samples;
                        for (int t = 0; t < a.n_timed; ++t) {
                            cudaEventRecord(e_start, stream);
                            gemm.Run(op, 1.0f, dA, Adesc, nullptr, Udesc,
                                     pw.weight_data, Bdesc, pw.scales_data, Vdesc,
                                     0.0f, nullptr, Cdesc, dD, Ddesc, workspace, stream);
                            cudaEventRecord(e_stop, stream);
                            cudaEventSynchronize(e_stop);
                            float ms = 0;
                            cudaEventElapsedTime(&ms, e_start, e_stop);
                            samples.push_back(ms);
                        }
                        cudaEventDestroy(e_start);
                        cudaEventDestroy(e_stop);
                        const double ms_med = median_ms(samples);
                        const double tflops = (2.0 * M * N * K) / (ms_med * 1e9);
                        printf("%s,turbomind,U(-1,1),%d,%d,%d,turbomind_%s,OK,ms=%.3f,tflops=%.2f\n",
                               name(c), M, N, K, name(c), ms_med, tflops);
                    }
                    fflush(stdout);
                }

                cudaFree(dA); cudaFree(dD);
            }
            free_packed(pw);
        }
    }

    cudaFree(d_barriers);
    cudaFree(d_partials);
    cudaFree(d_flags);
    return 0;
}

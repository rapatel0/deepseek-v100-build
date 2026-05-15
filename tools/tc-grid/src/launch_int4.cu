#include "tc_grid.h"
#include "int4_kernels.cuh"
#include "int4_bitshift_kernels.cuh"
#include "multi_format_opt_kernels.cuh"
#include "v3_kernels.cuh"
#include "v3splitk_kernels.cuh"
#include "v4_kernels.cuh"
#include "v5_kernels.cuh"

#include <vector>
#include <algorithm>
#include <cstdio>

namespace tc_grid {

LaunchResult launch_int4(const LaunchSpec & s, const void * d_W,
                         const float * d_act, float * d_dst, const float * d_ref) {
    LaunchResult R; R.ok = false;
    const int M = s.M, N = s.N, K = s.K;
    const int BM = s.BM, BN = s.BN, BK = s.BK, WARPS = s.warps;
    if (BN / WARPS != 16) { R.note = "BN/WARPS != 16"; return R; }
    if (BK % 16 || BK % QK_INT4) { R.note = "BK constraints"; return R; }
    if (BM < 16) { R.note = "BM < 16"; return R; }
    if (s.frag_n != 1) { R.note = "MF only INT8 LUT"; return R; }
    if (s.version != 0 && s.path != UnpackPath::LUT) { R.note = "opt only LUT path"; return R; }
    if (s.version == 6) { R.note = "v6 only INT8 (A1 retry)"; return R; }
    if (s.version == 7) { R.note = "v7 only INT8 (B3 triple-buffer)"; return R; }
    if (s.version == 8) { R.note = "v8 only INT8 (fixed bank-pad)"; return R; }
    if (s.version == 9) { R.note = "v9 only INT8 (mixed-prec)"; return R; }
    if (s.version == 10) { R.note = "v10 only INT8 (row-major B)"; return R; }
    if (s.version == 20) { R.note = "v10s only INT8 (use version=30 for INT4 SplitK)"; return R; }
    if (N % BN || K % BK) { R.note = "NK %tile != 0"; return R; }

    const int num_tile_x = N / BN;
    const int num_tile_y = (M + BM - 1) / BM;
    const int num_tiles_total = num_tile_x * num_tile_y;
    const int persistent_blocks = 160;
    const dim3 grid_normal((unsigned) num_tile_x, (unsigned) num_tile_y, 1);
    const dim3 grid_persistent((unsigned)((num_tiles_total < persistent_blocks) ? num_tiles_total : persistent_blocks), 1, 1);
    // v3s (SplitK) uses grid.z = split_k factor (passed via s.split_k).
    const int v3s_ks = (s.version == 30) ? std::max(1, s.split_k) : 1;
    const dim3 grid_v3s((unsigned) num_tile_x, (unsigned) num_tile_y, (unsigned) v3s_ks);
    const dim3 grid = (s.version == 5)  ? grid_persistent
                    : (s.version == 30) ? grid_v3s
                    : grid_normal;
    const dim3 block((unsigned)(WARPS * 32));
    const size_t smem_bytes_opt_v1 =
        (size_t)(BM * BK) * sizeof(__half)
      + (size_t)(BK * BN) * sizeof(__half);
    const size_t smem_bytes_opt_v2 = 2 * smem_bytes_opt_v1;
    const int BK_PAD = BK + 8;
    const size_t smem_bytes_opt_v3 = 2 * (
        (size_t)(BM * BK) * sizeof(__half)
      + (size_t)(BK_PAD * BN) * sizeof(__half));
    const size_t smem_bytes_base =
        smem_bytes_opt_v1
      + (size_t)(WARPS * BM * 16) * sizeof(float);
    const size_t smem_bytes_opt_v4 = smem_bytes_opt_v3;
    const size_t smem_bytes_opt_v5 = smem_bytes_opt_v3;
    size_t smem_bytes;
    if      (s.version == 30) smem_bytes = smem_bytes_opt_v3;  // v3s reuses v3 SMEM layout
    else if (s.version == 5) smem_bytes = smem_bytes_opt_v5;
    else if (s.version == 4) smem_bytes = smem_bytes_opt_v4;
    else if (s.version == 3) smem_bytes = smem_bytes_opt_v3;
    else if (s.version == 2) smem_bytes = smem_bytes_opt_v2;
    else if (s.version == 1) smem_bytes = smem_bytes_opt_v1;
    else                     smem_bytes = smem_bytes_base;
    if (smem_bytes > 48 * 1024) { R.note = "smem > 48KiB"; return R; }

    const uint8_t * qs = (const uint8_t *) d_W;
    const __half  * sc = (const __half *)((const char *) d_W + (size_t) N * (K / 2));

#define LAUNCH_V5(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==5 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        kernels::int4_v5::mm_int4_lut_v5<(b_m),(b_n),(b_k),(w),(fm),(fn)>       \
            <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K, num_tile_x, num_tile_y); \
        goto done;                                                              \
    }
    LAUNCH_V5(128, 128, 32, 4, 8, 2)
    LAUNCH_V5(128, 64,  32, 4, 8, 1)
    LAUNCH_V5(64,  128, 32, 4, 4, 2)
    LAUNCH_V5(64,  64,  32, 4, 4, 1)
#undef LAUNCH_V5

#define LAUNCH_V4(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==4 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        kernels::int4_v4::mm_int4_lut_v4<(b_m),(b_n),(b_k),(w),(fm),(fn)>       \
            <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);       \
        goto done;                                                              \
    }
    LAUNCH_V4(128, 128, 32, 4, 8, 2)
    LAUNCH_V4(128, 64,  32, 4, 8, 1)
    LAUNCH_V4(64,  128, 32, 4, 4, 2)
    LAUNCH_V4(64,  64,  32, 4, 4, 1)
#undef LAUNCH_V4

#define LAUNCH_V3_B1(b_m, b_n, b_k, w, fm, fn)                                  \
    if (s.version==3 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        kernels::int4_v3::mm_int4_lut_v3<(b_m),(b_n),(b_k),(w),(fm),(fn)>       \
            <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);       \
        goto done;                                                              \
    }
    LAUNCH_V3_B1(32, 128, 32, 4, 2, 2)
    LAUNCH_V3_B1(32, 64,  32, 4, 2, 1)
    LAUNCH_V3_B1(32, 256, 32, 8, 2, 2)
#undef LAUNCH_V3_B1

#define LAUNCH_V3(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==3 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        kernels::int4_v3::mm_int4_lut_v3<(b_m),(b_n),(b_k),(w),(fm),(fn)>       \
            <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);       \
        goto done;                                                              \
    }
    LAUNCH_V3(128, 128, 32, 4, 8, 2)
    LAUNCH_V3(128, 64,  32, 4, 8, 1)
    LAUNCH_V3(128, 256, 32, 4, 8, 4)
    LAUNCH_V3(128, 256, 32, 8, 8, 2)
    LAUNCH_V3(64,  128, 32, 4, 4, 2)
    LAUNCH_V3(64,  64,  32, 4, 4, 1)
#undef LAUNCH_V3

    // v3s (INT4 SplitK). version=30. grid.z = ks. Pre-zero C when ks > 1.
#define LAUNCH_V3S(b_m, b_n, b_k, w, fm, fn, ks)                                \
    if (s.version==30 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && s.split_k==(ks)) { \
        if ((ks) > 1) cudaMemsetAsync(d_dst, 0, (size_t) M * N * sizeof(float)); \
        kernels::int4_v3s::mm_int4_lut_v3s<(b_m),(b_n),(b_k),(w),(fm),(fn),(ks)> \
            <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);       \
        goto done;                                                              \
    }
    LAUNCH_V3S(128, 128, 32, 4, 8, 2, 1)
    LAUNCH_V3S(128, 128, 32, 4, 8, 2, 2)
    LAUNCH_V3S(128, 128, 32, 4, 8, 2, 4)
    LAUNCH_V3S(128, 128, 32, 4, 8, 2, 8)
    LAUNCH_V3S(128, 64,  32, 4, 8, 1, 2)
    LAUNCH_V3S(128, 64,  32, 4, 8, 1, 4)
    LAUNCH_V3S(64,  128, 32, 4, 4, 2, 2)
    LAUNCH_V3S(64,  128, 32, 4, 4, 2, 4)
    LAUNCH_V3S(64,  128, 32, 4, 4, 2, 8)
    LAUNCH_V3S(64,  64,  32, 4, 4, 1, 2)
    LAUNCH_V3S(64,  64,  32, 4, 4, 1, 4)
    LAUNCH_V3S(64,  64,  32, 4, 4, 1, 8)
#undef LAUNCH_V3S

#define LAUNCH_V2(b_m, b_n, b_k, w, fm, fn)                                     \
    if (s.version==2 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) { \
        kernels::int4_opt::mm_int4_lut_v2<(b_m),(b_n),(b_k),(w),(fm),(fn)>      \
            <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);       \
        goto done;                                                              \
    }
    LAUNCH_V2(128, 128, 32, 4, 8, 2)
    LAUNCH_V2(128, 64,  32, 4, 8, 1)
    LAUNCH_V2(128, 256, 32, 4, 8, 4)
    LAUNCH_V2(128, 256, 32, 8, 8, 2)
    LAUNCH_V2(64,  128, 32, 4, 4, 2)
    LAUNCH_V2(64,  64,  32, 4, 4, 1)
#undef LAUNCH_V2

#define DISPATCH(b_m, b_n, b_k, w)                                              \
    if (BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w)) {                    \
        if (s.path == UnpackPath::LUT) {                                        \
            kernels::int4::mm_int4_lut<(b_m),(b_n),(b_k),(w)>                   \
                <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);   \
        } else {                                                                \
            kernels::int4_b::mm_int4_bitshift<(b_m),(b_n),(b_k),(w)>            \
                <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);   \
        }                                                                       \
        goto done;                                                              \
    }
    DISPATCH(16, 64, 32, 4)
    DISPATCH(16, 64, 64, 4)
    DISPATCH(16, 128, 32, 8)
    DISPATCH(16, 128, 64, 8)
    R.note = "no template instantiation for this tile"; return R;
done:

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { R.note = cudaGetErrorString(err); return R; }
    TCG_CHECK(cudaDeviceSynchronize());

    constexpr int WARM = 2, ITERS = 8;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> times; times.reserve(ITERS);
    for (int i = 0; i < WARM + ITERS; ++i) {
        cudaEventRecord(e0);
#define RERUN_V5(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==5 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) \
            kernels::int4_v5::mm_int4_lut_v5<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K, num_tile_x, num_tile_y);
        RERUN_V5(128, 128, 32, 4, 8, 2)
        RERUN_V5(128, 64,  32, 4, 8, 1)
        RERUN_V5(64,  128, 32, 4, 4, 2)
        RERUN_V5(64,  64,  32, 4, 4, 1)
#undef RERUN_V5
#define RERUN_V4(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==4 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) \
            kernels::int4_v4::mm_int4_lut_v4<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);
        RERUN_V4(128, 128, 32, 4, 8, 2)
        RERUN_V4(128, 64,  32, 4, 8, 1)
        RERUN_V4(64,  128, 32, 4, 4, 2)
        RERUN_V4(64,  64,  32, 4, 4, 1)
#undef RERUN_V4
#define RERUN_V3_B1(b_m, b_n, b_k, w, fm, fn)                                   \
        if (s.version==3 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) \
            kernels::int4_v3::mm_int4_lut_v3<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);
        RERUN_V3_B1(32, 128, 32, 4, 2, 2)
        RERUN_V3_B1(32, 64,  32, 4, 2, 1)
        RERUN_V3_B1(32, 256, 32, 8, 2, 2)
#undef RERUN_V3_B1
#define RERUN_V3(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==3 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) \
            kernels::int4_v3::mm_int4_lut_v3<(b_m),(b_n),(b_k),(w),(fm),(fn)>   \
                <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);
        RERUN_V3(128, 128, 32, 4, 8, 2)
        RERUN_V3(128, 64,  32, 4, 8, 1)
        RERUN_V3(128, 256, 32, 4, 8, 4)
        RERUN_V3(128, 256, 32, 8, 8, 2)
        RERUN_V3(64,  128, 32, 4, 4, 2)
        RERUN_V3(64,  64,  32, 4, 4, 1)
#undef RERUN_V3
#define RERUN_V3S(b_m, b_n, b_k, w, fm, fn, ks)                                 \
        if (s.version==30 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn) && s.split_k==(ks)) { \
            if ((ks) > 1) cudaMemsetAsync(d_dst, 0, (size_t) M * N * sizeof(float)); \
            kernels::int4_v3s::mm_int4_lut_v3s<(b_m),(b_n),(b_k),(w),(fm),(fn),(ks)> \
                <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);   \
        }
        RERUN_V3S(128, 128, 32, 4, 8, 2, 1)
        RERUN_V3S(128, 128, 32, 4, 8, 2, 2)
        RERUN_V3S(128, 128, 32, 4, 8, 2, 4)
        RERUN_V3S(128, 128, 32, 4, 8, 2, 8)
        RERUN_V3S(128, 64,  32, 4, 8, 1, 2)
        RERUN_V3S(128, 64,  32, 4, 8, 1, 4)
        RERUN_V3S(64,  128, 32, 4, 4, 2, 2)
        RERUN_V3S(64,  128, 32, 4, 4, 2, 4)
        RERUN_V3S(64,  128, 32, 4, 4, 2, 8)
        RERUN_V3S(64,  64,  32, 4, 4, 1, 2)
        RERUN_V3S(64,  64,  32, 4, 4, 1, 4)
        RERUN_V3S(64,  64,  32, 4, 4, 1, 8)
#undef RERUN_V3S
#define RERUN_V2(b_m, b_n, b_k, w, fm, fn)                                      \
        if (s.version==2 && BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w) && s.frag_n==(fn)) \
            kernels::int4_opt::mm_int4_lut_v2<(b_m),(b_n),(b_k),(w),(fm),(fn)>  \
                <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K);
        RERUN_V2(128, 128, 32, 4, 8, 2)
        RERUN_V2(128, 64,  32, 4, 8, 1)
        RERUN_V2(128, 256, 32, 4, 8, 4)
        RERUN_V2(128, 256, 32, 8, 8, 2)
        RERUN_V2(64,  128, 32, 4, 4, 2)
        RERUN_V2(64,  64,  32, 4, 4, 1)
#undef RERUN_V2

#define RERUN(b_m, b_n, b_k, w)                                                 \
        if (BM==(b_m) && BN==(b_n) && BK==(b_k) && WARPS==(w)) {                \
            if (s.path == UnpackPath::LUT)                                      \
                kernels::int4::mm_int4_lut<(b_m),(b_n),(b_k),(w)>               \
                    <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K); \
            else                                                                 \
                kernels::int4_b::mm_int4_bitshift<(b_m),(b_n),(b_k),(w)>        \
                    <<<grid, block, smem_bytes>>>(qs, sc, d_act, d_dst, M, N, K); \
        }
        RERUN(16, 64, 32, 4)
        RERUN(16, 64, 64, 4)
        RERUN(16, 128, 32, 8)
        RERUN(16, 128, 64, 8)
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        if (i >= WARM) {
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            times.push_back(ms);
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    double sum = 0, mn = times[0], mx = times[0];
    for (float t : times) { sum += t; mn = std::min<double>(mn, t); mx = std::max<double>(mx, t); }
    R.ms_mean = sum / times.size(); R.ms_min = mn; R.ms_max = mx;
    R.tflops = (2.0 * M * N * K) / (R.ms_mean * 1e-3) / 1e12;
    size_t bytes_w = (size_t)N * (K / 2) + (size_t) N * (K / QK_INT4) * 2;
    size_t bytes_a = (size_t)M * K * 4;
    size_t bytes_c = (size_t)M * N * 4;
    R.gbytes_per_s = (double)(bytes_w + bytes_a + bytes_c) / (R.ms_mean * 1e-3) / 1e9;
    ToleranceStats t = evaluate_tolerance(d_dst, d_ref, (size_t) M * N);
    R.max_abs_err = t.max_abs; R.p99_abs_err = t.p99_abs; R.rel_err = t.rel_err;
    R.ok = true; return R;
}

}  // namespace tc_grid

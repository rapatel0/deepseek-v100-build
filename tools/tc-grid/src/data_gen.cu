// tc-grid data generators + quantizers + reference dequant.
//
// All distributions produce FP32 matrices that downstream quantizers reduce
// to their target storage formats. The choice of distribution is the lever
// for stressing numerical precision regimes.

#include "tc_grid.h"

#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tc_grid {

const char * format_name(Format f) {
    switch (f) {
        case Format::INT8:          return "INT8";
        case Format::INT4:          return "INT4";
        case Format::MXFP4:         return "MXFP4";
        case Format::F8_E4M3_B128:  return "F8_E4M3_B128";
    }
    return "?";
}

const char * unpack_name(UnpackPath p) {
    switch (p) {
        case UnpackPath::LUT:       return "LUT";
        case UnpackPath::BITSHIFT:  return "BITSHIFT";
    }
    return "?";
}

const char * dist_name(DataDist d) {
    switch (d) {
        case DataDist::UNIFORM_SMALL:  return "U(-1,1)";
        case DataDist::UNIFORM_WIDE:   return "U(-8,8)";
        case DataDist::LOGNORMAL:      return "LogN(0,1.5)";
        case DataDist::SPARSE_SPIKES:  return "SparseSpikes";
        case DataDist::ADVERSARIAL:    return "Adversarial";
    }
    return "?";
}

// =========================================================== generators ===

// Per-thread Philox state, each thread covers many elements via strided loop.
constexpr int kGenThreads = 256;
constexpr int kGenBlocks  = 256;
constexpr int kGenTotal   = kGenThreads * kGenBlocks;

__global__ void k_uniform(float * out, size_t n, float lo, float hi, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    for (size_t i = tid; i < n; i += (size_t) kGenTotal) {
        out[i] = lo + (hi - lo) * curand_uniform(&state);
    }
}

__global__ void k_lognormal(float * out, size_t n, float mu, float sigma, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    for (size_t i = tid; i < n; i += (size_t) kGenTotal) {
        float x = curand_normal(&state);
        float sign = curand_uniform(&state) < 0.5f ? -1.0f : 1.0f;
        out[i] = sign * expf(mu + sigma * x);
    }
}

__global__ void k_sparse_spikes(float * out, size_t n, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    for (size_t i = tid; i < n; i += (size_t) kGenTotal) {
        float u = curand_uniform(&state);
        float g = curand_normal(&state);
        if (u < 0.10f) {
            out[i] = (curand_uniform(&state) < 0.5f ? -1.0f : 1.0f) * (1.0f + 5.0f * curand_uniform(&state));
        } else {
            out[i] = 0.1f * g;
        }
    }
}

// Adversarial: build a matrix whose row dot-products with a fixed activation
// vector would exhibit near-total cancellation in FP16. We achieve this by
// alternating sign and modulating magnitude so partial sums oscillate.
__global__ void k_adversarial(float * out, size_t rows, size_t cols, uint64_t seed) {
    size_t r = blockIdx.y;
    size_t c = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    // Pseudo-random but reproducible sign + magnitude.
    uint32_t h = (uint32_t)(r * 2654435761u ^ c * 2246822519u ^ (uint32_t)seed);
    float sign = (h & 1u) ? -1.0f : 1.0f;
    float mag  = 1.0f + 6.0f * ((float)((h >> 1) & 0xFFFFu) / 65535.0f);
    // Oscillate at row-stride to cause cancellation.
    if ((c & 1u) == 0) mag *= 1.001f;  // tiny imbalance so result != exactly 0
    out[r * cols + c] = sign * mag;
}

void gen_matrix_f32(float * d_out, int rows, int cols, DataDist dist, uint64_t seed) {
    const size_t n = (size_t) rows * (size_t) cols;
    if (dist == DataDist::ADVERSARIAL) {
        dim3 block(256);
        dim3 grid((cols + 255) / 256, rows);
        k_adversarial<<<grid, block>>>(d_out, rows, cols, seed);
        TCG_CHECK(cudaGetLastError());
        return;
    }

    const dim3 block(kGenThreads);
    const dim3 grid (kGenBlocks);

    switch (dist) {
        case DataDist::UNIFORM_SMALL:
            k_uniform<<<grid, block>>>(d_out, n, -1.0f, 1.0f, seed);
            break;
        case DataDist::UNIFORM_WIDE:
            k_uniform<<<grid, block>>>(d_out, n, -8.0f, 8.0f, seed);
            break;
        case DataDist::LOGNORMAL:
            k_lognormal<<<grid, block>>>(d_out, n, 0.0f, 1.5f, seed);
            break;
        case DataDist::SPARSE_SPIKES:
            k_sparse_spikes<<<grid, block>>>(d_out, n, seed);
            break;
        default: break;
    }
    TCG_CHECK(cudaGetLastError());
}

// =========================================================== quantizers ===

// Per-row per-block scale: scale = max(|x|) / max_repr  per `blocksize` chunk.
// For INT8: max_repr = 127. For INT4: max_repr = 7.
__device__ __forceinline__ float blk_amax_f32(const float * src, int K, int row, int blk_idx, int blk) {
    float m = 0.0f;
    int base = row * K + blk_idx * blk;
    for (int j = 0; j < blk; ++j) m = fmaxf(m, fabsf(src[base + j]));
    return m;
}

__global__ void k_quantize_int8(const float * src, int K_pad,
                                int8_t * qs, __half * scales,
                                int rows, int K) {
    const int row = blockIdx.y;
    const int blk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = K / QK_INT8;
    if (row >= rows || blk_idx >= blocks_per_row) return;
    float amax = blk_amax_f32(src, K, row, blk_idx, QK_INT8);
    float scale = amax / 127.0f;
    if (scale == 0.0f) scale = 1.0f;
    scales[row * blocks_per_row + blk_idx] = __float2half(scale);
    int base_in  = row * K     + blk_idx * QK_INT8;
    int base_out = row * K_pad + blk_idx * QK_INT8;
    for (int j = 0; j < QK_INT8; ++j) {
        float v = src[base_in + j] / scale;
        int q = __float2int_rn(v);
        q = max(-127, min(127, q));
        qs[base_out + j] = (int8_t) q;
    }
}

// INT4 packed: two 4-bit signed values per byte. Low nibble first, high nibble second.
__global__ void k_quantize_int4(const float * src, int K_pad_bytes,
                                uint8_t * qs, __half * scales,
                                int rows, int K) {
    const int row = blockIdx.y;
    const int blk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = K / QK_INT4;
    if (row >= rows || blk_idx >= blocks_per_row) return;
    float amax = blk_amax_f32(src, K, row, blk_idx, QK_INT4);
    float scale = amax / 7.0f;
    if (scale == 0.0f) scale = 1.0f;
    scales[row * blocks_per_row + blk_idx] = __float2half(scale);
    int base_in  = row * K           + blk_idx * QK_INT4;
    int base_out = row * K_pad_bytes + blk_idx * (QK_INT4 / 2);
    for (int j = 0; j < QK_INT4; j += 2) {
        int q0 = __float2int_rn(src[base_in + j    ] / scale);
        int q1 = __float2int_rn(src[base_in + j + 1] / scale);
        q0 = max(-7, min(7, q0));
        q1 = max(-7, min(7, q1));
        uint8_t lo = (uint8_t)(q0 & 0xF);
        uint8_t hi = (uint8_t)(q1 & 0xF);
        qs[base_out + j / 2] = (uint8_t)((hi << 4) | lo);
    }
}

// MXFP4 quantize: per-32-block uint8 E8M0 shared exponent + nibble for each value.
// MXFP4 codes (signed E2M1 magnitudes scaled by 0.5): +/-{0,0.5,1.0,1.5,2.0,3.0,4.0,6.0} -- but
// the ggml-cuda dequant additionally multiplies the table by 0.5 to match the
// codebase convention. We replicate that: stored value = E8M0_scale * kvalues_mxfp4[nibble] * 0.5.
__device__ __forceinline__ int8_t mxfp4_round_to_nibble(float v_scaled) {
    // signed magnitudes for nibbles 0..15 (matches kvalues_mxfp4 table):
    static constexpr float kvals[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    // Find closest magnitude in {0,0.5,1.0,1.5,2.0,3.0,4.0,6.0}.
    float a = fabsf(v_scaled);
    int   idx = 0;
    float best = fabsf(a - kvals[0]);
    for (int i = 1; i < 8; ++i) {
        float d = fabsf(a - kvals[i]);
        if (d < best) { best = d; idx = i; }
    }
    if (v_scaled < 0.0f) idx |= 8;
    return (int8_t) idx;
}

__device__ __forceinline__ uint8_t f32_to_e8m0(float scale_pow2) {
    // Encode scale ~ 2^(e-127). Round log2(scale) to nearest int.
    if (scale_pow2 <= 0.0f || !isfinite(scale_pow2)) return 127;  // 2^0
    int e = (int) lroundf(log2f(scale_pow2)) + 127;
    if (e < 0)   e = 0;
    if (e > 255) e = 255;
    return (uint8_t) e;
}

__device__ __forceinline__ float e8m0_to_f32(uint8_t e) {
    return ldexpf(1.0f, (int) e - 127);
}

__global__ void k_quantize_mxfp4(const float * src, int K_bytes_pad,
                                 uint8_t * blocks,
                                 int rows, int K) {
    const int row     = blockIdx.y;
    const int blk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = K / QK_MXFP4;
    if (row >= rows || blk_idx >= blocks_per_row) return;
    constexpr int kBlkBytes = 1 + (QK_MXFP4 / 2);  // 1 byte E8M0 + 16 nibble-bytes
    // Pick scale so max(|x|) maps to ~6.0 (the max magnitude in the table)
    // BEFORE the 0.5 codebase-convention factor, i.e. scale * 6 * 0.5 = max(|x|).
    float amax = blk_amax_f32(src, K, row, blk_idx, QK_MXFP4);
    float scale = (amax > 0.0f) ? amax / (6.0f * 0.5f) : 1.0f;
    uint8_t e = f32_to_e8m0(scale);
    float   d = e8m0_to_f32(e);

    int base_in  = row * K           + blk_idx * QK_MXFP4;
    int base_out = row * K_bytes_pad + blk_idx * kBlkBytes;
    blocks[base_out + 0] = e;
    for (int j = 0; j < QK_MXFP4; j += 2) {
        float v0 = src[base_in + j    ] / (d * 0.5f);
        float v1 = src[base_in + j + 1] / (d * 0.5f);
        int8_t n0 = mxfp4_round_to_nibble(v0);
        int8_t n1 = mxfp4_round_to_nibble(v1);
        // Match ggml-cuda dequant layout: byte index = (j/2),
        // low nibble -> j+0 (in_blk < 16), high nibble -> j+16 (in_blk >= 16).
        // For natural sequential order we just pack (j, j+1) into low/high.
        // Both consumers (LUT decoder and PRMT decoder) read the same layout.
        uint8_t b = (uint8_t)((n1 & 0xF) << 4) | (uint8_t)(n0 & 0xF);
        blocks[base_out + 1 + (j / 2)] = b;
    }
}

// FP8 E4M3 conversion (E4M3FN format, finite-only, no Inf, NaN at 0x7F/0xFF).
__device__ __forceinline__ uint8_t f32_to_e4m3fn(float v) {
    if (v == 0.0f) return 0;
    bool neg = v < 0.0f;
    float a = fabsf(v);
    if (a >= 448.0f) a = 448.0f;     // saturate to max finite
    // log2 split: exp/mantissa
    int   e = (int) floorf(log2f(a));
    float m = a / ldexpf(1.0f, e) - 1.0f;  // m in [0, 1)
    int   E = e + 7;                        // bias = 7 for E4M3
    int   M = (int) lroundf(m * 8.0f);      // 3-bit mantissa
    if (M == 8) { M = 0; E += 1; }
    if (E < 0)   { return neg ? 0x80 : 0x00; }
    if (E > 15)  { E = 15; M = 6; }         // saturate just below NaN
    uint8_t r = (uint8_t)((E << 3) | M);
    if (neg) r |= 0x80;
    return r;
}

__device__ __forceinline__ float e4m3fn_to_f32(uint8_t b) {
    bool neg = (b & 0x80) != 0;
    int  E   = (b >> 3) & 0xF;
    int  M   = b & 0x7;
    float v;
    if (E == 0) {
        v = ldexpf((float) M / 8.0f, -6);   // subnormal: 2^-6 * (M/8)
    } else if (E == 15 && M == 7) {
        v = NAN;                             // NaN slot per E4M3FN spec
    } else {
        v = ldexpf(1.0f + (float) M / 8.0f, E - 7);
    }
    return neg ? -v : v;
}

__global__ void k_quantize_f8_e4m3_b128(const float * src, int K_bytes_pad,
                                        uint8_t * blocks,
                                        int rows, int K) {
    const int row     = blockIdx.y;
    const int blk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = K / QK_F8;
    if (row >= rows || blk_idx >= blocks_per_row) return;
    constexpr int kBlkBytes = 1 + QK_F8;  // 1 byte E8M0 + 128 FP8 bytes
    float amax = blk_amax_f32(src, K, row, blk_idx, QK_F8);
    float scale = (amax > 0.0f) ? amax / 448.0f : 1.0f;  // 448 = max finite E4M3
    uint8_t e = f32_to_e8m0(scale);
    float   d = e8m0_to_f32(e);
    int base_in  = row * K           + blk_idx * QK_F8;
    int base_out = row * K_bytes_pad + blk_idx * kBlkBytes;
    blocks[base_out + 0] = e;
    for (int j = 0; j < QK_F8; ++j) {
        float v = src[base_in + j] / d;
        blocks[base_out + 1 + j] = f32_to_e4m3fn(v);
    }
}

void quantize(Format fmt, const float * d_in, void * d_q_blocks,
              int rows, int K, cudaStream_t stream) {
    dim3 block(64);
    switch (fmt) {
        case Format::INT8: {
            int blocks_per_row = K / QK_INT8;
            dim3 grid((blocks_per_row + 63) / 64, rows);
            // Layout: int8 qs[rows*K] then half scales[rows*blocks_per_row]
            int8_t * qs = (int8_t *) d_q_blocks;
            __half * sc = (__half *)((char *) d_q_blocks + (size_t) rows * K);
            k_quantize_int8<<<grid, block, 0, stream>>>(d_in, K, qs, sc, rows, K);
            break;
        }
        case Format::INT4: {
            int blocks_per_row = K / QK_INT4;
            int Kpad_bytes = K / 2;
            dim3 grid((blocks_per_row + 63) / 64, rows);
            uint8_t * qs = (uint8_t *) d_q_blocks;
            __half  * sc = (__half *)((char *) d_q_blocks + (size_t) rows * Kpad_bytes);
            k_quantize_int4<<<grid, block, 0, stream>>>(d_in, Kpad_bytes, qs, sc, rows, K);
            break;
        }
        case Format::MXFP4: {
            int blocks_per_row = K / QK_MXFP4;
            int blkBytes = 1 + (QK_MXFP4 / 2);
            int Kbytes = blocks_per_row * blkBytes;
            dim3 grid((blocks_per_row + 63) / 64, rows);
            uint8_t * b = (uint8_t *) d_q_blocks;
            k_quantize_mxfp4<<<grid, block, 0, stream>>>(d_in, Kbytes, b, rows, K);
            break;
        }
        case Format::F8_E4M3_B128: {
            int blocks_per_row = K / QK_F8;
            int blkBytes = 1 + QK_F8;
            int Kbytes = blocks_per_row * blkBytes;
            dim3 grid((blocks_per_row + 63) / 64, rows);
            uint8_t * b = (uint8_t *) d_q_blocks;
            k_quantize_f8_e4m3_b128<<<grid, block, 0, stream>>>(d_in, Kbytes, b, rows, K);
            break;
        }
    }
    TCG_CHECK(cudaGetLastError());
}

// ======================================================== dequant reference ===

__global__ void k_dequant_int8(const int8_t * qs, const __half * scales,
                               float * out, int rows, int K) {
    int row = blockIdx.y;
    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || j >= K) return;
    int blocks_per_row = K / QK_INT8;
    int blk = j / QK_INT8;
    float s = __half2float(scales[row * blocks_per_row + blk]);
    out[row * K + j] = (float) qs[row * K + j] * s;
}

__global__ void k_dequant_int4(const uint8_t * qs, const __half * scales,
                               float * out, int rows, int K) {
    int row = blockIdx.y;
    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || j >= K) return;
    int blocks_per_row = K / QK_INT4;
    int blk = j / QK_INT4;
    int K_pad_bytes = K / 2;
    int byte_idx = j / 2;
    uint8_t b = qs[row * K_pad_bytes + byte_idx];
    int8_t n;
    if ((j & 1) == 0) n = (int8_t)((b & 0xF) << 4) >> 4;  // sign-extend low nibble
    else              n = (int8_t)(b & 0xF0)        >> 4;
    float s = __half2float(scales[row * blocks_per_row + blk]);
    out[row * K + j] = (float) n * s;
}

__global__ void k_dequant_mxfp4(const uint8_t * blocks, float * out,
                                int rows, int K) {
    int row = blockIdx.y;
    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || j >= K) return;
    static constexpr float kvals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    constexpr int kBlkBytes = 1 + (QK_MXFP4 / 2);
    int blocks_per_row = K / QK_MXFP4;
    int Kbytes = blocks_per_row * kBlkBytes;
    int blk_idx = j / QK_MXFP4;
    int in_blk  = j - blk_idx * QK_MXFP4;
    const uint8_t * b = blocks + (size_t) row * Kbytes + blk_idx * kBlkBytes;
    uint8_t e = b[0];
    uint8_t code = b[1 + (in_blk >> 1)];
    uint8_t nibble = ((in_blk & 1) == 0) ? (code & 0xF) : (code >> 4);
    float v = kvals[nibble & 7] * ((nibble & 8) ? -1.0f : 1.0f);
    float d = e8m0_to_f32(e);
    out[row * K + j] = d * v * 0.5f;
}

__global__ void k_dequant_f8(const uint8_t * blocks, float * out,
                             int rows, int K) {
    int row = blockIdx.y;
    int j   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || j >= K) return;
    constexpr int kBlkBytes = 1 + QK_F8;
    int blocks_per_row = K / QK_F8;
    int Kbytes = blocks_per_row * kBlkBytes;
    int blk_idx = j / QK_F8;
    int in_blk  = j - blk_idx * QK_F8;
    const uint8_t * b = blocks + (size_t) row * Kbytes + blk_idx * kBlkBytes;
    uint8_t e = b[0];
    uint8_t q = b[1 + in_blk];
    float d = e8m0_to_f32(e);
    out[row * K + j] = d * e4m3fn_to_f32(q);
}

void dequant_reference(Format fmt, const void * d_q_blocks, float * d_out,
                       int rows, int K, cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((K + 127) / 128, rows);
    switch (fmt) {
        case Format::INT8: {
            const int8_t * qs = (const int8_t *) d_q_blocks;
            const __half * sc = (const __half *)((const char *) d_q_blocks + (size_t) rows * K);
            k_dequant_int8<<<grid, block, 0, stream>>>(qs, sc, d_out, rows, K);
            break;
        }
        case Format::INT4: {
            int Kpad_bytes = K / 2;
            const uint8_t * qs = (const uint8_t *) d_q_blocks;
            const __half  * sc = (const __half *)((const char *) d_q_blocks + (size_t) rows * Kpad_bytes);
            k_dequant_int4<<<grid, block, 0, stream>>>(qs, sc, d_out, rows, K);
            break;
        }
        case Format::MXFP4:
            k_dequant_mxfp4<<<grid, block, 0, stream>>>((const uint8_t *) d_q_blocks, d_out, rows, K);
            break;
        case Format::F8_E4M3_B128:
            k_dequant_f8<<<grid, block, 0, stream>>>((const uint8_t *) d_q_blocks, d_out, rows, K);
            break;
    }
    TCG_CHECK(cudaGetLastError());
}

}  // namespace tc_grid

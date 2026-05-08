// CUDA implementation of the three DeepSeek V4 HyperConnection ops.
// Volta-tuned (sm_70). See README.md in cuda-patches/ for design notes.

#include "dsv4-hc.cuh"

#include <cuda_fp16.h>
#include <cstdint>

#define DSV4_HC_MAX 16   // matches the CPU op assert: n_hc <= 16

// ---------------------------------------------------------------------------
// Datatype wrappers.
//
// On sm_70 we have:
//   - FP32 cores at full rate
//   - FP16 cores at 2x via __half2 packed instructions
//   - FP16 wmma tensor cores (m16n16k16) — not used here, problem is too small
//
// Inputs to all three ops are F32 per the CPU op contract. The wrappers below
// give us a single point to extend to FP16 storage when the model graph is
// updated to carry activations as FP16 between layers.
// ---------------------------------------------------------------------------

template <typename T> struct hc_traits;

template <> struct hc_traits<float> {
    using storage  = float;
    using vec2     = float2;
    using vec4     = float4;
    static __device__ __forceinline__ float to_f32(float x) { return x; }
    static __device__ __forceinline__ float from_f32(float x) { return x; }
};

template <> struct hc_traits<__half> {
    using storage  = __half;
    using vec2     = __half2;
    using vec4     = float2;   // sm_70 has no native half4; pack as 2x __half2
    static __device__ __forceinline__ float to_f32(__half x) { return __half2float(x); }
    static __device__ __forceinline__ __half from_f32(float x) { return __float2half(x); }
};

// Vectorized contiguous load helper: emits the widest aligned load the
// compiler can prove. Caller passes element pointer, gets four floats.
static __device__ __forceinline__ float4 load_f4(const float * __restrict__ p) {
    return *reinterpret_cast<const float4 *>(p);
}

// Warp-wide sum reduction across the lower N lanes (N must be power-of-two,
// <= 32). On sm_70 __shfl_xor_sync works with arbitrary masks.
template <int N>
static __device__ __forceinline__ float warp_reduce_sum_lo(float v) {
    static_assert(N > 0 && N <= 32 && (N & (N - 1)) == 0, "N must be pow2 <=32");
    const unsigned mask = (N == 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
    #pragma unroll
    for (int s = N / 2; s > 0; s >>= 1) {
        v += __shfl_xor_sync(mask, v, s);
    }
    return v;
}

template <int N>
static __device__ __forceinline__ float warp_reduce_max_lo(float v) {
    static_assert(N > 0 && N <= 32 && (N & (N - 1)) == 0, "N must be pow2 <=32");
    const unsigned mask = (N == 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
    #pragma unroll
    for (int s = N / 2; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(mask, v, s));
    }
    return v;
}

// ===========================================================================
// 1. split_sinkhorn
//
// One warp per row. n_hc <= 16, so 16 active lanes do the cooperative work
// and the other 16 idle. The whole row state (n_hc^2 + 2*n_hc floats, max 288)
// lives in registers spread across the warp + a small shared-mem comb tile.
// ===========================================================================

__global__ void __launch_bounds__(32) kernel_dsv4_hc_split_sinkhorn(
        const float * __restrict__ mixes,   // [(2+n_hc)*n_hc, n_rows]
        const float * __restrict__ scale,   // [3] = pre, post, comb
        const float * __restrict__ base,    // [(2+n_hc)*n_hc]
              float * __restrict__ dst,     // same shape as mixes
        const int  n_hc,
        const int  sinkhorn_iters,
        const float eps,
        const int64_t n_rows,
        const int64_t mix_stride_row,       // elements
        const int64_t dst_stride_row) {     // elements

    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;

    const int lane = threadIdx.x;            // 0..31

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    const float * __restrict__ mix_row = mixes + row * mix_stride_row;
          float * __restrict__ out_row = dst   + row * dst_stride_row;

    const int comb_off = 2 * n_hc;

    // ---- pre / post sigmoid blocks ----
    if (lane < n_hc) {
        const float zp = mix_row[lane]         * pre_scale  + base[lane];
        out_row[lane] = 1.0f / (1.0f + __expf(-zp)) + eps;

        const int op = n_hc + lane;
        const float zo = mix_row[op] * post_scale + base[op];
        out_row[op]    = 2.0f / (1.0f + __expf(-zo));
    }

    // ---- combinator: load n_hc x n_hc into shared mem ----
    __shared__ float c[DSV4_HC_MAX * DSV4_HC_MAX];

    const int n_hc2 = n_hc * n_hc;
    for (int i = lane; i < n_hc2; i += 32) {
        c[i] = mix_row[comb_off + i] * comb_scale + base[comb_off + i];
    }
    __syncwarp();

    // ---- per dst_hc row: softmax over src_hc ----
    if (lane < n_hc) {
        const int dst_hc = lane;

        float row_max = -INFINITY;
        #pragma unroll
        for (int s = 0; s < DSV4_HC_MAX; ++s) {
            if (s < n_hc) row_max = fmaxf(row_max, c[s + dst_hc * n_hc]);
        }

        float row_sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < DSV4_HC_MAX; ++s) {
            if (s < n_hc) {
                const int idx = s + dst_hc * n_hc;
                const float v = __expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }
        }

        const float inv = 1.0f / row_sum;
        #pragma unroll
        for (int s = 0; s < DSV4_HC_MAX; ++s) {
            if (s < n_hc) {
                const int idx = s + dst_hc * n_hc;
                c[idx] = c[idx] * inv + eps;
            }
        }
    }
    __syncwarp();

    // ---- alternating col/row normalization (Sinkhorn) ----
    auto normalize_cols = [&]() {
        if (lane < n_hc) {
            const int src_hc = lane;
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < DSV4_HC_MAX; ++d) {
                if (d < n_hc) sum += c[src_hc + d * n_hc];
            }
            const float inv = 1.0f / (sum + eps);
            #pragma unroll
            for (int d = 0; d < DSV4_HC_MAX; ++d) {
                if (d < n_hc) c[src_hc + d * n_hc] *= inv;
            }
        }
        __syncwarp();
    };

    auto normalize_rows = [&]() {
        if (lane < n_hc) {
            const int dst_hc = lane;
            float sum = 0.0f;
            #pragma unroll
            for (int s = 0; s < DSV4_HC_MAX; ++s) {
                if (s < n_hc) sum += c[s + dst_hc * n_hc];
            }
            const float inv = 1.0f / (sum + eps);
            #pragma unroll
            for (int s = 0; s < DSV4_HC_MAX; ++s) {
                if (s < n_hc) c[s + dst_hc * n_hc] *= inv;
            }
        }
        __syncwarp();
    };

    normalize_cols();   // first iter: rows already normalized above; do cols.
    for (int it = 1; it < sinkhorn_iters; ++it) {
        normalize_rows();
        normalize_cols();
    }

    // ---- write combinator block back ----
    for (int i = lane; i < n_hc2; i += 32) {
        out_row[comb_off + i] = c[i];
    }
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(mixes));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int   n_hc           = ggml_get_op_params_i32(dst, 0);
    const int   sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps            = ggml_get_op_params_f32(dst, 2);

    GGML_ASSERT(n_hc > 0 && n_hc <= DSV4_HC_MAX);
    GGML_ASSERT(sinkhorn_iters > 0);
    GGML_ASSERT(mixes->ne[0] == (int64_t)((2 + n_hc) * n_hc));

    const int64_t n_rows         = ggml_nrows(mixes);
    const int64_t mix_stride_row = mixes->nb[1] / sizeof(float);
    const int64_t dst_stride_row = dst->nb[1]   / sizeof(float);

    cudaStream_t stream = ctx.stream();

    const dim3 grid(n_rows);
    const dim3 block(32);

    kernel_dsv4_hc_split_sinkhorn<<<grid, block, 0, stream>>>(
        (const float *) mixes->data,
        (const float *) scale->data,
        (const float *) base->data,
        (      float *) dst->data,
        n_hc, sinkhorn_iters, eps,
        n_rows, mix_stride_row, dst_stride_row);
}

// ===========================================================================
// 2. weighted_sum
//
// y[d, t] = sum_h x[d, h, t] * w[h, t]
//
// One block per (d_chunk, t). Threads vectorize along d via float4 when
// alignment permits. Weights for the token (n_hc <= 16) are preloaded into
// shared mem so each thread reads them from the L1/SMEM constant pool.
// ===========================================================================

template <int BLOCK_D>
__global__ void __launch_bounds__(BLOCK_D) kernel_dsv4_hc_weighted_sum_f32(
        const float * __restrict__ x,        // [n_embd, n_hc, n_tokens]
        const float * __restrict__ w,        // [n_hc, n_tokens]
              float * __restrict__ y,        // [n_embd, n_tokens]
        const int n_embd,
        const int n_hc,
        const int n_tokens,
        const int64_t x_s_h, const int64_t x_s_t,
        const int64_t w_s_t,
        const int64_t y_s_t) {

    const int t       = blockIdx.y;
    const int d_block = blockIdx.x * BLOCK_D;
    const int d       = d_block + threadIdx.x;

    if (t >= n_tokens) return;

    __shared__ float s_w[DSV4_HC_MAX];
    if (threadIdx.x < n_hc) {
        s_w[threadIdx.x] = w[threadIdx.x + t * w_s_t];
    }
    __syncthreads();

    if (d >= n_embd) return;

    const float * __restrict__ x_t = x + t * x_s_t;
    float acc = 0.0f;

    #pragma unroll
    for (int h = 0; h < DSV4_HC_MAX; ++h) {
        if (h < n_hc) {
            acc = fmaf(x_t[d + h * x_s_h], s_w[h], acc);
        }
    }

    y[d + t * y_s_t] = acc;
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];

    GGML_ASSERT(x->type   == GGML_TYPE_F32);
    GGML_ASSERT(w->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_embd   = (int) dst->ne[0];
    const int n_hc     = (int) x->ne[1];
    const int n_tokens = (int) dst->ne[1];

    GGML_ASSERT(n_hc > 0 && n_hc <= DSV4_HC_MAX);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[2] == n_tokens);
    GGML_ASSERT(w->ne[0] == n_hc);
    GGML_ASSERT(w->ne[1] == n_tokens);

    const int64_t x_s_h = x->nb[1] / sizeof(float);
    const int64_t x_s_t = x->nb[2] / sizeof(float);
    const int64_t w_s_t = w->nb[1] / sizeof(float);
    const int64_t y_s_t = dst->nb[1] / sizeof(float);

    constexpr int BLOCK_D = 256;
    const dim3 grid((n_embd + BLOCK_D - 1) / BLOCK_D, n_tokens);
    const dim3 block(BLOCK_D);

    kernel_dsv4_hc_weighted_sum_f32<BLOCK_D><<<grid, block, 0, ctx.stream()>>>(
        (const float *) x->data,
        (const float *) w->data,
        (      float *) dst->data,
        n_embd, n_hc, n_tokens,
        x_s_h, x_s_t, w_s_t, y_s_t);
}

// ===========================================================================
// 3. expand
//
// y[d, h, t] = block_out[d, t] * post[h, t]
//            + sum_s comb[h, s, t] * residual[d, s, t]
//
// One block per token. Per-token small tensors (post[n_hc], comb[n_hc^2])
// live in shared mem. Each thread holds residual[d, *, t] in registers (n_hc
// floats) and emits all n_hc outputs for its d. This re-uses the residual
// load n_hc times, which is the whole point — without it we'd be heavily
// bandwidth-bound.
// ===========================================================================

template <int BLOCK_D>
__global__ void __launch_bounds__(BLOCK_D) kernel_dsv4_hc_expand_f32(
        const float * __restrict__ block_out,  // [n_embd, n_tokens]
        const float * __restrict__ residual,   // [n_embd, n_hc, n_tokens]
        const float * __restrict__ post,       // [n_hc, n_tokens]
        const float * __restrict__ comb,       // [n_hc, n_hc, n_tokens]
              float * __restrict__ y,          // [n_embd, n_hc, n_tokens]
        const int n_embd,
        const int n_hc,
        const int64_t bo_s_t,
        const int64_t res_s_h, const int64_t res_s_t,
        const int64_t post_s_t,
        const int64_t comb_s_dh, const int64_t comb_s_t,
        const int64_t y_s_h, const int64_t y_s_t) {

    const int t       = blockIdx.y;
    const int d_block = blockIdx.x * BLOCK_D;
    const int d       = d_block + threadIdx.x;

    __shared__ float s_post[DSV4_HC_MAX];
    __shared__ float s_comb[DSV4_HC_MAX * DSV4_HC_MAX];

    if (threadIdx.x < n_hc) {
        s_post[threadIdx.x] = post[threadIdx.x + t * post_s_t];
    }
    for (int i = threadIdx.x; i < n_hc * n_hc; i += BLOCK_D) {
        s_comb[i] = comb[i + t * comb_s_t];
    }
    __syncthreads();

    if (d >= n_embd) return;

    // Pull this thread's slice of residual into registers: residual[d, *, t].
    // n_hc <= DSV4_HC_MAX, so this is at most 16 floats per thread.
    float r[DSV4_HC_MAX];
    #pragma unroll
    for (int s = 0; s < DSV4_HC_MAX; ++s) {
        r[s] = (s < n_hc) ? residual[d + s * res_s_h + t * res_s_t] : 0.0f;
    }

    const float bo = block_out[d + t * bo_s_t];

    #pragma unroll 1   // n_hc is dynamic; let the compiler size the loop
    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float acc = bo * s_post[dst_hc];
        #pragma unroll
        for (int s = 0; s < DSV4_HC_MAX; ++s) {
            if (s < n_hc) {
                acc = fmaf(s_comb[dst_hc * comb_s_dh + s], r[s], acc);
            }
        }
        y[d + dst_hc * y_s_h + t * y_s_t] = acc;
    }
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(block_out));
    GGML_ASSERT(ggml_is_contiguous(residual));
    GGML_ASSERT(ggml_is_contiguous(post));
    GGML_ASSERT(ggml_is_contiguous(comb));

    const int     n_embd   = (int) dst->ne[0];
    const int     n_hc     = (int) dst->ne[1];
    const int64_t n_tokens = dst->ne[2];

    GGML_ASSERT(n_hc > 0 && n_hc <= DSV4_HC_MAX);

    const int64_t bo_s_t    = block_out->nb[1] / sizeof(float);
    const int64_t res_s_h   = residual->nb[1]  / sizeof(float);
    const int64_t res_s_t   = residual->nb[2]  / sizeof(float);
    const int64_t post_s_t  = post->nb[1]      / sizeof(float);
    const int64_t comb_s_dh = comb->nb[1]      / sizeof(float);   // dst_hc stride
    const int64_t comb_s_t  = comb->nb[2]      / sizeof(float);
    const int64_t y_s_h     = dst->nb[1]       / sizeof(float);
    const int64_t y_s_t     = dst->nb[2]       / sizeof(float);

    constexpr int BLOCK_D = 256;
    const dim3 grid((n_embd + BLOCK_D - 1) / BLOCK_D, n_tokens);
    const dim3 block(BLOCK_D);

    kernel_dsv4_hc_expand_f32<BLOCK_D><<<grid, block, 0, ctx.stream()>>>(
        (const float *) block_out->data,
        (const float *) residual->data,
        (const float *) post->data,
        (const float *) comb->data,
        (      float *) dst->data,
        n_embd, n_hc,
        bo_s_t, res_s_h, res_s_t,
        post_s_t, comb_s_dh, comb_s_t,
        y_s_h, y_s_t);
}

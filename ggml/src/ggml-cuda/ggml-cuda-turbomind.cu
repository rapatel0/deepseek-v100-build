// SPRINT-023 P3 — CUDA_TURBOMIND buffer type.
//
// Implements ggml-cuda-turbomind.cuh: a ggml buffer type that pipes weights
// of type GGML_TYPE_MXFP4 / GGML_TYPE_F8_E4M3_B128 through turbomind's pack
// step (libggml-turbomind.so via dlopen) at set_tensor time. Anything else
// behaves like a plain CUDA device buffer.

#include "ggml-cuda-turbomind.cuh"
#include "common.cuh"
#include "convert.cuh"
#include "getrows.cuh"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

// Mirror of the enum in ggml-turbomind-api.h (not included to keep the .so
// optional at build time; the enum values are part of the stable ABI).
enum {
    TM_DTYPE_FP16         = 0,
    TM_DTYPE_F8_E4M3_B128 = 1,
    TM_DTYPE_MXFP4        = 2,
};

// ===========================================================================
// libggml-turbomind.so dlopen surface
// ===========================================================================
namespace {

typedef int  (*pfn_api_version)(void);
typedef int  (*pfn_init)(int);
typedef void (*pfn_shutdown)(void);
typedef int  (*pfn_packed_bytes)(int, int, int, int, size_t *, size_t *);
typedef int  (*pfn_pack_weight)(const void *, int, int, int, int,
                                void *, void *, int *, void *);
typedef int  (*pfn_mul_mat)(const void *, const void *, const void *,
                            int, int, int, int, int, int, void *, void *);
typedef int  (*pfn_mul_mat_grouped)(const void *, const int *, const int *, int,
                                    const void * const *, const void * const *,
                                    int, int, int, int, int, void *, void *);

struct TmLib {
    std::mutex            mtx;
    bool                  tried_load = false;
    bool                  loaded     = false;
    // SPRINT-025 P2: previously a single init_device int that the loader
    // tracked. After the per-device api.cc refactor each cuda_device has
    // its own State entry inside the .so, so the loader only needs to
    // remember which devices it has already called init() for.
    bool                  per_device_inited[32] = {};
    void                * handle      = nullptr;
    pfn_api_version       api_version = nullptr;
    pfn_init              init        = nullptr;
    pfn_shutdown          shutdown    = nullptr;
    pfn_packed_bytes      packed_bytes    = nullptr;
    pfn_pack_weight       pack_weight     = nullptr;
    pfn_mul_mat           mul_mat         = nullptr;
    pfn_mul_mat_grouped   mul_mat_grouped = nullptr;
};

TmLib & g_tm() {
    static TmLib t;
    return t;
}

// SPRINT-025 P2: lazy load + idempotent per-device init. The .so dlopen
// happens once per process. ggml_turbomind_init(device) is called once
// per cuda_device — subsequent dispatches on the same device skip init.
// No shutdown-on-hop; each device retains its own workspace inside the .so.
bool tm_ensure_loaded(int device) {
    TmLib & t = g_tm();
    std::lock_guard<std::mutex> lk(t.mtx);
    if (!t.tried_load) {
        t.tried_load = true;
        t.handle = dlopen("libggml-turbomind.so", RTLD_NOW | RTLD_LOCAL);
        if (!t.handle) {
            GGML_LOG_ERROR("%s: dlopen(libggml-turbomind.so) failed: %s\n", __func__, dlerror());
            return false;
        }
        t.api_version     = (pfn_api_version)     dlsym(t.handle, "ggml_turbomind_api_version");
        t.init            = (pfn_init)            dlsym(t.handle, "ggml_turbomind_init");
        t.shutdown        = (pfn_shutdown)        dlsym(t.handle, "ggml_turbomind_shutdown");
        t.packed_bytes    = (pfn_packed_bytes)    dlsym(t.handle, "ggml_turbomind_packed_bytes");
        t.pack_weight     = (pfn_pack_weight)     dlsym(t.handle, "ggml_turbomind_pack_weight_expert");
        t.mul_mat         = (pfn_mul_mat)         dlsym(t.handle, "ggml_turbomind_mul_mat");
        t.mul_mat_grouped = (pfn_mul_mat_grouped) dlsym(t.handle, "ggml_turbomind_mul_mat_grouped");
        if (!t.init || !t.shutdown || !t.packed_bytes || !t.pack_weight || !t.mul_mat || !t.mul_mat_grouped) {
            GGML_LOG_ERROR("%s: libggml-turbomind.so missing required symbols\n", __func__);
            return false;
        }
        t.loaded = true;
    }
    if (!t.loaded) return false;
    if (device < 0 || device >= 32) {
        GGML_LOG_ERROR("%s: cuda_device=%d out of range [0,32)\n", __func__, device);
        return false;
    }
    if (t.per_device_inited[device]) return true;
    if (t.init(device) != 0) {
        GGML_LOG_ERROR("%s: ggml_turbomind_init(%d) failed\n", __func__, device);
        return false;
    }
    t.per_device_inited[device] = true;
    return true;
}

int ggml_type_to_tm_dtype(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F8_E4M3_B128: return TM_DTYPE_F8_E4M3_B128;
        case GGML_TYPE_MXFP4:        return TM_DTYPE_MXFP4;
        default:                      return -1;
    }
}

int tm_group_size_for(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F8_E4M3_B128: return 128;
        case GGML_TYPE_MXFP4:        return 32;
        default:                      return 0;
    }
}

bool tm_supports_type(ggml_type t) {
    return t == GGML_TYPE_F8_E4M3_B128 || t == GGML_TYPE_MXFP4;
}

}  // namespace

// ===========================================================================
// Buffer + buffer-type contexts
// ===========================================================================
struct ggml_backend_cuda_tm_buffer_context {
    int    device       = -1;
    void * dev_ptr      = nullptr;
    size_t size         = 0;
    // Per-tensor scale allocations and extras — freed together with the
    // backing buffer. Stored as raw pointers (deleted on free_buffer).
    std::vector<void *>                          scale_allocs;
    std::vector<ggml_turbomind_tensor_extra *>   extra_allocs;
};

struct ggml_backend_cuda_tm_buft_context {
    int          device;
    std::string  name;
};

// ===========================================================================
// Buffer interface
// ===========================================================================
static void ggml_backend_cuda_tm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_cuda_tm_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    for (auto * p : ctx->scale_allocs) {
        if (p) cudaFree(p);
    }
    for (auto * e : ctx->extra_allocs) {
        // SPRINT-024 P1.3 — also free per-grouped pointer-table caches.
        if (e->weight_ptrs_dev) cudaFree(e->weight_ptrs_dev);
        if (e->scale_ptrs_dev)  cudaFree(e->scale_ptrs_dev);
        delete e;
    }
    if (ctx->dev_ptr) {
        cudaFree(ctx->dev_ptr);
    }
    delete ctx;
}

static void * ggml_backend_cuda_tm_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_cuda_tm_buffer_context *) buffer->context)->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_tm_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_cuda_tm_buffer_context *) buffer->context;
    if (tensor->view_src) {
        return GGML_STATUS_SUCCESS;
    }
    if (ggml_is_quantized(tensor->type) && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size   = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);
        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_tm_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size)
{
    auto * ctx = (ggml_backend_cuda_tm_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);

    // Fast path: type we don't handle, or partial write (set_tensor is
    // called once for the whole tensor in our case, but be conservative).
    if (!tm_supports_type(tensor->type) || tensor->view_src || offset != 0 || size != ggml_nbytes(tensor)) {
        CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size,
                                   cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    const int N          = (int) tensor->ne[1];   // output channels per expert
    const int K          = (int) tensor->ne[0];   // input dim
    const int n_experts  = (int) tensor->ne[2];   // 1 for non-MoE
    GGML_ASSERT(tensor->ne[3] == 1);

    // Single dense/shared-expert tensors currently have a different numerical
    // contract from llama.cpp's native F8 path: TurboMind consumes FP16 A/D,
    // while the native path uses the ggml quantized-activation kernels. That
    // small per-layer drift breaks DSv4-Flash shared experts. Keep dense
    // tensor packing behind an explicit experiment flag; routed stacked
    // experts remain the production path.
    const char * enable_single = getenv("GGML_TM_ENABLE_SINGLE");
    if (n_experts <= 1 && (!enable_single || enable_single[0] != '1')) {
        CUDA_CHECK(cudaMemcpyAsync(tensor->data, data, size,
                                   cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    if (!tm_ensure_loaded(ctx->device)) {
        // Fall back to plain upload if the .so can't load. Since we leave
        // tensor->extra unset, the normal CUDA path will consume this tensor.
        GGML_LOG_WARN("%s: turbomind .so unavailable, falling back to plain upload\n", __func__);
        CUDA_CHECK(cudaMemcpyAsync(tensor->data, data, size,
                                   cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    const int tm_type    = ggml_type_to_tm_dtype(tensor->type);
    const int group_size = tm_group_size_for(tensor->type);

    size_t weight_bytes = 0, scale_bytes = 0;
    if (g_tm().packed_bytes(tm_type, N, K, group_size, &weight_bytes, &scale_bytes) != 0) {
        GGML_LOG_ERROR("%s: ggml_turbomind_packed_bytes failed (N=%d K=%d gs=%d)\n",
                       __func__, N, K, group_size);
        return;
    }
    const size_t per_expert_stride = (size_t) tensor->nb[2];
    GGML_ASSERT(weight_bytes <= per_expert_stride &&
                "packed weight per expert larger than GGML per-expert stride");

    // Upload entire source to scratch device buffer in one shot.
    void * d_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, size));
    CUDA_CHECK(cudaMemcpyAsync(d_src, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));

    // Single contiguous scales buffer for all experts.
    void * d_scales = nullptr;
    if (scale_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_scales, scale_bytes * n_experts));
    }

    // Per-expert byte stride of the SOURCE GGML data.
    const size_t src_per_expert = ggml_row_size(tensor->type, (int64_t) N * K);

    int k_pack = 0;
    for (int e = 0; e < n_experts; ++e) {
        char * src_e   = (char *) d_src + (size_t) e * src_per_expert;
        char * dst_e   = (char *) tensor->data + (size_t) e * per_expert_stride;
        char * scale_e = d_scales ? ((char *) d_scales + (size_t) e * scale_bytes) : nullptr;
        const int rc = g_tm().pack_weight(src_e, tm_type, N, K, group_size,
                                          dst_e, scale_e, &k_pack,
                                          cudaStreamPerThread);
        if (rc != 0) {
            GGML_LOG_ERROR("%s: pack_weight rc=%d on expert %d/%d (%s)\n",
                           __func__, rc, e, n_experts, tensor->name);
            if (d_scales) cudaFree(d_scales);
            cudaFree(d_src);
            return;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    cudaFree(d_src);

    auto * extra = new ggml_turbomind_tensor_extra;
    extra->k_pack            = k_pack;
    extra->scales_dev        = d_scales;
    extra->scales_bytes      = scale_bytes * n_experts;
    extra->scales_per_expert = scale_bytes;
    extra->group_size        = group_size;
    extra->n_experts         = n_experts;
    extra->weight_ptrs_dev   = nullptr;  // lazily filled on first grouped dispatch
    extra->scale_ptrs_dev    = nullptr;
    extra->packed_b_ld       = 0;
    extra->packed_v_ld       = 0;
    tensor->extra            = extra;

    if (d_scales) ctx->scale_allocs.push_back(d_scales);
    ctx->extra_allocs.push_back(extra);
}

static void ggml_backend_cuda_tm_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size)
{
    auto * ctx = (ggml_backend_cuda_tm_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size,
                               cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_tm_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_cuda_tm_buffer_context *) buffer->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_tm_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_tm_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_tm_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_tm_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_cuda_tm_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_tm_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_cuda_tm_buffer_clear,
    /* .reset           = */ nullptr,
};

// ===========================================================================
// Buffer-type interface
// ===========================================================================
static const char * ggml_backend_cuda_tm_buft_get_name(ggml_backend_buffer_type_t buft) {
    return ((ggml_backend_cuda_tm_buft_context *) buft->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cuda_tm_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size)
{
    auto * bctx = (ggml_backend_cuda_tm_buft_context *) buft->context;
    ggml_cuda_set_device(bctx->device);
    void * dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: cudaMalloc(%.2f MiB) on dev %d failed: %s\n",
                       __func__, size / 1024.0 / 1024.0, bctx->device, cudaGetErrorString(err));
        return nullptr;
    }
    auto * ctx     = new ggml_backend_cuda_tm_buffer_context;
    ctx->device    = bctx->device;
    ctx->dev_ptr   = dev_ptr;
    ctx->size      = size;
    return ggml_backend_buffer_init(buft, ggml_backend_cuda_tm_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_tm_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t ggml_backend_cuda_tm_buft_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor)
{
    GGML_UNUSED(buft);
    // Packed weight is ≤ ggml_nbytes for our supported quant types
    // (FP8: 1 vs 129/128 bytes/elem; MXFP4: 0.5 vs 17/32 bytes/elem).
    size_t size = ggml_nbytes(tensor);
    if (ggml_is_quantized(tensor->type)) {
        const int64_t ne0 = tensor->ne[0];
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }
    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_tm_buft_interface = {
    /* .get_name         = */ ggml_backend_cuda_tm_buft_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_tm_buft_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_tm_buft_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ ggml_backend_cuda_tm_buft_get_alloc_size,
    /* .is_host          = */ nullptr,
};

// ===========================================================================
// Singletons
// ===========================================================================
ggml_backend_buffer_type_t ggml_backend_cuda_turbomind_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lk(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_cuda_tm_buft_context *> ctxs;
    static bool initialized = false;

    if (!initialized) {
        const int n = ggml_backend_cuda_get_device_count();
        bufts.resize(n);
        ctxs.resize(n);
        for (int i = 0; i < n; ++i) {
            ctxs[i] = new ggml_backend_cuda_tm_buft_context{
                /* .device = */ i,
                /* .name   = */ std::string("CUDA_TURBOMIND") + std::to_string(i),
            };
            bufts[i] = ggml_backend_buffer_type{
                /* .iface   = */ ggml_backend_cuda_tm_buft_interface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context = */ ctxs[i],
            };
        }
        initialized = true;
    }
    if (device < 0 || device >= (int) bufts.size()) {
        return nullptr;
    }
    return &bufts[device];
}

ggml_backend_buffer_type_t * ggml_backend_cuda_turbomind_get_extra_bufts(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lk(mutex);

    // One slot per device. Each holds a NULL-terminated list of length 2
    // (our buft + sentinel). The slots are static so the pointer stays
    // valid forever.
    static std::vector<std::vector<ggml_backend_buffer_type_t>> slots;
    if (slots.empty()) {
        const int n = ggml_backend_cuda_get_device_count();
        slots.resize(n);
        for (int i = 0; i < n; ++i) {
            slots[i].push_back(ggml_backend_cuda_turbomind_buffer_type(i));
            slots[i].push_back(nullptr);
        }
    }
    if (device < 0 || device >= (int) slots.size()) {
        return nullptr;
    }
    return slots[device].data();
}

bool ggml_backend_buft_is_cuda_turbomind(ggml_backend_buffer_type_t buft) {
    if (!buft) return false;
    const int n = ggml_backend_cuda_get_device_count();
    for (int i = 0; i < n; ++i) {
        if (buft == ggml_backend_cuda_turbomind_buffer_type(i)) {
            return true;
        }
    }
    return false;
}

// ===========================================================================
// P4 — dispatch helper. Routes a mul_mat through libggml-turbomind.so by:
//   1. FP32 -> FP16 conversion of A (src1)
//   2. ggml_turbomind_mul_mat using the packed weight in tensor->data and
//      the scales from tensor->extra
//   3. FP16 -> FP32 conversion of D into dst->data
//
// Layout (matches our P2.3 contract):
//   K = src0->ne[0]  (input dim)
//   N = src0->ne[1]  (output dim)
//   M = ggml_nrows(src1) (all logical rows after K, flattened)
//   src1: [K, rows...] row-major contiguous FP32
//   dst:  [N, rows...] row-major contiguous FP32
//
// For mul_mat_id, the caller (ggml_cuda_mul_mat_id fallback path) slices
// per-expert and calls ggml_cuda_mul_mat once per expert with the sorted
// tokens. That dispatcher routes to us here.
void ggml_cuda_mul_mat_turbomind(ggml_backend_cuda_context & ctx,
                                 const ggml_tensor * src0,
                                 const ggml_tensor * src1,
                                 ggml_tensor * dst)
{
    GGML_ASSERT(src0->type == GGML_TYPE_F8_E4M3_B128 || src0->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // For MoE per-expert slicing in ggml_cuda_mul_mat_id, src0 is a VIEW of
    // the original tensor with ne[2] = 1 and data offset advanced by
    // i02 * nb02. Resolve the original tensor's extra and compute which
    // expert this is.
    const ggml_tensor * src0_orig = src0->view_src ? src0->view_src : src0;
    GGML_ASSERT(src0_orig->extra != nullptr && "tensor missing turbomind extra (was it on CUDA_TURBOMIND?)");
    auto * extra = (ggml_turbomind_tensor_extra *) src0_orig->extra;

    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int64_t M64 = ggml_nrows(src1);
    GGML_ASSERT(M64 <= INT_MAX);
    const int M = (int) M64;
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[0] == src0->ne[0]);
    GGML_ASSERT(dst->ne[0]  == src0->ne[1]);
    GGML_ASSERT(ggml_nelements(dst) == (int64_t) N * M);

    const ptrdiff_t data_offset = (const char *) src0->data - (const char *) src0_orig->data;
    GGML_ASSERT(data_offset % src0_orig->nb[2] == 0);
    const size_t expert_idx = (size_t) (data_offset / (ptrdiff_t) src0_orig->nb[2]);
    GGML_ASSERT(expert_idx < (size_t) extra->n_experts);

    const int group_size = extra->group_size;
    const int k_pack     = extra->k_pack;
    void *    scales_dev = extra->scales_dev
        ? (void *) ((char *) extra->scales_dev + expert_idx * extra->scales_per_expert)
        : nullptr;

    cudaStream_t stream = ctx.stream();

    // sm70 HMMA.884 kernels are tiled in M. The grouped MoE path naturally
    // runs several routed rows at once, but decode-time shared experts and the
    // LM head can call the single path with M=1. Pad the scratch rows so the
    // kernel cannot write past the end of D when it rounds up internally.
    const int M_padded = (M + 7) & ~7;

    // FP32 -> FP16 for A.
    ggml_cuda_pool_alloc<half> A_fp16(ctx.pool(), (size_t) M_padded * K);
    if (M_padded != M) {
        CUDA_CHECK(cudaMemsetAsync(A_fp16.ptr, 0, (size_t) M_padded * K * sizeof(half), stream));
    }
    auto fp32_to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    fp32_to_fp16(src1->data, A_fp16.ptr, (int64_t) M * K, stream);

    if (!g_tm().mul_mat) {
        GGML_LOG_ERROR("%s: libggml-turbomind.so::ggml_turbomind_mul_mat not loaded\n", __func__);
        return;
    }
    const int tm_type = ggml_type_to_tm_dtype(src0->type);

    // Output FP16 buffer.
    ggml_cuda_pool_alloc<half> D_fp16(ctx.pool(), (size_t) M_padded * N);

    const int rc = g_tm().mul_mat(
        A_fp16.ptr,
        src0->data,
        scales_dev,
        tm_type, M_padded, N, K, group_size, k_pack,
        D_fp16.ptr,
        stream);
    if (rc != 0) {
        GGML_LOG_ERROR("%s: ggml_turbomind_mul_mat rc=%d\n", __func__, rc);
        return;
    }

    // FP16 -> FP32 for D.
    auto fp16_to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
    fp16_to_fp32(D_fp16.ptr, (float *) dst->data, (int64_t) M * N, stream);
}

// ===========================================================================
// SPRINT-024 P1.5 — Grouped MoE dispatch helper.
// Replaces per-expert slicing in ggml_cuda_mul_mat_id when src0 is on a
// CUDA_TURBOMIND buffer. One ggml_turbomind_mul_mat_grouped launch per
// MoE-linear amortizes per-expert launch cost.
// ===========================================================================

// Local mirror of turbomind's StridedPtr (matrix_ptr.h:9-13). MUST be 16 B
// aligned because the kernel reads it via __ldg((const uint4*)...).
struct alignas(16) tm_strided_ptr {
    void * ptr;
    int    stride;
};
static_assert(sizeof(tm_strided_ptr) == 16, "tm_strided_ptr must be 16 bytes");

// Compute the packed leading dimension a turbomind dispatch needs.
// Mirrors the formula in ggml/vendor/turbomind/api.cc (single-expert path
// reconstruction) + memory turbomind_packed_b_ld_factor.md.
//
// For sm70 HMMA_884 OPERAND_B Pack_M=1 (the only sm70 packed config we ship):
//   packed_b_ld = K * 32
// For OPERAND_V on sm70 (post-swap, no Pack_M expansion):
//   packed_v_ld = N
static inline int tm_packed_b_ld(ggml_type, int /*N*/, int K) {
    // Pack_M = 1 in the sm70 registry; col-multiplication factor = 32.
    return K * 32;
}
static inline int tm_packed_v_ld(ggml_type, int N, int /*K*/) {
    return N;
}

// Lazily populate extra->weight_ptrs_dev / scale_ptrs_dev once per tensor.
// Returns true on success, false on alloc / copy error.
static bool tm_ensure_grouped_ptr_tables(
        ggml_turbomind_tensor_extra * extra,
        const ggml_tensor * src0,
        cudaStream_t stream)
{
    if (extra->weight_ptrs_dev) {
        // Already cached. Sanity: scales must be there too (unless type has
        // no scales — currently never the case for our supported types).
        return true;
    }

    const int n_experts  = extra->n_experts;
    const int K          = (int) src0->ne[0];
    const int N          = (int) src0->ne[1];
    const int packed_b   = tm_packed_b_ld(src0->type, N, K);
    const int packed_v   = tm_packed_v_ld(src0->type, N, K);

    std::vector<tm_strided_ptr> h_w(n_experts);
    std::vector<tm_strided_ptr> h_s(n_experts);
    for (int e = 0; e < n_experts; ++e) {
        h_w[e].ptr    = (char *) src0->data + (size_t) e * src0->nb[2];
        h_w[e].stride = packed_b;
        if (extra->scales_dev) {
            h_s[e].ptr    = (char *) extra->scales_dev + (size_t) e * extra->scales_per_expert;
            h_s[e].stride = packed_v;
        } else {
            h_s[e].ptr    = nullptr;
            h_s[e].stride = 0;
        }
    }

    void * d_w = nullptr;
    void * d_s = nullptr;
    if (cudaMalloc(&d_w, sizeof(tm_strided_ptr) * n_experts) != cudaSuccess) {
        return false;
    }
    if (extra->scales_dev) {
        if (cudaMalloc(&d_s, sizeof(tm_strided_ptr) * n_experts) != cudaSuccess) {
            cudaFree(d_w);
            return false;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(d_w, h_w.data(), sizeof(tm_strided_ptr) * n_experts,
                               cudaMemcpyHostToDevice, stream));
    if (d_s) {
        CUDA_CHECK(cudaMemcpyAsync(d_s, h_s.data(), sizeof(tm_strided_ptr) * n_experts,
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    extra->weight_ptrs_dev = d_w;
    extra->scale_ptrs_dev  = d_s;
    extra->packed_b_ld     = packed_b;
    extra->packed_v_ld     = packed_v;
    return true;
}

void ggml_cuda_mul_mat_grouped_turbomind(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor * src0,
                                          const ggml_tensor * src1,
                                          const ggml_tensor * ids,
                                          ggml_tensor * dst)
{
    GGML_ASSERT(src0->type == GGML_TYPE_F8_E4M3_B128 || src0->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->extra != nullptr && "tensor missing turbomind extra");

    auto * extra = (ggml_turbomind_tensor_extra *) src0->extra;
    const int n_experts     = extra->n_experts;
    const int K             = (int) src0->ne[0];
    const int N             = (int) src0->ne[1];
    const int tm_type       = ggml_type_to_tm_dtype(src0->type);
    const int group_size    = extra->group_size;
    const int k_pack        = extra->k_pack;

    cudaStream_t stream = ctx.stream();

    // ---- mirror ggml_cuda_mul_mat_id's routing build (it's the source of
    // truth for the ids layout). We need: ids_to_sorted, ids_from_sorted,
    // tokens_per_expert.
    const int64_t ne12 = src1->ne[2];   // tokens
    const int64_t ne10 = src1->ne[0];   // = K
    const int64_t ne0  = dst->ne[0];    // = N
    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows   = ne12 * n_expert_used;

    GGML_ASSERT(ne10 == K);
    GGML_ASSERT(ne0  == N);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);
    std::vector<int32_t> tokens_per_expert(n_experts, 0);
    std::vector<int32_t> expert_offsets_host(n_experts + 1, 0);

    for (int64_t i02 = 0; i02 < n_experts; ++i02) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data()
                    + i12 * ids->nb[1] + iex * ids->nb[0]);
                GGML_ASSERT(expert_to_use >= 0 && expert_to_use < n_experts);
                if (expert_to_use == (int32_t) i02) {
                    ids_from_sorted_host[i12 * n_expert_used + iex] =
                        (int32_t) ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(
                        (int32_t)(i12 * src1->ne[1] + iex % src1->ne[1]));
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    for (int i = 0; i < n_experts; ++i) {
        expert_offsets_host[i + 1] = expert_offsets_host[i] + tokens_per_expert[i];
    }
    const int total_routes = expert_offsets_host[n_experts];
    GGML_ASSERT((int64_t) ids_to_sorted_host.size() == ne_get_rows);
    GGML_ASSERT(total_routes == (int) ne_get_rows);

    // ---- upload routing metadata to the pool ----
    ggml_cuda_pool_alloc<int32_t> ids_to_sorted_dev(ctx.pool(), total_routes);
    ggml_cuda_pool_alloc<int32_t> ids_from_sorted_dev(ctx.pool(), total_routes);
    ggml_cuda_pool_alloc<int32_t> expert_offsets_dev(ctx.pool(), n_experts + 1);
    CUDA_CHECK(cudaMemcpyAsync(ids_to_sorted_dev.ptr, ids_to_sorted_host.data(),
                               sizeof(int32_t) * total_routes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ids_from_sorted_dev.ptr, ids_from_sorted_host.data(),
                               sizeof(int32_t) * total_routes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(expert_offsets_dev.ptr, expert_offsets_host.data(),
                               sizeof(int32_t) * (n_experts + 1),
                               cudaMemcpyHostToDevice, stream));

    // ---- gather + cast src1 -> A_fp16 in one shot via get_rows_cuda ----
    ggml_cuda_pool_alloc<half> A_fp16(ctx.pool(), (size_t) total_routes * K);
    get_rows_cuda(src1->data, src1->type, ids_to_sorted_dev.ptr,
                  A_fp16.ptr, GGML_TYPE_F16,
                  ne10, src1->nb[1], src1->nb[2], src1->nb[3],
                  total_routes, 1, 1,
                  sizeof(int32_t), (size_t) total_routes * sizeof(int32_t),
                  (size_t) total_routes * sizeof(int32_t),
                  ne10 * sizeof(half), (size_t) total_routes * ne10 * sizeof(half),
                  (size_t) total_routes * ne10 * sizeof(half),
                  stream);
    CUDA_CHECK(cudaGetLastError());

    // ---- ensure cached pointer tables exist ----
    if (!tm_ensure_grouped_ptr_tables(extra, src0, stream)) {
        GGML_LOG_ERROR("%s: tm_ensure_grouped_ptr_tables failed\n", __func__);
        return;
    }

    // ---- output FP16 scratch ----
    ggml_cuda_pool_alloc<half> D_fp16(ctx.pool(), (size_t) total_routes * N);

    if (!g_tm().mul_mat_grouped) {
        GGML_LOG_ERROR("%s: mul_mat_grouped not loaded\n", __func__);
        return;
    }
    int rc = g_tm().mul_mat_grouped(
        A_fp16.ptr,
        /*token_indices=*/nullptr,
        expert_offsets_dev.ptr,
        n_experts,
        (const void * const *) extra->weight_ptrs_dev,
        (const void * const *) extra->scale_ptrs_dev,
        tm_type, N, K, group_size, k_pack,
        D_fp16.ptr,
        stream);
    if (rc != 0) {
        GGML_LOG_ERROR("%s: ggml_turbomind_mul_mat_grouped rc=%d\n", __func__, rc);
        return;
    }

    // ---- D_fp16 [total_routes, N] -> dst FP32 via inverse scatter ----
    // get_rows_cuda copies rows; we need to scatter to dst positions given
    // by ids_from_sorted. dst is [N, n_tokens, n_expert_used] FP32 row-major.
    // The scatter pattern matches what ggml_cuda_mul_mat_id does at line 2795.
    get_rows_cuda(D_fp16.ptr, GGML_TYPE_F16, ids_from_sorted_dev.ptr,
                  dst->data, dst->type,
                  N, N * sizeof(half),
                  (size_t) total_routes * N * sizeof(half),
                  (size_t) total_routes * N * sizeof(half),
                  total_routes, 1, 1,
                  sizeof(int32_t), (size_t) total_routes * sizeof(int32_t),
                  (size_t) total_routes * sizeof(int32_t),
                  dst->nb[1], dst->nb[2], dst->nb[3],
                  stream);
    CUDA_CHECK(cudaGetLastError());

    if (const char * v = getenv("GGML_TM_VERBOSE")) {
        if (v[0] == '1') {
            static int n_calls = 0;
            if (n_calls++ < 4) {
                fprintf(stderr, "[ggml-cuda-turbomind] GROUPED dispatch: "
                        "n_experts=%d n_tokens=%lld n_expert_used=%lld "
                        "total_routes=%d N=%d K=%d gs=%d\n",
                        n_experts, (long long) ne12, (long long) n_expert_used,
                        total_routes, N, K, group_size);
            }
        }
    }
}

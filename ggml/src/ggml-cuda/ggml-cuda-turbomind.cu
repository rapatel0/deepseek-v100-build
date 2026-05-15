// SPRINT-023 P3 — CUDA_TURBOMIND buffer type.
//
// Implements ggml-cuda-turbomind.cuh: a ggml buffer type that pipes weights
// of type GGML_TYPE_MXFP4 / GGML_TYPE_F8_E4M3_B128 through turbomind's pack
// step (libggml-turbomind.so via dlopen) at set_tensor time. Anything else
// behaves like a plain CUDA device buffer.

#include "ggml-cuda-turbomind.cuh"
#include "common.cuh"
#include "convert.cuh"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
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

struct TmLib {
    std::mutex          mtx;
    bool                tried_load = false;
    bool                loaded     = false;
    int                 init_device = -1;
    void              * handle      = nullptr;
    pfn_api_version     api_version = nullptr;
    pfn_init            init        = nullptr;
    pfn_shutdown        shutdown    = nullptr;
    pfn_packed_bytes    packed_bytes = nullptr;
    pfn_pack_weight     pack_weight = nullptr;
    pfn_mul_mat         mul_mat     = nullptr;
};

TmLib & g_tm() {
    static TmLib t;
    return t;
}

// Lazy load. Safe to call from multiple threads; loads at most once.
bool tm_ensure_loaded(int device) {
    TmLib & t = g_tm();
    std::lock_guard<std::mutex> lk(t.mtx);
    if (t.tried_load) {
        if (!t.loaded) return false;
        if (t.init_device != device) {
            // Re-init on different device.
            t.shutdown();
            if (t.init(device) != 0) return false;
            t.init_device = device;
        }
        return true;
    }
    t.tried_load = true;
    t.handle = dlopen("libggml-turbomind.so", RTLD_NOW | RTLD_LOCAL);
    if (!t.handle) {
        GGML_LOG_ERROR("%s: dlopen(libggml-turbomind.so) failed: %s\n", __func__, dlerror());
        return false;
    }
    t.api_version  = (pfn_api_version)  dlsym(t.handle, "ggml_turbomind_api_version");
    t.init         = (pfn_init)         dlsym(t.handle, "ggml_turbomind_init");
    t.shutdown     = (pfn_shutdown)     dlsym(t.handle, "ggml_turbomind_shutdown");
    t.packed_bytes = (pfn_packed_bytes) dlsym(t.handle, "ggml_turbomind_packed_bytes");
    t.pack_weight  = (pfn_pack_weight)  dlsym(t.handle, "ggml_turbomind_pack_weight_expert");
    t.mul_mat      = (pfn_mul_mat)      dlsym(t.handle, "ggml_turbomind_mul_mat");
    if (!t.init || !t.shutdown || !t.packed_bytes || !t.pack_weight || !t.mul_mat) {
        GGML_LOG_ERROR("%s: libggml-turbomind.so missing required symbols\n", __func__);
        return false;
    }
    if (t.init(device) != 0) {
        GGML_LOG_ERROR("%s: ggml_turbomind_init(%d) failed\n", __func__, device);
        return false;
    }
    t.init_device = device;
    t.loaded = true;
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

    if (!tm_ensure_loaded(ctx->device)) {
        // Fall back to plain upload if the .so can't load — kernel won't run
        // through us, but at least the tensor is on the device.
        GGML_LOG_WARN("%s: turbomind .so unavailable, falling back to plain upload\n", __func__);
        CUDA_CHECK(cudaMemcpyAsync(tensor->data, data, size,
                                   cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    const int tm_type    = ggml_type_to_tm_dtype(tensor->type);
    const int group_size = tm_group_size_for(tensor->type);
    const int N          = (int) tensor->ne[1];   // output channels per expert
    const int K          = (int) tensor->ne[0];   // input dim
    const int n_experts  = (int) tensor->ne[2];   // 1 for non-MoE
    GGML_ASSERT(tensor->ne[3] == 1);

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
//   M = src1->ne[1]  (number of tokens)
//   src1: [K, M] row-major contiguous FP32
//   dst:  [N, M] row-major contiguous FP32
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
    const int M = (int) src1->ne[1];
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[0] == src0->ne[0]);
    GGML_ASSERT(dst->ne[0]  == src0->ne[1]);
    GGML_ASSERT(dst->ne[1]  == src1->ne[1]);

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

    // FP32 -> FP16 for A.
    ggml_cuda_pool_alloc<half> A_fp16(ctx.pool(), (size_t) M * K);
    auto fp32_to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    fp32_to_fp16(src1->data, A_fp16.ptr, (int64_t) M * K, stream);

    // Output FP16 buffer.
    ggml_cuda_pool_alloc<half> D_fp16(ctx.pool(), (size_t) M * N);

    if (!g_tm().mul_mat) {
        GGML_LOG_ERROR("%s: libggml-turbomind.so::ggml_turbomind_mul_mat not loaded\n", __func__);
        return;
    }
    const int tm_type = ggml_type_to_tm_dtype(src0->type);
    const int rc = g_tm().mul_mat(
        A_fp16.ptr,
        src0->data,
        scales_dev,
        tm_type, M, N, K, group_size, k_pack,
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

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

constexpr int BLOCK = 256;

template <typename T>
__global__ void concat_dim0(const T * x, const T * y, T * dst, int ne0, int ne00, int ne1, int ne2) {
    const int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) return;
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z;
    const int dst_off = nidx + i1 * ne0 + i2 * ne0 * ne1;
    if (nidx < ne00) {
        dst[dst_off] = x[nidx + i1 * ne00 + i2 * ne00 * ne1];
    } else {
        const int ne10 = ne0 - ne00;
        dst[dst_off] = y[(nidx - ne00) + i1 * ne10 + i2 * ne10 * ne1];
    }
}

template <typename T>
__global__ void concat_dim1(const T * x, const T * y, T * dst, int ne0, int ne01, int ne1, int ne2) {
    const int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    if (i0 >= ne0) return;
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z;
    const int dst_off = i0 + i1 * ne0 + i2 * ne0 * ne1;
    if (i1 < ne01) {
        dst[dst_off] = x[i0 + i1 * ne0 + i2 * ne0 * ne01];
    } else {
        dst[dst_off] = y[i0 + (i1 - ne01) * ne0 + i2 * ne0 * (ne1 - ne01)];
    }
}

template <typename T>
__global__ void concat_dim2(const T * x, const T * y, T * dst, int ne0, int ne1, int ne02, int ne2) {
    const int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    if (i0 >= ne0) return;
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z;
    const int dst_off = i0 + i1 * ne0 + i2 * ne0 * ne1;
    if (i2 < ne02) {
        dst[dst_off] = x[i0 + i1 * ne0 + i2 * ne0 * ne1];
    } else {
        dst[dst_off] = y[i0 + i1 * ne0 + (i2 - ne02) * ne0 * ne1];
    }
}

template <typename T>
void cpu_concat(
        const std::vector<T> & a,
        const std::vector<T> & b,
        std::vector<T> & out,
        int dim, int ne0, int ne1, int ne2, int ne3,
        int a0, int a1, int a2, int a3) {
    for (int i3 = 0; i3 < ne3; ++i3) {
        for (int i2 = 0; i2 < ne2; ++i2) {
            for (int i1 = 0; i1 < ne1; ++i1) {
                for (int i0 = 0; i0 < ne0; ++i0) {
                    const int out_off = i0 + ne0 * (i1 + ne1 * (i2 + ne2 * i3));
                    bool from_a = i0 < a0 && i1 < a1 && i2 < a2 && i3 < a3;
                    int src_off;
                    if (from_a) {
                        src_off = i0 + a0 * (i1 + a1 * (i2 + a2 * i3));
                        out[out_off] = a[src_off];
                    } else {
                        const int b0 = dim == 0 ? ne0 - a0 : ne0;
                        const int b1 = dim == 1 ? ne1 - a1 : ne1;
                        const int b2 = dim == 2 ? ne2 - a2 : ne2;
                        const int bi0 = dim == 0 ? i0 - a0 : i0;
                        const int bi1 = dim == 1 ? i1 - a1 : i1;
                        const int bi2 = dim == 2 ? i2 - a2 : i2;
                        const int bi3 = dim == 3 ? i3 - a3 : i3;
                        src_off = bi0 + b0 * (bi1 + b1 * (bi2 + b2 * bi3));
                        out[out_off] = b[src_off];
                    }
                }
            }
        }
    }
}

template <typename T>
void run_case(const char * name, int dim) {
    int a0 = 5, a1 = 3, a2 = 2, a3 = 2;
    int b0 = a0, b1 = a1, b2 = a2, b3 = a3;
    if (dim == 0) b0 = 4;
    if (dim == 1) b1 = 4;
    if (dim == 2) b2 = 3;
    if (dim == 3) b3 = 3;

    const int ne0 = a0 + (dim == 0 ? b0 : 0);
    const int ne1 = a1 + (dim == 1 ? b1 : 0);
    const int ne2 = a2 + (dim == 2 ? b2 : 0);
    const int ne3 = a3 + (dim == 3 ? b3 : 0);
    if (dim != 0) b0 = ne0;
    if (dim != 1) b1 = ne1;
    if (dim != 2) b2 = ne2;

    std::vector<T> a(a0 * a1 * a2 * a3);
    std::vector<T> b(b0 * b1 * b2 * b3);
    std::vector<T> expected(ne0 * ne1 * ne2 * ne3);
    std::vector<T> actual(expected.size());

    for (size_t i = 0; i < a.size(); ++i) a[i] = (T) (0x1000u + i * 17u);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (T) (0x8000u + i * 19u);
    cpu_concat(a, b, expected, dim, ne0, ne1, ne2, ne3, a0, a1, a2, a3);

    T * da = nullptr;
    T * db = nullptr;
    T * dout = nullptr;
    CHECK_CUDA(cudaMalloc(&da, a.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&db, b.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&dout, actual.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));

    if (dim == 3) {
        CHECK_CUDA(cudaMemcpy(dout, da, a.size() * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dout + a.size(), db, b.size() * sizeof(T), cudaMemcpyDeviceToDevice));
    } else {
        const dim3 grid((ne0 + BLOCK - 1) / BLOCK, ne1, ne2);
        for (int i3 = 0; i3 < ne3; ++i3) {
            T * od = dout + i3 * ne0 * ne1 * ne2;
            if (dim == 0) {
                concat_dim0<<<grid, BLOCK>>>(da + i3 * a0 * a1 * a2, db + i3 * b0 * b1 * b2, od, ne0, a0, ne1, ne2);
            } else if (dim == 1) {
                concat_dim1<<<grid, BLOCK>>>(da + i3 * a0 * a1 * a2, db + i3 * b0 * b1 * b2, od, ne0, a1, ne1, ne2);
            } else {
                concat_dim2<<<grid, BLOCK>>>(da + i3 * a0 * a1 * a2, db + i3 * b0 * b1 * b2, od, ne0, ne1, a2, ne2);
            }
        }
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaMemcpy(actual.data(), dout, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(da));
    CHECK_CUDA(cudaFree(db));
    CHECK_CUDA(cudaFree(dout));

    if (actual != expected) {
        std::fprintf(stderr, "concat bitcompare failed for %s dim%d\n", name, dim);
        std::exit(2);
    }
    std::printf("concat bitcompare passed for %s dim%d\n", name, dim);
}

int main() {
    for (int dim = 0; dim < 4; ++dim) {
        run_case<uint32_t>("f32", dim);
        run_case<uint16_t>("f16", dim);
        run_case<uint16_t>("bf16", dim);
    }
    return 0;
}

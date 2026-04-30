#include "internal_header.h"
#include <cuda_runtime.h>
#include <cassert>

void soft_cuda_memset_zero(void *ptr, size_t bytes) {
    if (ptr) {
        cudaError_t err = cudaMemsetAsync(ptr, 0, bytes);
        if (err != cudaSuccess)
            debug("cudaMemsetAsync failed: %s\n", cudaGetErrorString(err));
    }
}

void soft_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void soft_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

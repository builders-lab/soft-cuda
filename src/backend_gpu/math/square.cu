#include "internal_header.h"
#include <cuda_runtime.h>

__global__
static void square_kernel(const float *a, float *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        out[i] = a[i] * a[i];
}

bool tensor_square_op_cuda(tensor_t *t, float *d_a, float *d_res) {
    int block = 256;
    int grid  = ((int)t->nvalues + block - 1) / block;
    square_kernel<<<grid, block>>>(d_a, d_res, t->nvalues);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        debug("CUDA Square Kernel Failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

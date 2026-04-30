#include "internal_header.h"
#include <cuda_runtime.h>

__global__
static void scalar_mul_kernel(const float *a, float *out,
                               float scalar, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        out[i] = a[i] * scalar;
}

bool tensor_scalar_mul_op_cuda(tensor_t *t, float *d_a,
                                [[maybe_unused]] float *d_b,
                                float *d_res) {
    float s = ((float *)t->b->data)[0];
    int block = 256;
    int grid  = ((int)t->nvalues + block - 1) / block;
    scalar_mul_kernel<<<grid, block>>>(d_a, d_res, s, t->nvalues);
  cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        debug("CUDA ScalarMul Kernel Failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

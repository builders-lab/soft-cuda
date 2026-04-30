#include "internal_header.h"
#include <cuda_runtime.h>

__global__
static void broadcast_add_kernel(const float *a, const float *bias,
                                  float *out,
                                  uint32_t rows, uint32_t cols) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = rows * cols;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        uint32_t col = idx % cols;
        out[idx] = a[idx] + bias[col];
    }
}

bool tensor_broadcast_add_op_cuda(tensor_t *t, float *d_a, float *d_b,
                                   float *d_res) {
    uint32_t rows = t->dims[0];
    uint32_t cols = t->dims[1];
    uint32_t total = rows * cols;
    int block = 256;
    int grid  = ((int)total + block - 1) / block;
    broadcast_add_kernel<<<grid, block>>>(d_a, d_b, d_res, rows, cols);
   cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        debug("CUDA BroadcastAdd Kernel Failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

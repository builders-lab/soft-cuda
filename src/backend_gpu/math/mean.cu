#include "internal_header.h"
#include <cuda_runtime.h>

#define MEAN_BLOCK 256

__global__
static void reduce_sum_k(const float *in, float *partial, uint32_t n) {
    extern __shared__ float smem[];
    uint32_t tid = threadIdx.x;
    uint32_t i   = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = smem[0];
}


__global__
static void divide_scalar_k(float *val, uint32_t n) {
    /* Single-thread, runs with <<<1,1>>> */
    val[0] /= (float)n;
}


bool tensor_mean_op_cuda(tensor_t *t, float *d_a, float *d_res) {
    uint32_t n     = t->nvalues;
    int block      = MEAN_BLOCK;
    int grid       = ((int)n + block - 1) / block;

    // Allocate partial-sum buffer 
    float *d_partial = nullptr;
    cudaError_t alloc_err = cudaMalloc(&d_partial, (size_t)grid * sizeof(float));
    if (alloc_err != cudaSuccess) {
        debug("CUDA Mean: failed to alloc partial buffer: %s\n",
              cudaGetErrorString(alloc_err));
        return false;
    }

    /* Pass 1 */
    reduce_sum_k<<<grid, block, (size_t)block * sizeof(float)>>>(d_a, d_partial, n);

    /* Pass 2 — reduce the partial sums */
    int grid2 = ((int)grid + block - 1) / block;
    float *d_partial2 = nullptr;
    cudaMalloc(&d_partial2, (size_t)grid2 * sizeof(float));
    reduce_sum_k<<<grid2, block, (size_t)block * sizeof(float)>>>(
        d_partial, d_partial2, (uint32_t)grid);
    // If there are still more elements, keep reducing 
    while (grid2 > 1) {
        int grid3 = (grid2 + block - 1) / block;
        float *d_tmp = nullptr;
        cudaMalloc(&d_tmp, (size_t)grid3 * sizeof(float));
        reduce_sum_k<<<grid3, block, (size_t)block * sizeof(float)>>>(
            d_partial2, d_tmp, (uint32_t)grid2);
        cudaFree(d_partial2);
        d_partial2 = d_tmp;
        grid2 = grid3;
    }

    // Copy scalar result to d_res[0] and divide 
    cudaMemcpy(d_res, d_partial2, sizeof(float), cudaMemcpyDeviceToDevice);
    divide_scalar_k<<<1, 1>>>(d_res, n);

    cudaFree(d_partial);
    cudaFree(d_partial2);

cudaError_t err = cudaSuccess;

#ifdef SC_DEBUG
    err = cudaDeviceSynchronize();
#else
    err = cudaGetLastError();
#endif
    if (err != cudaSuccess) {
        debug("CUDA Mean Kernel Failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}


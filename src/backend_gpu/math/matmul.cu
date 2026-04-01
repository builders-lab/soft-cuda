#include <stdio.h>
#include "internal_header.h"
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__
void sgemm_naive(float *A, float *B, float *C, uint32_t M, uint32_t K, uint32_t N) {
    
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int l = 0; l < K; l++) {
            sum += A[row * K + l] * B[l*N+col];
        }
        C[row * N + col] = sum;
    }
}

bool tensor_mul_op_cuda(tensor_t *t,float *d_a, float *d_b, float *d_res) {
    uint32_t M = t->a->dims[0];
    uint32_t K = t->a->dims[1];
    uint32_t N = t->b->dims[1];
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemm_naive<<<gridDim, blockDim>>>(d_a, d_b, d_res,M, K, N);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
          debug("CUDA Add Kernel Failed: %s\n", cudaGetErrorString(err));
          return false;
    }

    return true;
}

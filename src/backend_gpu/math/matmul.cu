#include <stdio.h>
#include "internal_header.h"
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include "../kernels/sgemm_double_buffer.cuh"
#include <cublas_v2.h>

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

cublasHandle_t handle = NULL;
bool tensor_mul_op_cuda(tensor_t *t, float *d_a, float *d_b, float *d_res) {
    uint32_t M = t->a->dims[0];
    uint32_t K = t->a->dims[1];
    uint32_t N = t->b->dims[1];
    
    float alpha = 1.0f, beta = 0.0f;
    if (handle == NULL) cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_res, N);

    // if (M % 128 == 0 && N % 128 == 0 && K % 16 == 0) {
    //     dim3 block(256);
    //     dim3 grid(CEIL_DIV(N, 128), CEIL_DIV(M, 128));
    //     runSgemmDoubleBuffering2<128, 128, 16, 64, 64, 4, 8, 4, 256>
    //         <<<grid, block>>>(M, N, K, 1.0f, d_a, d_b, 0.0f, d_res);
    // } else {
    //     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    //     dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
    //     sgemm_naive<<<gridDim, blockDim>>>(d_a, d_b, d_res, M, K, N);
    // }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        debug("CUDA Matmul Kernel Failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

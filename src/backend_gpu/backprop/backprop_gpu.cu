#include "internal_header.h"
#include <cuda_runtime.h>
#include <cassert>



__global__
static void grad_add_k(const float *g_out,
                        float *g_a, float *g_b,
                        uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        if (g_a) g_a[i] += g_out[i];
        if (g_b) g_b[i] += g_out[i];
    }
}

/* -------------------------------------------------------------------------
 * SUB backward
 *   dL/dA[i] += dL/dOut[i]
 *   dL/dB[i] -= dL/dOut[i]
 * ---------------------------------------------------------------------- */
__global__
static void grad_sub_k(const float *g_out,
                        float *g_a, float *g_b,
                        uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        if (g_a) g_a[i] += g_out[i];
        if (g_b) g_b[i] -= g_out[i];
    }
}

/* -------------------------------------------------------------------------
 * RELU backward
 *   dL/dA[i] += dL/dOut[i]  if A[i] > 0, else 0
 *   d_a_fwd : the forward input values (needed for the mask)
 * ---------------------------------------------------------------------- */
__global__
static void grad_relu_k(const float *g_out,
                         const float *a_fwd,   /* forward input */
                         float *g_a,
                         uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        if (g_a) g_a[i] += (a_fwd[i] > 0.f) ? g_out[i] : 0.f;
}

/* -------------------------------------------------------------------------
 * SQUARE backward
 *   dL/dA[i] += 2 * A[i] * dL/dOut[i]
 * ---------------------------------------------------------------------- */
__global__
static void grad_square_k(const float *g_out,
                            const float *a_fwd,
                            float *g_a,
                            uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        if (g_a) g_a[i] += 2.f * a_fwd[i] * g_out[i];
}

/* -------------------------------------------------------------------------
 * MEAN backward
 *   dL/dA[i] += dL/dOut[0] / N     (uniform distribution)
 *   g_out has exactly 1 element (it's a scalar).
 * ---------------------------------------------------------------------- */
__global__
static void grad_mean_k(const float *g_out_scalar,
                         float *g_a,
                         uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    float scale = g_out_scalar[0] / (float)n;
    for (; i < n; i += stride)
        if (g_a) g_a[i] += scale;
}

/* -------------------------------------------------------------------------
 * MUL_SCALAR backward
 *   dL/dA[i] += dL/dOut[i] * scalar
 *   scalar is a CPU-side float (the scalar tensor b stays on host)
 * ---------------------------------------------------------------------- */
__global__
static void grad_scalar_mul_k(const float *g_out,
                                float *g_a,
                                float scalar,
                                uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        if (g_a) g_a[i] += g_out[i] * scalar;
}

/* -------------------------------------------------------------------------
 * BROADCAST_ADD backward
 *   dL/dA[i,j] += dL/dOut[i,j]                   (full pass-through)
 *   dL/dBias[j] += sum_i dL/dOut[i,j]             (column-wise sum)
 *
 *   We run two separate kernels:
 *     1. Identity for g_a (same as ADD kernel)
 *     2. Parallel column-reduce for g_bias
 * ---------------------------------------------------------------------- */
__global__
static void grad_broadcast_add_a_k(const float *g_out,
                                     float *g_a,
                                     uint32_t total) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; i < total; i += stride)
        if (g_a) g_a[i] += g_out[i];
}

/* Each thread handles one column of the bias gradient.
 * Accumulates over all rows. */
__global__
static void grad_broadcast_add_bias_k(const float *g_out,
                                        float *g_b,
                                        uint32_t rows,
                                        uint32_t cols) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float acc = 0.f;
    for (uint32_t row = 0; row < rows; row++)
        acc += g_out[row * cols + col];
    if (g_b) g_b[col] += acc;
}

/* -------------------------------------------------------------------------
 * NAIVE_MATRIX_MUL backward
 *
 *   C = A × B    (M × K) × (K × N) → (M × N)
 *
 *   dL/dA[i,k] += sum_j  dL/dC[i,j] * B[k,j]   ≡  g_C × B^T
 *   dL/dB[k,j] += sum_i  A[i,k]     * dL/dC[i,j] ≡  A^T × g_C
 *
 * We implement two separate SGEMM kernels using BLOCK_SIZE tiles.
 * ---------------------------------------------------------------------- */
#define MATMUL_BLOCK 32

/* dA = g_C × B^T   shape: (M, N) × (N, K)^T → (M, K)
 * Equivalently: dA[i,k] = sum_j g_C[i,j] * B[k,j] */
__global__
static void grad_matmul_dA_k(const float *g_C,  // (M, N) 
                               const float *B,    // (K, N) 
                               float *g_A,        // (M, K) 
                               uint32_t M, uint32_t K, uint32_t N) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y; // i ∈ [0,M) 
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; // k ∈ [0,K) 
    if (row >= M || col >= K) return;
    float acc = 0.f;
    for (uint32_t j = 0; j < N; j++)
        acc += g_C[row * N + j] * B[col * N + j]; /* B[k,j] */
    if (g_A) g_A[row * K + col] += acc;
}

// dB = A^T × g_C   shape: (K, M)^T × (M, N) → (K, N)
//  Equivalently: dB[k,j] = sum_i A[i,k] * g_C[i,j] 
__global__
static void grad_matmul_dB_k(const float *A,    // (M, K) 
                               const float *g_C, // (M, N) 
                               float *g_B,       // (K, N) 
                               uint32_t M, uint32_t K, uint32_t N) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;  //k ∈ [0,K) 
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;//  j ∈ [0,N) 
    if (row >= K || col >= N) return;
    float acc = 0.f;
    for (uint32_t i = 0; i < M; i++)
        acc += A[i * K + row] * g_C[i * N + col];
    if (g_B) g_B[row * N + col] += acc;
}

/* =========================================================================
 * For each node, we need:
 *   g_out       : this node's grad buffer (device)     = node->device_ptr_grad
 *   d_a_fwd     : parent A forward data (device)       = parent_a_exec->device_ptr
 *   d_b_fwd     : parent B forward data (device)       = parent_b_exec->device_ptr
 *   g_a         : parent A grad buffer  (device)       = parent_a_exec->device_ptr_grad
 *   g_b         : parent B grad buffer  (device)       = parent_b_exec->device_ptr_grad
 *
* We need parent execution nodes during GPU backprop, but we don't want to modify the struct. Since the caller already has the whole graph, we just pass parent nodes as arguments instead.
 * ========================================================================= */

bool backprop_gpu_dispatch(execution_node_t *node,
                            execution_node_t *parent_a,
                            execution_node_t *parent_b) {
    assert(node != NULL);
    assert(node->t != NULL);
    assert(node->device_ptr_grad != NULL);

    float *g_out = (float *)node->device_ptr_grad;
    uint32_t n   = node->t->nvalues;
    int block    = 256;
    int grid     = ((int)n + block - 1) / block;

    float *d_a_fwd = (parent_a) ? (float *)parent_a->device_ptr      : nullptr;
    float *d_b_fwd = (parent_b) ? (float *)parent_b->device_ptr      : nullptr;

    float *g_a     = (parent_a && parent_a->device_ptr_grad)
                         ? (float *)parent_a->device_ptr_grad : nullptr;
    float *g_b     = (parent_b && parent_b->device_ptr_grad)
                         ? (float *)parent_b->device_ptr_grad : nullptr;

    cudaError_t err = cudaSuccess;

    switch (node->t->op) {
    case tensor_op_t::ADD:
        grad_add_k<<<grid, block>>>(g_out, g_a, g_b, n);
        break;

    case tensor_op_t::SUB:
        grad_sub_k<<<grid, block>>>(g_out, g_a, g_b, n);
        break;

    case tensor_op_t::RELU:
        if (d_a_fwd == nullptr) return false;  
        grad_relu_k<<<grid, block>>>(g_out, d_a_fwd, g_a, n);
        break;

    case tensor_op_t::SQUARE:
        if (d_a_fwd == nullptr) return false;
        grad_square_k<<<grid, block>>>(g_out, d_a_fwd, g_a, n);
        break;

    case tensor_op_t::MEAN: {
        uint32_t na = node->t->a->nvalues;
        int ga = ((int)na + block - 1) / block;
        grad_mean_k<<<ga, block>>>(g_out, g_a, na);
        break;
    }

    case tensor_op_t::MUL_SCALAR: {
        float scalar = ((float *)node->t->b->data)[0]; /* scalar on CPU */
        grad_scalar_mul_k<<<grid, block>>>(g_out, g_a, scalar, n);
        break;
    }

    case tensor_op_t::BROADCAST_ADD: {
        uint32_t rows = node->t->dims[0];
        uint32_t cols = node->t->dims[1];
        uint32_t total = rows * cols;
        int g_total = ((int)total + block - 1) / block;
        grad_broadcast_add_a_k<<<g_total, block>>>(g_out, g_a, total);
        int g_cols = ((int)cols + block - 1) / block;
        grad_broadcast_add_bias_k<<<g_cols, block>>>(g_out, g_b, rows, cols);
        break;
    }

    case tensor_op_t::NAIVE_MATRIX_MUL: {
        assert(node->t->a != nullptr);
        assert(node->t->b != nullptr);
        if (d_a_fwd == nullptr || d_b_fwd == nullptr) return false;

        uint32_t M = node->t->a->dims[0];  
        uint32_t K = node->t->a->dims[1]; 
        uint32_t N = node->t->b->dims[1];

        dim3 blk(MATMUL_BLOCK, MATMUL_BLOCK);
        dim3 grd_dA((K + MATMUL_BLOCK - 1) / MATMUL_BLOCK,
                    (M + MATMUL_BLOCK - 1) / MATMUL_BLOCK);
        dim3 grd_dB((N + MATMUL_BLOCK - 1) / MATMUL_BLOCK,
                    (K + MATMUL_BLOCK - 1) / MATMUL_BLOCK);

        if (g_a)
            grad_matmul_dA_k<<<grd_dA, blk>>>(g_out, d_b_fwd, g_a, M, K, N);
        if (g_b)
            grad_matmul_dB_k<<<grd_dB, blk>>>(d_a_fwd, g_out, g_b, M, K, N);
        break;
    }

    case tensor_op_t::MUL_MATRIX: {
        assert(node->t->a != nullptr);
        assert(node->t->b != nullptr);
        if (d_a_fwd == nullptr || d_b_fwd == nullptr) return false;

        uint32_t M = node->t->a->dims[0];
        uint32_t K = node->t->a->dims[1];
        uint32_t N = node->t->b->dims[node->t->b->is_transposed ? 0 : 1];

        dim3 blk(MATMUL_BLOCK, MATMUL_BLOCK);
        dim3 grd_dA((K + MATMUL_BLOCK - 1) / MATMUL_BLOCK,
                    (M + MATMUL_BLOCK - 1) / MATMUL_BLOCK);
        dim3 grd_dB((N + MATMUL_BLOCK - 1) / MATMUL_BLOCK,
                    (K + MATMUL_BLOCK - 1) / MATMUL_BLOCK);

        if (g_a)
            grad_matmul_dA_k<<<grd_dA, blk>>>(g_out, d_b_fwd, g_a, M, K, N);
        if (g_b)
            grad_matmul_dB_k<<<grd_dB, blk>>>(d_a_fwd, g_out, g_b, M, K, N);
        break;
    }

    case tensor_op_t::TRANSPOSE:
        return false;

    case tensor_op_t::NONE:
    case tensor_op_t::CAST:
        return true;

    default:
        debug("backprop_gpu_dispatch: unhandled op=%d\n", (int)node->t->op);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        debug("backprop_gpu_dispatch: CUDA error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

__global__ static void gpu_sgd_k(float *w, const float *g, float lr, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] -= lr * g[i];
    }
}

extern "C" bool tensor_sgd_gpu(float *d_w, float *d_g, float lr, uint32_t n) {
    int block = 256;
    int grid = ((int)n + block - 1) / block;
    gpu_sgd_k<<<grid, block>>>(d_w, d_g, lr, n);
    return cudaDeviceSynchronize() == cudaSuccess;
}


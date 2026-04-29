#include "internal_header.h"
#include <cuda_runtime.h>

// We are gonna write a backprop fucntion for matmul cause that is easy thing to do
// Since our backprop is an eagar evaluation we will make tensor objects inside the function
// transpose it execute it as well and then work with it

// Now I am slightly confused but I think it would have gone something like this
// Since we are going to get execution node we could have a decision made here if it's data will be
// written on device_ptr of node_.grad->data
// We are gonna do this cause making decsion on higher level will require us to make two kernels for similar work.
void gradInitializer(std::vector<execution_node_t *> &nodes) {
    for (auto &node : nodes) {
        if (!node->t->grad_compute) continue;
        // CPU
        memset(node->t->grad->data, 0, node->t->grad->nvalues * sizeof(float));
        // GPU
        if (node->device_ptr_grad != NULL) {
            cudaMemset(node->device_ptr_grad, 0,
                       node->t->nvalues * sizeof(float));
        }
    }
    // Seeding root node gradient
    if (!nodes.empty()) {
        execution_node_t *root = nodes.back();
        if (root->t != nullptr && root->t->grad != nullptr &&
            root->t->grad->data != nullptr) {
            ((float *)root->t->grad->data)[0] = 1.0f;
            // if on GPU then seed it don't know if it;s even possible
            if (root->device_ptr_grad != NULL) {
                float one = 1.0f;
                cudaMemcpy(root->device_ptr_grad, &one,
                           sizeof(float), cudaMemcpyHostToDevice);
            }
        }
    }
}
// Since this will be eagar evalution we are gonna execute it here now and we are gonna return a
// boolean.
// @return: boolean
// @params: execution_node_t
// Since all the memory is already executed we just need the execution node

bool backprop__(std::vector<execution_node_t *> &nodes) {
    for (int i = (int)nodes.size() - 1; i >= 0; i--) {
        execution_node_t *node = nodes[(size_t)i];
        if (!node->t->grad_compute) continue;
        if (node->t->op == tensor_op_t::NONE) continue;

        bool success = false;

        if (node->device_ptr_grad != NULL) {
            //  Resolve parent execution nodes for the GPU backward kernels 
            execution_node_t *parent_a = nullptr;
            execution_node_t *parent_b = nullptr;
            int32_t a_idx = getTheExecutionNodeIndex(node, 0);
            int32_t b_idx = getTheExecutionNodeIndex(node, 1);
            if (a_idx >= 0) parent_a = nodes[(size_t)a_idx];
            if (b_idx >= 0) parent_b = nodes[(size_t)b_idx];

            success = backprop_gpu_dispatch(node, parent_a, parent_b);
            if (!success) {
                 // Graceful fallback: copy grad from GPU, run CPU backward 
                debug("backprop__: GPU backward unsupported for op=%d, falling back to CPU\n",
                      (int)node->t->op);
                //  Pull this node's grad from GPU to CPU 
                if (node->device_ptr_grad != NULL && node->t->grad != NULL) {
                    cudaMemcpy(node->t->grad->data, node->device_ptr_grad,
                               node->t->grad->nvalues * sizeof(float),
                               cudaMemcpyDeviceToHost);
                }
                 // Pull parent forward data from GPU if needed 
                if (parent_a && parent_a->device_ptr != NULL &&
                    parent_a->t->device == device_type::GPU) {
                    cudaMemcpy(parent_a->t->data, parent_a->device_ptr,
                               parent_a->t->nvalues * sizeof(float),
                               cudaMemcpyDeviceToHost);
                }
                if (parent_b && parent_b->device_ptr != NULL &&
                    parent_b->t->device == device_type::GPU) {
                    cudaMemcpy(parent_b->t->data, parent_b->device_ptr,
                               parent_b->t->nvalues * sizeof(float),
                               cudaMemcpyDeviceToHost);
                }
                success = backprop_cpu(node);
                // Push parent grads back to GPU so SGD sees them 
                if (parent_a && parent_a->device_ptr_grad != NULL &&
                    parent_a->t->grad != NULL) {
                    cudaMemcpy(parent_a->device_ptr_grad,
                               parent_a->t->grad->data,
                               parent_a->t->grad->nvalues * sizeof(float),
                               cudaMemcpyHostToDevice);
                }
                if (parent_b && parent_b->device_ptr_grad != NULL &&
                    parent_b->t->grad != NULL) {
                    cudaMemcpy(parent_b->device_ptr_grad,
                               parent_b->t->grad->data,
                               parent_b->t->grad->nvalues * sizeof(float),
                               cudaMemcpyHostToDevice);
                }
            }
        } else {
            success = backprop_cpu(node);
        }

        if (!success) {
            debug("backprop__: FUBAR at node pos=%d\n", (int)node->pos);
            return false;
        }
    }
    return true;
}

// ================================================================================ 
// CPU BACKWARD DISPATCHER
// ================================================================================ 

bool backprop_cpu(execution_node_t *node) {
    assert(node != NULL);
    assert(node->t != NULL);
    bool success = false;

    switch (node->t->op) {
    case tensor_op_t::NONE:
        // Leaf node
        success = true;
        break;
    case tensor_op_t::CAST:
        // TODO: Implement here
        success = false;
        break;
    case tensor_op_t::MUL_MATRIX:
        assert(node->t->a != NULL);
        assert(node->t->b != NULL);
        assert(node->t->a->dtype == node->t->b->dtype);
        success = tensor_mul_grad_op_matrix(node);
        break;
    case tensor_op_t::MUL_SCALAR:
        assert(node->t->a != NULL);
        assert(node->t->b != NULL);
        assert(node->t->a->dtype == node->t->b->dtype);
        success = tensor_mul_grad_op_scalar(node);
        break;
    case tensor_op_t::TRANSPOSE:
        assert(node->t->a != NULL);
        success = tensor_tranpose_grad_op_matrix(node);
        break;
    case tensor_op_t::NAIVE_MATRIX_MUL:
        assert(node->t->a != NULL);
        assert(node->t->b != NULL);
        assert(node->t->a->dtype == node->t->b->dtype);
        success = tensor_mul_grad_op_matrix_naive(node);
        break;
    case tensor_op_t::ADD:
        assert(node->t->a != NULL);
        assert(node->t->b != NULL);
        assert(node->t->a->dtype == node->t->b->dtype);
        success = tensor_grad_op_add(node);
        break;
    case tensor_op_t::BROADCAST_ADD:
        assert(node->t->a != NULL);
        assert(node->t->b != NULL);
        assert(node->t->a->dtype == node->t->b->dtype);
        success = tensor_grad_op_broadcasting_add(node);
        break;
    case tensor_op_t::RELU:
        assert(node->t->a != NULL);
        success = tensor_grad_op_relu(node);
        break;
    case tensor_op_t::SUB:
        assert(node->t->a != NULL);
        assert(node->t->b != NULL);
        assert(node->t->a->dtype == node->t->b->dtype);
        success = tensor_grad_op_sub(node);
        break;
    case tensor_op_t::MEAN:
        assert(node->t->a != NULL);
        success = tensor_grad_op_mean(node);
        break;
    case tensor_op_t::SQUARE:
        assert(node->t->a != NULL);
        success = tensor_grad_op_square(node);
        break;
    default:
        assert(false);
    }
    if (success) {
        debug("backprop_cpu: success for op=%d\n", (int)node->t->op);
    } else {
        debug("backprop_cpu: FUBAR for op=%d\n", (int)node->t->op);
    }
    return success;
}


/// GPU DISPATCHER lives here  src/backend_gpu/backprop/backprop_gpu.cu :: backprop_gpu_dispatch().


// TODO: ONE PROBLEM IS WE NEED MULTIPLE CASE FOR 
bool tensor_grad_op_add(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a = node->t->a;
    tensor_t *b = node->t->b;

    assert(out_grad != NULL && a != NULL && b != NULL);
    float *g_out = (float *)out_grad->data;
    float *g_a = a->grad != NULL ? (float *)a->grad->data : NULL;
    float *g_b = b->grad != NULL ? (float *)b->grad->data : NULL;
    uint32_t n = node->t->nvalues;

    for (uint32_t i = 0; i < n; i++) {
        if (g_a) g_a[i] += g_out[i];
        if (g_b) g_b[i] += g_out[i];
    }
    return true;
}

bool tensor_grad_op_broadcasting_add(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a = node->t->a;
    tensor_t *b = node->t->b;

    assert(out_grad != NULL && a != NULL && b != NULL);
    float *g_out = (float *)out_grad->data;
    float *g_a = a->grad != NULL ? (float *)a->grad->data : NULL;
    float *g_b = b->grad != NULL ? (float *)b->grad->data : NULL;

    uint32_t rows = node->t->dims[0];
    uint32_t cols = node->t->dims[1];
    // We are assuming that b would be the broadcasted array
    uint32_t val = node->t->nvalues;
    if (g_a) {
        for (uint32_t i = 0; i < val; i++) {
            g_a[i] += g_out[i];
        }
    }

    if (g_b) {
        // For broadcasting we accumulate btw
        for (uint32_t i = 0; i < rows; i++) {
          for (uint32_t j = 0; j < cols; j++) {
            uint32_t b_idx = i * b->broadcast_stride[0]  + j * b->broadcast_stride[1];
            g_b[b_idx] += g_out[i*cols + j];
          }
        }
    }
    return true;
}

bool tensor_grad_op_sub(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a        = node->t->a;
    tensor_t *b        = node->t->b;
    assert(out_grad != NULL && a != NULL && b != NULL);

    float *g_out = (float *)out_grad->data;
    float *g_a   = a->grad != NULL ? (float *)a->grad->data : NULL;
    float *g_b   = b->grad != NULL ? (float *)b->grad->data : NULL;
    uint32_t n   = node->t->nvalues;

    for (uint32_t i = 0; i < n; i++) {
        if (g_a) g_a[i] += g_out[i];
        if (g_b) g_b[i] -= g_out[i];
    }
    return true;
}



bool tensor_grad_op_relu(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a        = node->t->a;
    assert(out_grad != NULL && a != NULL);

    if (a->grad != NULL) {
        float *g_out  = (float *)out_grad->data;
        float *g_a    = (float *)a->grad->data;
        float *a_data = (float *)a->data;   
        uint32_t n    = node->t->nvalues;

        for (uint32_t i = 0; i < n; i++) {
            if (a_data[i] > 0.0f) {
                g_a[i] += g_out[i];
            }
        }
    }
    return true;
}

bool tensor_grad_op_mean(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a        = node->t->a;
    assert(out_grad != NULL && a != NULL);

    if (a->grad != NULL) {
        // out->grad is a scalar — one value
        float upstream    = ((float *)out_grad->data)[0];
        float *g_a        = (float *)a->grad->data;
        uint32_t n        = a->nvalues;
        float scale       = upstream / (float)n;

        for (uint32_t i = 0; i < n; i++) {
            g_a[i] += scale;
        }
    }
    return true;
}

bool tensor_mul_grad_op_scalar(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a        = node->t->a;
    tensor_t *b        = node->t->b;   
    assert(out_grad != NULL && a != NULL && b != NULL);

    if (a->grad != NULL) {
        float *g_out  = (float *)out_grad->data;
        float *g_a    = (float *)a->grad->data;
        float  s      = tensor_float32_value(b);
        uint32_t n    = a->nvalues;

        for (uint32_t i = 0; i < n; i++) {
            g_a[i] += g_out[i] * s;
        }
    }
    return true;
}


bool tensor_tranpose_grad_op_matrix(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a        = node->t->a;
    assert(out_grad != NULL && a != NULL);

    if (a->grad != NULL) {
        uint32_t rows    = a->dims[0];   
        uint32_t cols    = a->dims[1];  
        float *g_out     = (float *)out_grad->data;
        float *g_a       = (float *)a->grad->data;

        for (uint32_t i = 0; i < rows; i++) {
            for (uint32_t j = 0; j < cols; j++) {
                g_a[i * cols + j] += g_out[j * rows + i];
            }
        }
    }
    return true;
}

bool tensor_mul_grad_op_matrix_naive(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a        = node->t->a;
    tensor_t *b        = node->t->b;
    assert(out_grad != NULL && a != NULL && b != NULL);

    uint32_t M = a->dims[0];
    uint32_t K = a->dims[1];
    uint32_t N = b->dims[1];

    float *g_out = (float *)out_grad->data;    // (M, N)
    float *a_data = (float *)a->data;          // (M, K)
    float *b_data = (float *)b->data;          // (K, N)
    float *g_a    = a->grad != NULL ? (float *)a->grad->data : NULL;    // (M, K)
    float *g_b    = b->grad != NULL ? (float *)b->grad->data : NULL;    // (K, N)

    if (g_a) {
        // dL/dA[i,k] += sum_j  dL/dOut[i,j] * B[k,j]
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t k = 0; k < K; k++) {
                float sum = 0.0f;
                for (uint32_t j = 0; j < N; j++) {
                    sum += g_out[i * N + j] * b_data[k * N + j];
                }
                g_a[i * K + k] += sum;
            }
        }
    }

    if (g_b) {
        // dL/dB[k,j] += sum_i  A[i,k] * dL/dOut[i,j]
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (uint32_t i = 0; i < M; i++) {
                    sum += a_data[i * K + k] * g_out[i * N + j];
                }
                g_b[k * N + j] += sum;
            }
        }
    }
    return true;
}

bool tensor_grad_op_square(execution_node_t *node) {
    tensor_t *out_grad = node->t->grad;
    tensor_t *a        = node->t->a;
    assert(out_grad != NULL && a != NULL);

    if (a->grad != NULL) {
        float *g_out  = (float *)out_grad->data;
        float *g_a    = (float *)a->grad->data;
        float *a_data = (float *)a->data;   
        uint32_t n    = node->t->nvalues;

        for (uint32_t i = 0; i < n; i++) {
            g_a[i] += g_out[i] * 2.0f * a_data[i];
        }
    }
    return true;
}



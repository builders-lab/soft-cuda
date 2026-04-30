#include "internal_header.h"

bool tensor_mul_grad_op_matrix(execution_node_t *node) {
  
    tensor_t *out_grad = node->t->grad;
    tensor_t *a        = node->t->a;
    tensor_t *b        = node->t->b;
    assert(out_grad != NULL && a != NULL && b != NULL);

    uint32_t M = a->dims[0];
    uint32_t K = a->dims[1];
    uint32_t N = b->is_transposed ? b->dims[0] : b->dims[1];

    float *g_out  = (float *)out_grad->data;           
    float *a_data = (float *)a->data;                  
    float *b_data = (float *)b->data;                  
    float *g_a    = a->grad ? (float *)a->grad->data : NULL;  
    float *g_b    = b->grad ? (float *)b->grad->data : NULL; 

    if (g_a) {
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

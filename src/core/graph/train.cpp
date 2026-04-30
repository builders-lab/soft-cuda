#include "internal_header.h"
#include <vector>

bool tensor_graph_backward(std::vector<execution_node_t *> &nodes) {
    return backprop__(nodes);
}


void tensor_sgd(std::vector<execution_node_t *> &nodes, float learning_rate) {
    for (auto &node : nodes) {
        tensor_t *t = node->t;

        if (t->op != tensor_op_t::NONE) continue;
        if (!t->grad_compute)           continue;
        if (t->grad == NULL)            continue;

        // If both weights and grads are on GPU, update on GPU
        if (node->device_ptr != NULL && node->device_ptr_grad != NULL) {
            if (tensor_sgd_gpu((float *)node->device_ptr, (float *)node->device_ptr_grad, 
                               learning_rate, t->nvalues)) {
                continue; // Success, skip CPU update
            }
        }

        if (node->device_ptr_grad != NULL && t->grad->data != NULL) {
            soft_cuda_memcpy_d2h(t->grad->data, node->device_ptr_grad,
                                  t->grad->nvalues * sizeof(float));
        }
        
        // If weights were on GPU, we might need to pull them too, 
        // but leaf weights usually stay on host unless moved.
        if (node->device_ptr != NULL && t->data != NULL) {
             soft_cuda_memcpy_d2h(t->data, node->device_ptr,
                                  t->nvalues * sizeof(float));
        }
        float *w    = (float *)t->data;
        float *grad = (float *)t->grad->data;
        uint32_t n  = t->nvalues;

        for (uint32_t i = 0; i < n; i++) {
            w[i] -= learning_rate * grad[i];
        }

        // If we updated on CPU but they are shadowed on GPU, push back
        if (node->device_ptr != NULL && t->data != NULL) {
             soft_cuda_memcpy_h2d(node->device_ptr, t->data,
                                  t->nvalues * sizeof(float));
        }
    }
}

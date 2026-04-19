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

        float *w    = (float *)t->data;
        float *grad = (float *)t->grad->data;
        uint32_t n  = t->nvalues;

        for (uint32_t i = 0; i < n; i++) {
            w[i] -= learning_rate * grad[i];
            // grad[i] = 0.0f;   
        }
    }
}

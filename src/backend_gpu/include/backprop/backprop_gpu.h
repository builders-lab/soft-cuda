#pragma once

extern "C" {
bool backprop_gpu_dispatch(execution_node_t *node,
                            execution_node_t *parent_a,
                            execution_node_t *parent_b);

bool tensor_sgd_gpu(float *d_w, float *d_g, float lr, uint32_t n);
}

#pragma once

void gradInitializer(std::vector<execution_node_t *> &nodes);

bool backprop__(std::vector<execution_node_t *> &nodes);

bool backprop_cpu(execution_node_t *node);


void soft_cuda_memset_zero(void *ptr, size_t bytes);
void soft_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes);
void soft_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes);
bool tensor_grad_op_add(execution_node_t *node);

bool tensor_grad_op_broadcasting_add(execution_node_t *node);

bool tensor_grad_op_sub(execution_node_t *node);

bool tensor_grad_op_relu(execution_node_t *node);

bool tensor_grad_op_mean(execution_node_t *node);

bool tensor_mul_grad_op_scalar(execution_node_t *node);

bool tensor_tranpose_grad_op_matrix(execution_node_t *node);

bool tensor_mul_grad_op_matrix_naive(execution_node_t *node);

bool tensor_grad_op_square(execution_node_t *node);

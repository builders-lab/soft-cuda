#pragma once

void gradInitializer(std::vector<execution_node_t *> &nodes);

bool backprop__(std::vector<execution_node_t *> &nodes);

bool backprop_cpu(execution_node_t *node);

bool backprop_gpu([[maybe_unused]] execution_node_t *node);

bool tensor_grad_op_add(execution_node_t *node);

bool tensor_grad_op_broadcasting_add(execution_node_t *node);

bool tensor_grad_op_sub(execution_node_t *node);

bool tensor_grad_op_relu(execution_node_t *node);

bool tensor_grad_op_mean(execution_node_t *node);

bool tensor_mul_grad_op_scalar(execution_node_t *node);

bool tensor_tranpose_grad_op_matrix(execution_node_t *node);

bool tensor_mul_grad_op_matrix_naive(execution_node_t *node);

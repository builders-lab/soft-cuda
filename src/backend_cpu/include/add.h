#pragma once

tensor_t *tensor_add(tensor_pool_t *pool, tensor_t *x, tensor_t *y);

bool tensor_op_add(tensor_pool_t *pool, tensor_t *t);

bool tensor_op_broadcasting_add(tensor_pool_t *pool, tensor_t *t);

float get_sum_for_col_op(tensor_t *b, uint32_t i, tensor_t *t);

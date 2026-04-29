#pragma once
#include <cstdint>

bool tensor_scalar_mul_op_cuda(tensor_t *t, float *d_a, float *d_b, float *d_res);

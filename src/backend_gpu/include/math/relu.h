#pragma once
#include <cstdint>

bool tensor_relu_op_cuda(tensor_t *t, float *d_a, float *d_res);

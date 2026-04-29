#pragma once
#include <cstdint>

bool tensor_mean_op_cuda(tensor_t *t, float *d_a, float *d_res);

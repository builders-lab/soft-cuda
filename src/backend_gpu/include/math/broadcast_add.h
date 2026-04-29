#pragma once
#include <cstdint>

bool tensor_broadcast_add_op_cuda(tensor_t *t, float *d_a, float *d_b, float *d_res);

#pragma once

#include "internal_header.h"

tensor_t *tensor_square(tensor_pool_t *pool, tensor_t *x);

bool tensor_op_square(tensor_pool_t *pool, tensor_t *t);

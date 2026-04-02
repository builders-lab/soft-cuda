#include "internal_header.h"

tensor_t *tensor_square(tensor_pool_t *pool, tensor_t *x) {
    return tensor_mul(pool, x, x);
}

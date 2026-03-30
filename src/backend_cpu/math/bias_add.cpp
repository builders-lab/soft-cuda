#include "internal_header.h"

tensor_t *tensor_add_bias(tensor_pool_t *pool, tensor_t *xw, tensor_t *bias) {
    return tensor_add(pool, xw, bias);
}

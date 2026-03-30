#include "internal_header.h"


tensor_t *tensor_relu(tensor_pool_t *pool, tensor_t *x) {
    assert(pool != NULL);
    assert(x != NULL);

    tensor_t *t = tensor_dtype_create(pool, x->dtype, x->ndims, x->dims, NULL);

    t->op = tensor_op_t::RELU;
    t->a = x;
    t->b = NULL;
    return t;
}

bool tensor_op_relu(tensor_pool_t *pool, tensor_t *t) {
    // TODO: Once again remainder to support different dtype cause it's just a prop with only float for now
    assert(pool != NULL);
    assert(t != NULL);

    float* float_data = ((float*)(t->data));

    for(uint32_t i = 0; i < t->nvalues; i++) {
        if (float_data[i] < 0) {
            float_data[i] = 0;
        }
    }
    return true;
}

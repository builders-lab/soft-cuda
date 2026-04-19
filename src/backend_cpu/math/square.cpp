#include "internal_header.h"

tensor_t *tensor_square(tensor_pool_t *pool, tensor_t *x) {
    assert(pool != NULL);
    assert(x != NULL);

    tensor_t *t = tensor_dtype_create(pool, x->dtype, x->ndims, x->dims, NULL);
    if (t == NULL) {
        return NULL;
    }

    t->op = tensor_op_t::SQUARE;
    t->a = x;
    
    return t;
}

bool tensor_op_square(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);
    assert(t->a != NULL);

    float *in = (float *)t->a->data;
    float *out = (float *)t->data;
    uint32_t n = t->nvalues;

    for (uint32_t i = 0; i < n; i++) {
        out[i] = in[i] * in[i];
    }
    return true;
}


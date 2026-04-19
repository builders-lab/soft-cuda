#include "internal_header.h"

tensor_t *tensor_sub(tensor_pool_t *pool, tensor_t *a, tensor_t *b) {
    assert(pool != NULL);
    assert(a != NULL);
    assert(b != NULL);

    tensor_t *t = tensor_dtype_create(pool, a->dtype, a->ndims, a->dims, NULL);
    if (t == NULL) {
        return NULL;
    }

    t->op = tensor_op_t::SUB;
    t->a = a;
    t->b = b;
    return t;
}

bool tensor_op_sub(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);

    for (uint32_t i = 0; i < t->dims[0] * t->dims[1]; i++) {
        ((float *)t->data)[i] = ((float *)t->a->data)[i] - ((float *)t->b->data)[i];
    }
    return true;
}

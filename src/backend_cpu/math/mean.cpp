#include "internal_header.h"

tensor_t *tensor_mean(tensor_pool_t *pool, tensor_t *a) {
    assert(pool != NULL);
    assert(a != NULL);

    tensor_t *t = tensor_dtype_create(pool, a->dtype, 0, 0, NULL);
    if (t == NULL) {
        return NULL;
    }

    t->op = tensor_op_t::MEAN;
    t->a = a;
    return t;
}

bool tensor_op_mean(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);
    float sum{};
    for (uint32_t i = 0; i < t->a->nvalues; i++) {
        sum += ((float *)t->a->data)[i];
    }
    ((float *)t->data)[0] = sum/(float)(t->a->nvalues);
    return true;
}

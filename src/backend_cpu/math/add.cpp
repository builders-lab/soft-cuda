#include "internal_header.h"

static tensor_t *tensor_broadcast_add(tensor_pool_t *pool, tensor_t *a, tensor_t *b) {
    assert(pool != NULL);
    assert(a != NULL);
    assert(b != NULL);

    if (b->dims[0] == 1) {
        b->broadcast_stride[0] = 0;
        b->broadcast_stride[1] = b->stride[1];
    } else {
        b->broadcast_stride[1] = 0;
        b->broadcast_stride[0] = b->stride[0];
    }

    tensor_t *t = tensor_dtype_create(pool, a->dtype, a->ndims, a->dims, NULL);
    if (t == NULL) {
        return NULL;
    }

    t->op = tensor_op_t::BROADCAST_ADD;
    t->a = a;
    t->b = b;
    return t;
}

// We are baking lots of assumptions in this one haha
tensor_t *tensor_add(tensor_pool_t *pool, tensor_t *x, tensor_t *y) {
    assert(pool != NULL);
    assert(x != NULL);
    assert(y != NULL);

    if (x->nvalues == y->nvalues) {
        tensor_t *t = tensor_dtype_create(pool, x->dtype, x->ndims, x->dims, NULL);
        if (t == NULL) return NULL;
        t->op = tensor_op_t::ADD;
        t->a = x;
        t->b = y;
        return t;
    }

    if (y->nvalues < x->nvalues && (y->dims[0] == 1 || y->dims[1] == 1)) {
        return tensor_broadcast_add(pool, x, y);
    }
    
    if (x->nvalues < y->nvalues && (x->dims[0] == 1 || x->dims[1] == 1)) {
        return tensor_broadcast_add(pool, y, x);
    }

    return NULL; // Unsupported broadcast shape
}

bool tensor_op_add(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);

    for (uint32_t i = 0; i < t->dims[0] * t->dims[1]; i++) {
        ((float *)t->data)[i] = ((float *)t->a->data)[i] + ((float *)t->b->data)[i];
    }
    return true;
}

bool tensor_op_broadcasting_add(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);
    assert(t->a != NULL);
    assert(t->b != NULL);

    for (uint32_t i = 0; i < t->dims[0] * t->dims[1]; i++) {
        ((float *)t->data)[i] = ((float *)t->a->data)[i] + get_sum_for_col_op(t->b, i, t);
    }
    return true;
}

float get_sum_for_col_op(tensor_t *b, uint32_t i, tensor_t *t) {
    uint32_t row = i / (t->dims[1]);
    uint32_t col = i % (t->dims[1]);
    float val =
        ((float *)b->data)[row * ((b->broadcast_stride)[0]) + col * ((b->broadcast_stride)[1])];
    return val;
}


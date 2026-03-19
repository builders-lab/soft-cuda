#include "internal_header.h"


// We are baking lots of assumptions in this one haha
tensor_t* tensor_add(tensor_pool_t *pool, tensor_t *x, tensor_t *y) {
    assert(pool != NULL);
    assert(x != NULL);
    assert(y != NULL);
    // Assumption no. 1 it will alwars be a similar shape matrix and 6
    assert(x->dims[0] == y->dims[0] && x->dims[1] == y->dims[1]);
    tensor_t *t = tensor_dtype_create(pool, x->dtype, x->ndims, x->dims, NULL);
    if(t == NULL) {
        return NULL;
    }
    t->op =  tensor_op_t::ADD;
    t->a = x;
    t->b = y;
    return t;
}


bool tensor_op_add(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);
    for(uint32_t i = 0; i < t->dims[0]*t->dims[1]; i++) {
        ((float *)t->data)[i] = ((float *)t->a->data)[i] + ((float *)t->b->data)[i];
    }
    return true;
}

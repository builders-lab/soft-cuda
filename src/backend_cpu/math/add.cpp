#include "internal_header.h"


// We are baking lots of assumptions in this one haha
tensor_t* tensor_add(tensor_pool_t *pool, tensor_t *x, tensor_t *y) {
    assert(pool != NULL);
    assert(x != NULL);
    assert(y != NULL);
    // Assumption no. 1 it will alwars be a similar shape matrix and 6
    // assert(x->dims[0] == y->dims[0] && x->dims[1] == y->dims[1]);
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

    if(t->a->dims[0] == 1 || t->a->dims[1] == 0) { 
       return tensor_op_broadcasting_add(pool, t, t->b, t->a);  
    } 
    if(t->b->dims[0] == 1 || t->b->dims[1] == 0) { 
       return tensor_op_broadcasting_add(pool, t, t->a, t->b);  
    }

    for(uint32_t i = 0; i < t->dims[0]*t->dims[1]; i++) {
        ((float *)t->data)[i] = ((float *)t->a->data)[i] + ((float *)t->b->data)[i];
    }
    return true;
}

bool tensor_op_broadcasting_add(tensor_pool_t *pool, tensor_t *t, tensor_t *a, tensor_t *b) {
    assert(pool != NULL);
    assert(t != NULL);
    assert(a != NULL);
    assert(b != NULL);
    
    if(b->dims[0] == 1) {
        b->stride[0] = 0;
    } else {
        b->stride[1] = 0;
    }

    for(uint32_t i = 0; i < t->dims[0]*t->dims[1]; i++) {
        ((float *)t->data)[i] = ((float *)a->data)[i] + get_sum_for_col_op(b, i, t);
    }
    return true;
}

float get_sum_for_col_op(tensor_t *b, uint32_t i, tensor_t *t) {
    uint32_t row = i/(t->dims[1]);
    uint32_t col = i%(t->dims[1]);
    float val = ((float*)b->data)[row*((b->stride)[0]) + col*((b->stride)[1])];
    return val;
}

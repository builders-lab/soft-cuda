#ifndef TENSOR_MUL_H
#define TENSOR_MUL_H

// Multiply a and b where they are the same type, and b is a scalar value.
bool tensor_mul_op_scalar(tensor_pool_t *pool, tensor_t *t);

// Multiply a and b where they are the same type and dimension.
bool tensor_mul_op_matrix(tensor_pool_t *pool, tensor_t *t);

inline bool tensor_is_scalar(tensor_t *t) {
    assert(t!= NULL);
    return t->ndims == 0;
}

tensor_t *tensor_mul(tensor_pool_t *pool, tensor_t *a, tensor_t *b);

#endif //  TENSOR_MUL_H

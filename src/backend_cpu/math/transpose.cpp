#include "internal_header.h"

tensor_t *tensor_transpose(tensor_pool_t *pool, tensor_t *a) {
    assert(pool != NULL);
    assert(a != NULL);
    if (tensor_is_scalar(a)) {
        return a;
    }
    uint32_t transposed_dim[TENSOR_MAX_DIMS] = {};
    uint32_t *transposed_matrix = NULL;
    if (a->ndims == 2) {
        transposed_dim[0] = a->dims[1];
        transposed_dim[1] = a->dims[0];
        transposed_matrix = transposed_dim;
    }

    if (transposed_matrix == NULL) {
        debug("tensor_transpose: tensor shape=%zu is not supported for transpose operation",
              a->ndims);
        return NULL;
    }

    tensor_t *t = tensor_dtype_create(pool, a->dtype, a->ndims, transposed_matrix, NULL);
    if (t == NULL) {
        return NULL;
    }
    t->op = tensor_op_t::TRANSPOSE;
    t->a = a;
    t->b = NULL;
    t->is_transposed = true;
    return t;
}

// TODO: Implement Loop tiling matrix transpose for better performance
// Performs deep transpose
bool tensor_tranpose_op_matrix(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);
    uint32_t row = t->a->dims[0];
    uint32_t col = t->a->dims[1];
    float *src = (float *)t->a->data;
    float *des = (float *)t->data;
    for (uint32_t i = 0; i < row; i++) {
        for (uint32_t j = 0; j < col; j++) {
            des[j * row + i] = src[i * col + j];
        }
    }
    return true;
}

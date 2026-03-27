#include "internal_header.h"

// TODO: multi-dtype support, see tensor_dtype_sizeof
// for now everything is float32

/////////////////////////////////////////////////////
// PRIVATE METHODS

static tensor_t *tensor_mul_scalar(tensor_pool_t *pool, tensor_t *a, tensor_t *b) {
    assert(pool != NULL);
    assert(a != NULL);
    assert(b != NULL);

    // Create a new tensor with same dimension
    tensor_t *t = tensor_dtype_create(pool, a->dtype, a->ndims, a->dims, NULL);
    if (t == NULL) {
        return NULL;
    }

    // Set the operation to MUL_SCALAR, and dependencies to a and b.
    t->op = tensor_op_t::MUL_SCALAR;
    t->a = a;
    t->b = b;

    return t;
}

/////////////////////////////////////////////////////////////////
// PUBLIC METHODS

tensor_t *tensor_mul(tensor_pool_t *pool, tensor_t *a, tensor_t *b) {
    assert(pool != NULL);
    assert(a != NULL);
    assert(b != NULL);

    if (a->dtype != b->dtype) {
        debug("tensor_mul: tensors do not have similar datatype\n");
        return NULL;
    }

    // Check for scalar values
    if (tensor_is_scalar(a)) {
        return tensor_mul_scalar(pool, b, a);
    }

    if (tensor_is_scalar(b)) {
        return tensor_mul_scalar(pool, a, b);
    }

    // Check for same number of dimensions
    if (a->ndims != b->ndims) {
        debug("tensor_mul: tensors do not have same number of dimensions\n");
        return NULL;
    }
    assert(a->ndims == 2);

    uint32_t matrix_mul_shape_placeholder[TENSOR_MAX_DIMS] = {};
    uint32_t *matrix_mul_shape;
    if (b->is_transposed) {
        debug("status of is_transposed=%d", b->is_transposed);
        assert(a->dims[1] == b->dims[1]);
        matrix_mul_shape_placeholder[0] = a->dims[0];
        matrix_mul_shape_placeholder[1] = b->dims[0];
        matrix_mul_shape = matrix_mul_shape_placeholder;
    } else {
        assert(a->dims[1] == b->dims[0]);
        matrix_mul_shape_placeholder[0] = a->dims[0];
        matrix_mul_shape_placeholder[1] = b->dims[1];
        matrix_mul_shape = matrix_mul_shape_placeholder;
    }
    tensor_t *t = tensor_dtype_create(pool, a->dtype, a->ndims, matrix_mul_shape, NULL);
    if (t == NULL) {
        return NULL;
    }

    t->op = tensor_op_t::MUL_MATRIX;
    t->a = a;
    t->b = b;

    return t;
}

static bool tensor_mul_op_scalar_float32(float *out, float *in, uint32_t nvalues, float b) {
    for (uint32_t i = 0; i < nvalues; i++) {
        out[i] = in[i] * b;
    }
    return true;
}

/////////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS - EVALUATE

bool tensor_mul_op_scalar(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);
    assert(t->a != NULL);
    assert(t->b != NULL);
    assert(t->dtype == t->a->dtype && t->a->dtype == t->b->dtype);
    assert(tensor_is_scalar(t->b));
    switch (t->dtype) {
    case tensor_dtype_t::INT32_T:
    case tensor_dtype_t::UINT32_T:
    case tensor_dtype_t::INT64_T:
    case tensor_dtype_t::UINT64_T:
    case tensor_dtype_t::FLOAT32_T:
        return tensor_mul_op_scalar_float32((float *)t->data, (float *)t->a->data, t->a->nvalues,
                                            tensor_float32_value(t->b));
    case tensor_dtype_t::FLOAT64_T:
    default:
        return false;
    }
}

// here the t->b is promised to be a transposed matrix.
bool tensor_mul_op_matrix(tensor_pool_t *pool, tensor_t *t) {
    // TODO: Implement matrix multiplication
    assert(pool != NULL);
    assert(t != NULL);
    if (t->b->is_transposed == false) {
        assert(t->a->dims[1] == t->b->dims[0]);
        return tensor_mul_op_matrix_naive(pool, t);
    }
    uint32_t row = t->a->dims[0];

    assert(t->a->dims[1] == t->b->dims[1]);
    // B is transposed: dims[0] is original cols = output cols
    uint32_t col = t->b->dims[0];
    uint32_t kdim = t->a->dims[1];
    float *src1 = (float *)t->a->data;
    float *src2 = (float *)t->b->data;
    float *des = (float *)t->data;
    for (uint32_t i = 0; i < row; i++) {
        for (uint32_t j = 0; j < col; j++) {
            float msum = 0.0f;
            for (uint32_t k = 0; k < kdim; k++) {
                msum += src1[i * kdim + k] * src2[j * kdim + k];
            }
            des[i * col + j] = msum;
        }
    }

    return true;
}

tensor_t *tensor_mul_naive(tensor_pool_t *pool, tensor_t *a, tensor_t *b) {
    assert(pool != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->dtype == b->dtype);

    if (a->ndims != b->ndims) {
        debug("tensor_mul: tensors do not have same number of dimensions\n");
        return NULL;
    }
    assert(a->ndims == 2);
    assert(a->dims[1] == b->dims[0]);

    uint32_t matrix_mul_shape[] = {a->dims[0], b->dims[1]};
    tensor_t *t = tensor_dtype_create(pool, a->dtype, a->ndims, matrix_mul_shape, NULL);
    if (t == NULL) {
        return NULL;
    }

    t->op = tensor_op_t::NAIVE_MATRIX_MUL;
    t->a = a;
    t->b = b;

    return t;
}

bool tensor_mul_op_matrix_naive(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);
    uint32_t row = t->a->dims[0];
    uint32_t col = t->b->dims[1];
    uint32_t kdim = t->a->dims[1];
    float *src1 = (float *)t->a->data;
    float *src2 = (float *)t->b->data;
    float *des = (float *)t->data;
    for (uint32_t i = 0; i < row; i++) {
        for (uint32_t j = 0; j < col; j++) {
            float msum = 0.0f;
            for (uint32_t k = 0; k < kdim; k++) {
                msum += src1[i * kdim + k] * src2[k * col + j];
            }
            des[i * col + j] = msum;
        }
    }
    return true;
}

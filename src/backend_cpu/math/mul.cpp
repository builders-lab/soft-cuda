#include "internal_header.h"

// TODO: multi-dtype support, see tensor_dtype_sizeof
// for now everything is float32


/////////////////////////////////////////////////////
// PRIVATE METHODS

inline bool tensor_is_scalar(tensor_t *t) {
    assert(t!= NULL);
    return t->ndims == 0;
}

static tensor_t *tensor_mul_scalar(tensor_pool_t *pool, tensor_t *a, tensor_t *b) {
    assert(pool != NULL);
    assert(a != NULL);
    assert(b != NULL);

    // Create a new tensor with same dimension
    tensor_t *t = tensor_dtype_create(pool, a->dtype, a->dims, NULL);
    if(t == NULL) {
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

    if (a -> dtype != b -> dtype) {
        debug("tensor_mul: tensors do not have same number of dimensions\n");
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
    if (a -> ndims != b-> ndims) {
        debug("tensor_mul: tensors do not have same number of dimensions\n");
        return NULL;
    }

    // Check for same dimension
    for (uint8_t i = 0; i < a->ndims; i++) {
        if (a->dims[i] != b->dims[i]) {
            debug("tensor_mul: tensors do not have same shape\n");
            return NULL;
        }
    }

    tensor_t *t = tensor_dtype_create(pool, a->dtype, a->dims, NULL);
    if(t == NULL) {
        return NULL;
    }

    t->op = tensor_op_t::MUL_MATRIX;
    t->a = a;
    t->b = b;

    return t;
}



static bool tensor_mul_op_scalar_float32(float *data, uint32_t nvalues, float b) {
    for (uint32_t i = 0; i < nvalues; i++) {
        data[i] *=b;
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
    switch (t->dtype)
    {
      case tensor_dtype_t::INT32_T:
      case tensor_dtype_t::UINT32_T: 
      case tensor_dtype_t::INT64_T: 
      case tensor_dtype_t::UINT64_T: 
      case tensor_dtype_t::FLOAT32_T: 
            return  tensor_mul_op_scalar_float32((float *)t->a->data, t->a->nvalues, tensor_float32_value(t->b));
      case tensor_dtype_t::FLOAT64_T: 
    }
}


bool tensor_mul_op_matrix(tensor_pool_t *pool, tensor_t *t) {
    // TODO: Implement matrix multiplication
    return false;
}

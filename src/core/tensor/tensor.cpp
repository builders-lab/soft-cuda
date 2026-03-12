#include "internal_header.h"
#include "stdlib.h"
#include "string.h"

// #define buf_size 80
// static char buf[buf_size];

inline uint32_t tensor_id(tensor_t *t) {
    assert(t != NULL);
    return t->id;
}

size_t tensor_dtype_sizeof(tensor_dtype_t dtype) {
    switch (dtype) {
        case tensor_dtype_t::INT32_T:
            return sizeof(int32_t);
        case tensor_dtype_t::UINT32_T:
            return sizeof(uint32_t);    
        case tensor_dtype_t::INT64_T:
            return sizeof(int64_t);  
        case tensor_dtype_t::UINT64_T:
            return sizeof(uint64_t);
        case tensor_dtype_t::FLOAT32_T:
            return sizeof(float);
        case tensor_dtype_t::FLOAT64_T:
            return sizeof(double);
        default:
            return 0;
    }
}


tensor_t *tensor_dtype_create(tensor_pool_t *pool, tensor_dtype_t dtype, uint32_t *dims, void *elems) {
    assert(pool != NULL);
    assert(tensor_dtype_sizeof(dtype) > 0);

    // Check for scalar
    size_t size = 0;
    int ndims = -1;
    if(dims == NULL || dims[0] == 0) {
        size = 1;
        ndims = 0;
    } else {
        while (dims[++ndims] != 0) {
            if (size == 0) {
                size = dims[ndims];
            } else {
                size *=dims[ndims];
            }
        }
    }

    assert(size <= UINT32_MAX);

    // If no size, return NULL
    if (size == 0) {
        debug("tensor_dtype_create: failed, zero size.\n");
        return NULL;
    }

    // Retrun NULL if no dim or too many dim
    if (ndims > TENSOR_MAX_DIMS) {
        debug("tensor_dtype_create: failed, ndims=%zu\n", ndims);
        return NULL;
    }
    
    // Allocate memory for the tensor
    uint32_t id;
    tensor_t *t = (tensor_t *)tensor_pool_alloc(pool, sizeof(tensor_t), &id);
    if (t == NULL) {
        debug("tensor_dtype_create: failed, out of memory allocating %ld bytes.\n", sizeof(tensor_t));
        return NULL;
    }

    // Allocate memory for data and set it
    t->data = tensor_pool_alloc(pool, size*tensor_dtype_sizeof(dtype), NULL);
    if (t->data == NULL) {
        debug("tensor_dtype_create: failed, out of memory allocating %ld bytes.\n", sizeof(tensor_t));
        return NULL;
    } else if (elems != NULL) {
        memcpy(t->data, elems, size*tensor_dtype_sizeof(dtype));
    } else {
        memset(t->data, 0, size*tensor_dtype_sizeof(dtype));
    }

    //set tensor properties
    t->dtype = dtype;
    t->ndims = (uint8_t)ndims;
    t->nvalues = size;
    for (uint8_t i = 0; i < ndims; i++) {
        t->dims[i] = dims[i];
    }
    // TODO: Implement the stride logic
    t->dims[ndims] = 0;
    t->op = tensor_op_t::NONE;
    t->id = id;

    t->device = device_type::CPU;
    t->grad_compute = false;

    // Return success
    return t;
}


// Create float32 tensor
inline tensor_t *tensor_create(tensor_pool_t *pool,tensor_dtype_t dtype, uint32_t num_dims ,uint32_t *dims, float *elems) {
    return tensor_dtype_create(pool, dtype, dims, elems);
    // TODO: Handle num_dims to proccede with stride logic
}

bool tensor_evaluate(tensor_pool_t *pool, tensor_t *t) {
    assert(pool != NULL);
    assert(t != NULL);
    bool success = false;

    switch (t->op) {
        case tensor_op_t::NONE:
            success = true;
            break;
        case tensor_op_t::CAST:
            break;
            // TODO: Implement here
        case tensor_op_t::MUL_MATRIX:
            assert(t->a != NULL);
            assert(t->b != NULL);
            assert(t->a->dtype == t->b->dtype);
            success = tensor_mul_op_matrix(pool,t);
            break;
        case tensor_op_t::MUL_SCALAR:
            assert(t->a != NULL);
            assert(t->b != NULL);
            assert(t->a->dtype == t->b->dtype);
            success = tensor_mul_op_scalar(pool,t);
            break;
        default:
            assert(false);

    }
    if (success) {
        debug("tensor_evaluate: success\n");
    } else {
        debug("tensor_evaluate: FUBAR\n");
    }
    return success;
}


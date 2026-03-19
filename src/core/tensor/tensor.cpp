#include "internal_header.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>

// #define buf_size 80
// static char buf[buf_size];

uint32_t tensor_id(tensor_t *t) {
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


tensor_t *tensor_dtype_create(tensor_pool_t *pool, tensor_dtype_t dtype,uint32_t ndims, uint32_t *dims, void *elems) {
    assert(pool != NULL);
    assert(tensor_dtype_sizeof(dtype) > 0);

    // Check for scalar
    size_t size = 1;
    
    if (ndims > TENSOR_MAX_DIMS) {
        debug("tensor_dtype_create: failed, ndims=%zu\n", ndims);
        return NULL;
    }
   
    for(int i = 0; i < (int)ndims; i++) {
        size *=dims[i];
    }

    assert(size <= UINT32_MAX);

    // If no size, return NULL
    if (size == 0) {
        debug("tensor_dtype_create: failed, zero size.\n");
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
    t->nvalues = (uint32_t)size;
    for (uint8_t i = 0; i < (uint8_t)ndims; i++) {
        t->dims[i] = dims[i];
    }
    // TODO: Implement the stride logic
    if(ndims > 0) {
        t->stride[ndims-1] = 1;
        for(int8_t i = ((int8_t)ndims) - 2; i >= 0; i--) {
            t->stride[i] = t->dims[i+1] * t->stride[i+1];
        }
    }
    t->dims[ndims] = 0;
    t->op = tensor_op_t::NONE;
    t->id = id;

    t->device = device_type::CPU;
    t->grad_compute = false;
    t->is_transposed = false;
    // Return success
    return t;
}


// Create float32 tensor
tensor_t *tensor_create(tensor_pool_t *pool,tensor_dtype_t dtype, uint32_t num_dims ,uint32_t *dims, void *elems) {
    return tensor_dtype_create(pool, dtype, num_dims, dims, elems);
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
        case tensor_op_t::TRANSPOSE:
            assert(t->a != NULL);
            success = tensor_tranpose_op_matrix(pool,t); 
            break;
        case tensor_op_t::NAIVE_MATRIX_MUL:
            assert(t->a != NULL);
            assert(t->b != NULL);
            assert(t->a->dtype == t->b->dtype);
            success = tensor_mul_op_scalar(pool,t);
            break;
        case tensor_op_t::ADD:
            assert(t->a != NULL);
            assert(t->b != NULL);
            assert(t->a->dtype == t->b->dtype);
            success = tensor_op_add(pool,t);
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

void* tensor_get_data(tensor_t *t) {
    return t->data;
}

uint8_t tensor_get_ndims(tensor_t *t) {
    return t->ndims;
}

uint32_t* tensor_get_dims(tensor_t *t) {
    return t->dims;
}

void tensor_print_data(tensor_t *t) {
    for(uint32_t i = 0; i < t->dims[0]; i++) {
        for(uint32_t j = 0; j < t->dims[1]; j++) {
            uint32_t index = i*t->dims[0] + j;
            std::cout << ((float*)t->data)[index] << " ";
        }
        std::cout << "\n";
    }
}


#include "internal_header.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

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

tensor_t *tensor_dtype_create(tensor_pool_t *pool, tensor_dtype_t dtype, uint32_t ndims,
                              uint32_t *dims, void *elems, bool grad_status) {
    assert(pool != NULL);
    assert(tensor_dtype_sizeof(dtype) > 0);

    // Check for scalar
    size_t size = 1;

    if (ndims > TENSOR_MAX_DIMS) {
        debug("tensor_dtype_create: failed, ndims=%zu\n", ndims);
        return NULL;
    }

    for (int i = 0; i < (int)ndims; i++) {
        size *= dims[i];
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
        debug("tensor_dtype_create: failed, out of memory allocating %ld bytes.\n",
              sizeof(tensor_t));
        return NULL;
    }

    // Allocate memory for data and set it
    t->data = tensor_pool_alloc(pool, size * tensor_dtype_sizeof(dtype), NULL);
    if (t->data == NULL) {
        debug("tensor_dtype_create: failed, out of memory allocating %ld bytes.\n",
              sizeof(tensor_t));
        return NULL;
    } else if (elems != NULL) {
        memcpy(t->data, elems, size * tensor_dtype_sizeof(dtype));
    } else {
        memset(t->data, 0, size * tensor_dtype_sizeof(dtype));
    }

    // set tensor properties
    t->dtype = dtype;
    t->ndims = (uint8_t)ndims;
    t->nvalues = (uint32_t)size;
    for (uint8_t i = 0; i < (uint8_t)ndims; i++) {
        t->dims[i] = dims[i];
    }

    if (ndims > 0) {
        t->stride[ndims - 1] = 1;
        for (int8_t i = ((int8_t)ndims) - 2; i >= 0; i--) {
            t->stride[i] = t->dims[i + 1] * t->stride[i + 1];
        }
    }
    t->dims[ndims] = 0;
    t->op = tensor_op_t::NONE;
    t->id = id;
    t->a = NULL;
    t->b = NULL;
    t->stateTracker = 0;
    t->device = device_type::CPU;
    t->grad_compute = grad_status;
    t->is_transposed = false;
    t->grad = NULL;
    // Return success
    return t;
}

// Create float32 tensor
tensor_t *tensor_create(tensor_pool_t *pool, tensor_dtype_t dtype, uint32_t num_dims,
                        uint32_t *dims, void *elems, bool grad_status) {
    return tensor_dtype_create(pool, dtype, num_dims, dims, elems, grad_status);
}

bool tensor_evaluate(tensor_pool_t *pool, tensor_t *t,  [[maybe_unused]]float *d_a, [[maybe_unused]]float *d_b, [[maybe_unused]]float *d_res) {
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
        success = tensor_mul_op_matrix(pool, t);
        break;
    case tensor_op_t::MUL_SCALAR:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_mul_op_scalar(pool, t);
        break;
    case tensor_op_t::TRANSPOSE:
        assert(t->a != NULL);
        success = tensor_tranpose_op_matrix(pool, t);
        break;
    case tensor_op_t::NAIVE_MATRIX_MUL:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_mul_op_matrix_naive(pool, t);
        break;
    case tensor_op_t::ADD:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_op_add(pool, t);
        break;
    case tensor_op_t::BROADCAST_ADD:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_op_broadcasting_add(pool, t);
        break;
    case tensor_op_t::RELU:
        assert(t->a != NULL);
        success = tensor_op_relu(pool, t);
        break;
    case tensor_op_t::SUB:
        assert(t->a != NULL);
        assert(t->b != NULL);
        assert(t->a->dtype == t->b->dtype);
        success = tensor_op_sub(pool, t);
        break;
    case tensor_op_t::MEAN:
        assert(t->a != NULL);
        success = tensor_op_mean(pool, t);
        break;
    case tensor_op_t::SQUARE:
        assert(t->a != NULL);
        success = tensor_op_square(pool, t);
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

/*************************************************************/
/*************************************************************/
/*************************************************************/
// TODO: Implement GPU module
bool tensor_evaluate_GPU(tensor_pool_t *pool, tensor_t *t,
                          float *d_a, float *d_b, float *d_res) {
    assert(pool != NULL);
    assert(t != NULL);
    bool success = false;

    switch (t->op) {
    case tensor_op_t::NONE:
        success = true;
        break;
    case tensor_op_t::CAST:
        break;
    case tensor_op_t::ADD:
        assert(t->a != NULL && t->b != NULL);
        success = tensor_add_op_cuda(t, d_a, d_b, d_res);
        break;
    case tensor_op_t::SUB:
        assert(t->a != NULL && t->b != NULL);
        success = tensor_sub_op_cuda(t, d_a, d_b, d_res);
        break;
    case tensor_op_t::RELU:
        assert(t->a != NULL);
        success = tensor_relu_op_cuda(t, d_a, d_res);
        break;
    case tensor_op_t::SQUARE:
        assert(t->a != NULL);
        success = tensor_square_op_cuda(t, d_a, d_res);
        break;
    case tensor_op_t::MEAN:
        assert(t->a != NULL);
        success = tensor_mean_op_cuda(t, d_a, d_res);
        break;
    case tensor_op_t::MUL_MATRIX:
        assert(t->a != NULL && t->b != NULL);
        success = tensor_mul_op_cuda(t, d_a, d_b, d_res);
        break;
    case tensor_op_t::NAIVE_MATRIX_MUL:
        assert(t->a != NULL && t->b != NULL);
        success = tensor_mul_op_cuda(t, d_a, d_b, d_res);
        break;
    case tensor_op_t::MUL_SCALAR:
        assert(t->a != NULL && t->b != NULL);
        success = tensor_scalar_mul_op_cuda(t, d_a, d_b, d_res);
        break;
    case tensor_op_t::BROADCAST_ADD:
        assert(t->a != NULL && t->b != NULL);
        success = tensor_broadcast_add_op_cuda(t, d_a, d_b, d_res);
        break;
    case tensor_op_t::TRANSPOSE:
        // Transpose on GPU: fall back to CPU with a single round-trip.
        // Transpose is an O(n) copy the overhead is acceptable.
        // Cause don't know how to do transpose on GPU lol
        if (t->a != nullptr && d_a != nullptr) {
            cudaMemcpy(t->a->data, d_a,
                       t->a->nvalues * sizeof(float), cudaMemcpyDeviceToHost);
        }
        success = tensor_tranpose_op_matrix(pool, t);
        if (success && d_res != nullptr) {
            cudaMemcpy(d_res, t->data,
                       t->nvalues * sizeof(float), cudaMemcpyHostToDevice);
        }
        break;
    default:
        debug("tensor_evaluate_GPU: unhandled op=%d\n", (int)t->op);
        success = false;
    }
    if (success) {
        debug("tensor_evaluate_GPU: success op=%d\n", (int)t->op);
    } else {
        debug("tensor_evaluate_GPU: FUBAR op=%d\n", (int)t->op);
    }
    return success;
}
/*************************************************************/
/*************************************************************/
/*************************************************************/

void *tensor_get_data(tensor_t *t) { return t->data; }

uint8_t tensor_get_ndims(tensor_t *t) { return t->ndims; }

uint32_t *tensor_get_dims(tensor_t *t) { return t->dims; }

void tensor_print_data(tensor_t *t) {
    if (t->ndims == 0) {
        std::cout << ((float *)t->data)[0] << "\n";
        return;
    }
    for (uint32_t i = 0; i < t->dims[0]; i++) {
        for (uint32_t j = 0; j < t->dims[1]; j++) {
            uint32_t index = i * t->dims[1] + j;
            std::cout << ((float *)t->data)[index] << " ";
        }
        std::cout << "\n";
    }
}

 // ((float*)t->data)[i]
bool tensor_fill_random_normal(tensor_t *t, float mean, float std_dev) {
    for(uint32_t i = 0; i < t->nvalues; i += 2) {
        regen1:
        float U1 = (float)(rand())/((float)RAND_MAX +1.0f);
        if (U1 == 0) {
            goto regen1;
        }
        regen2:
        float U2 = (float)(rand())/((float)RAND_MAX +1.0f);
        if (U2 == 0) {
            goto regen2;
        }
        float R1 = sqrtf(-2 * logf(U1));
        float K1 = 2.0f*((float)M_PI)*U2;

        float Z1 = R1*cosf(K1);
        float Z2 = R1*sinf(K1);

        ((float *)t->data)[i] = mean + (std_dev*Z1);
        if (i+1 < t->nvalues) {
            ((float *)t->data)[i+1] = mean + (std_dev*Z2);
        }
    }
    return true;
}

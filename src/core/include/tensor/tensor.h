#ifndef PRIVATE_TENSOR_H
#define PRIVATE_TENSOR_H
#include "soft-cuda/tensor/api.h"
#include <cstdint>
enum class tensor_op_t {
    NONE,
    CAST,             // Cast to a new dtype
    MUL_SCALAR,       // Multiply a by scalar b
    MUL_MATRIX,       // Multiply a by b
    TRANSPOSE,        // Transpose a 2D matrix
    NAIVE_MATRIX_MUL, // Multiply a by b using O(n3) solution
    ADD,              // Add a and b
    BROADCAST_ADD,    // Performs broadcasting during addition
    RELU,             // Activation function
    SUB,              // Subtract two tensor of same shape
    MEAN,             // Returns a scalar value mean of the tensor
};

struct tensor_instance {

    tensor_dtype_t dtype;

    // Np. of dimension and values
    uint8_t ndims;
    uint32_t nvalues;

    // Number of element in each dimension
    uint32_t dims[TENSOR_MAX_DIMS + 1];

    uint32_t stride[TENSOR_MAX_DIMS];

    uint32_t broadcast_stride[TENSOR_MAX_DIMS];
    // Data
    void *data;

    // Operation and arguments
    tensor_op_t op;
    tensor_t *a;
    tensor_t *b;

    // ID and name
    uint32_t id;

    // Tensor aware of it's postion could be useful
    device_type device;

    // If autograd is required
    bool grad_compute;

    // If tensor is transposed
    bool is_transposed;
    
    // Writing it for common interface and ease of access
    // grad is a data-only tensor. Fields op, a, b,
    // stateTracker, grad_compute, grad are always NULL/zero.
    // Only data, nvalues, ndims, dims, dtype are valid.
    // For storing autograd result
    tensor_t *grad;

    // Storing the status if it has been evaluated
    uint8_t stateTracker; // 0-> unvisited, 1-> in processing, 2-> fully processed
};

// Create a new tensor with given data type and dimensions. If dims is NULL then
// a scalar is created. If elems is NULL then the tensor is created with zeros.
// If elems is not NULL then it is assigned as initial values.
tensor_t *tensor_dtype_create(tensor_pool_t *pool, tensor_dtype_t dtype, uint32_t num_dims,
                              uint32_t *dims, void *elems);

// Evaluate the tensor, return true on success
bool tensor_evaluate( tensor_pool_t *pool,tensor_t *t,  float *d_a, float *d_b, float *d_res);

// Return the data type size
size_t tensor_dtype_sizeof(tensor_dtype_t dtype);

tensor_t *tensor_create(tensor_pool_t *pool, tensor_dtype_t dtype, uint32_t num_dims,
                        uint32_t *dims, void *elems);
#endif // !PRIVATE_TENSOR_H

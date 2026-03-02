#pragma once
#include <string>

///////////////////////////////////////////////

// Maximum number of supported dimension
#define TENSOR_MAX_DIMS 8

///////////////////////////////////////////////

// Data type

enum class tensor_dtype_t {
  UINT32_T,
  INT32_T,
  UINT64_T,
  INT64_T,
  FLOAT32_T,
  FLOAT64_T,
};

enum class device_type {
  GPU,
  CPU,
};

// Opaque tensor
typedef struct tensor_instance tensor_t;

// Opaque pool of tensors
typedef struct tensor_pool_instance tensor_pool_t;

// Opaque graph
typedef struct tensor_graph_instance tensor_graph_t;

///////////////////////////////////////////////
// RETURN TENSOR INFORMATION


/* *
 * Return a unique identifier for the tensor
 *
 * @param t            The tensor
 * @return             Return the identifier for the tensor, which will be unique across all 
 *                     other tensors in the same memory pool
 * */
uint32_t tensor_id(tensor_t *t);



////////////////////////////////////////////////
// CREATE TENSOR

/**
 * Create a float64 tensor
 *
 * @param pool        The memory pool to use for creating the tensor operation
 * @param dims        The shape of the tensor. Set to NULL to return a scalar
 *                    value, or provide an array of dimensions for a tensor, with
 *                    the last dimension followed by a zero.
 * @param elems       The elements of the tensor. The number of elements must
 *                    match the number of elements implied by the dimensions,
 *                    or can be NULL to create a tensor with zero values.
 * @return            Returns a tensor or NULL on error. Typically the error will be due
 *                    to insufficient memory in the pool.
 */

tensor_t *tensor_create_float32(tensor_pool_t *pool, uint32_t *dims, float *elems);

///////////////////////////////////////////////////////////////////////////////
// OPERATIONS


/*
 * Free the tensor
 *
 * @param tensor_id   Unique Id of the tensor to free
 *
 * @NOTE this API will be implemented in future for now it's just a place holder
 * We will be working with just BUMP allocator for now.
 * */
void tensor_free(uint32_t tensor_id);


/*
 * Move the memory pool between device_type
 *
 * @param device       Name of the device to move the pool to. GPU/CPU
 * */
uint8_t move_tensor_device(tensor_t *t, device_type target_device);


/*
 * Do matrix multiplication
 * @param out          Pointer to the tensor where result will be stored
 * @param x            Pointer to the tensor which will be mulptiplied
 * @param y            Pointer to the tensor which will be mulptiplied
 *
 * */
void tensor_matmul(tensor_t *out, tensor_t *x, tensor_t *y);


/*
 * Do matrix transpose
 * @param a            Pointer to the tensor which will be transposed
 *
 * */
void tensor_transpose(tensor_t *out, tensor_t *a);


/*
 * Do matrix addition
 * @param out          Pointer to the tensor where result will be stored
 * @param x            Pointer to the tensor which will be added to
 * @param y            Pointer to the tensor which will be added
 *
 * */
void tensor_add(tensor_t *out, tensor_t *x, tensor_t *y);


/*
 * Do scalar matrix multiplication
 * @param out          Pointer to the tensor where result will be stored
 * @param x            Pointer to the tensor which will be mulptiplied
 * @param y            Scalar with which the matrix will be multiplied
 *
 * */
void tensor_scalar_mul(tensor_t *out, tensor_t *x, double y);

// The activation function
// @params out                     Output tensor
// @param  a                       Input tensor

void tensor_relu(tensor_t *out, tensor_t *a);

// Compares result
// return how correct we were b/w 0-1
tensor_t* tensor_mse_loss(tensor_t *predictions, tensor_t *target);

//////////////////////
// THE BACKWARD PASS

// The trigger function
void tensor_backward(tensor_t *loss_matrix);

// The Optimizer
void tensor_sgd_template(tensor_t *m, double learning_rate);

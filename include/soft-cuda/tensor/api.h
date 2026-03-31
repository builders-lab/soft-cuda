#pragma once
#include <cstdint>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////////

// Maximum number of supported dimension
#define TENSOR_MAX_DIMS 8

///////////////////////////////////////////////

// Data type

// DONE
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

// DONE
//  Opaque tensor
typedef struct tensor_instance tensor_t;

// DONE
//  Opaque pool of tensors
typedef struct tensor_pool_instance tensor_pool_t;

typedef struct execution_node execution_node_t;

/*******************************************************************************
 * !!!!!!!! DISCARDED !!!!!!!!
 * Discarding this in favour of vector of execution_node
 *
 * Opaque graph
 * typedef struct tensor_graph_instance tensor_graph_t;
 * *****************************************************************************
 */
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
// DONE
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
 *
 * @param dtype       The data type of tensor objects.
 * @param num_dims    The rank of the tensor
 * @param elems       The elements of the tensor. The number of elements must
 *                    match the number of elements implied by the dimensions,
 *                    or can be NULL to create a tensor with zero values.
 * @return            Returns a tensor or NULL on error. Typically the error will be due
 *                    to insufficient memory in the pool.
 */
// DONE
tensor_t *tensor_create(tensor_pool_t *pool, tensor_dtype_t dtype, uint32_t num_dims,
                        uint32_t *dims, void *elems);

/*
 * Establish a new memory arena
 * @param capacity_bytes  The total size of the raw memory block to pre-allocate
 * @return                Pointer to the opaque pool instance or NULL on failure
 */
// DONE
tensor_pool_t *tensor_pool_create(size_t capacity_bytes, bool isOfDevice = false);

///////////////////////////////////////////////////////////////////////////////
// OPERATIONS

/*
 * Free the tensor
 *
 * @param pool   Address of the tensor to free
 *
 * We will be working with just BUMP allocator for now.
 *
 * Instantly invalidates all tensors in the pool by resetting the bump pointer to zero.
 * Does NOT return memory to the OS. Highly efficient for the training loop.
 */
// DONE
void tensor_pool_zero(tensor_pool_t *pool);

/*
 * Completely destroys the arena and returns the memory to the system.
 */
// DONE
void tensor_pool_destroy(tensor_pool_t *pool);

// Allocate bytes on the pool, return NULL if memory exhausted
// DONE
void *tensor_pool_alloc(tensor_pool_t *pool, size_t size, uint32_t *id);

// Return size of memory pool
// DONE
size_t tensor_pool_size(tensor_pool_t *pool);
// Return used bytes of memory pool
// DONE
size_t tensor_pool_used(tensor_pool_t *pool);

////////////////////////////////////////////////////////////////////////////////////////////
// WILL LIKELY BE DEPRECEATED
/*
 * Move the memory pool between device_type
 *
 * @param device       Name of the device to move the pool to. GPU/CPU @param t            The
 * tensor to move
 * @pool               Pool where the tensor will be allocated
 * */

// DEPRECEATED
// bool tensor_move_device(tensor_t *t, device_type target_device, tensor_pool_t *pool);
// NEW IMPLEMENTATION

// DONE
bool execution_node_to_host(execution_node_t *node);
////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Fetches the data of the tensor.
 * @return           Returns a void pointer to the array
 * */
// DONE
void *tensor_get_data(tensor_t *t);

/*
 * @return returns the dimension of the tensor
 * */
// DONE
uint8_t tensor_get_ndims(tensor_t *t);

/*
 * Get the dimension of the tensor
 * */
// DONE
uint32_t *tensor_get_dims(tensor_t *t);

/*
 * Print the tensor data
 * */
// DONE
void tensor_print_data(tensor_t *t);

/*
 * Do cache optimized matrix multplication multiplication
 * @note               Implicitly handle sclar and matrix
 *                     multplication, for cache optimization calls
 *                     tensor_transpose function
 * @param pool         Pointer to the result will be stored
 * @param x            Pointer to the tensor which will be mulptiplied
 * @param y            Pointer to the tensor which will be mulptiplied
 * @return             Returns a tensor object with operation set.
 *
 * @Note               tensor_matmul expects B to be transposed and
 *                     contiguous. Call tensor_transpose(B) first.
 * */
// DONE
tensor_t *tensor_mul(tensor_pool_t *pool, tensor_t *x, tensor_t *y);

/* The naive version of matrix multiplication
 * @param pool       Pointer to the tensor pool
 * @param x            Pointer to the tensor which will be mulptiplied
 * @param y            Pointer to the tensor which will be mulptiplied
 *
 * @return             Returns a tensor object with operation set.
 * */

// DONE
tensor_t *tensor_mul_naive(tensor_pool_t *pool, tensor_t *x, tensor_t *y);

/*
 * Do matrix transpose
 * @param a            Pointer to the tensor which will be transposed
 *
 * @return             Returns a tensor object with operation set.
 * */
// DONE
tensor_t *tensor_transpose(tensor_pool_t *pool, tensor_t *a);

/*
 * Do matrix addition
 * @param out          Pointer to the tensor where result will be stored
 * @param x            Pointer to the tensor which will be added to
 * @param y            Pointer to the tensor which will be added
 *
 * @return             Returns a tensor object with operation set.
 * */
// DONE
tensor_t *tensor_add(tensor_pool_t *pool, tensor_t *x, tensor_t *y);

// Explicit broadcasting for layers (e.g., Y = XW + b)
// DONE
tensor_t *tensor_add_bias(tensor_pool_t *pool, const tensor_t *xw, const tensor_t *bias);

/////////////////////////////////////////////////////////////
/// DEPRECEATED tensor_mul operation handles it automatically
// /*
//  * Do scalar matrix multiplication
//  * @param out          Pointer to the tensor where result will be stored
//  * @param x            Pointer to the tensor which will be mulptiplied
//  * @param y            Scalar with which the matrix will be multiplied
//  *
//  * */
// // void tensor_scalar_mul(tensor_t *out, tensor_t *x, double y);
//
// upon request it can be exposed seperately
//////////////////////////////////////////////////////////////

// The activation function
// @params out                     Output tensor
// @param  a                       Input tensor

// DONE
tensor_t *tensor_relu(tensor_pool_t *pool, tensor_t *a);

// Compares result
// return how correct we were b/w 0-1
tensor_t *tensor_mse_loss(tensor_pool_t *pool, tensor_t *predictions, tensor_t *target);

// Fills an existing tensor with normally distributed random numbers.
bool tensor_fill_random_normal(tensor_t *t, float mean, float std_dev);

// Fused operation combining Softmax and Cross-Entropy for stability
tensor_t *tensor_cross_entropy_loss(tensor_pool_t *pool, const tensor_t *predictions,
                                    const tensor_t *targets);

// ***********************************************************************************
// TODO: HAVE TO UPDATED FUNC SIG SPEC 
// Evalutes the operation(Forward) with depth=1
// @return             boolean flag for status
// DONE
bool tensor_evaluate( tensor_pool_t *pool,tensor_t *t,  float *d_a, float *d_b, float *d_res);
// DONE
bool tensor_evaluate_GPU( tensor_pool_t *pool,tensor_t *t,  float *d_a, float *d_b, float *d_res);

/////////////////////////////////////////////////////////////
// NO DESIGNING DONE HENCE NOT RECOMMENDED TO WORK AROUND USE IT JUST AS PLACEHOLDER BUT BE READY TO
// UPDATE API below it is still unstable
//////////////////////
// THE BACKWARD PASS

// Evalutes the operation(BACKWARD) with depth=1
// @return             boolean flag for status
bool tensor_backward(tensor_pool_t *pool, tensor_t *t);

// The Optimizer
void tensor_sgd_template(tensor_pool_t *static_weights_pool, double learning_rate);

// API UNSTABLE
/////////////////////////////////////////////////////////////
// GRAPH OPERATIONS


/////////////////////////////////////////////////////////////
// DEPRECEATED verifyIfDAG DOES THIS AND BETTER
/*
 * Creates Tensor graph struct.
 *
 * @params graph_pool     Takes tensor pool to store the graph.
 * @return tensor_graph_t Graph struct.
 *
 * Is HIGHLY RECOMMENDED to make seperate pool for graph.
 * */
// tensor_graph_t *tensor_graph_create(tensor_pool_t *graph_pool);
/////////////////////////////////////////////////////////////////


/* !!!!! SIGNATURE WAS CHANGED !!!!!
 * Topologically sort the tensors, detect dependency and return a vector of .
 * 
 * @params pool       The data pool where the execution nodes will be 
 *                    stored RECOMMENDED/ADVISED to give different memory pool.
 * @params t          The tensor from where you want to build graph
 *                    doesn't take ops after this into account.
 * @params seq        vector refrence for storing sorted tensors.
 *
 * @return            Boolean status flag
 * */
// DONE
bool verifyIfDAG(tensor_pool_t *pool, tensor_t *t, std::vector<execution_node_t *> &seq);


/* This function marks the tensor for transfer to GPU, This also assigns the space on VRAM.
 * 
 * @params pool       The VRAM pool where GPU ops data is stored,
 *                    only data is transfered not whole tensor,
 *                    ADVISED/RECOMMENDED/COMPUSLORY to give different memory pool.
 * @params nodes      The vector of execution_node_t in which original data was stored passed by refrence.
 *
 * @return void       Doesn't return anything.
 * */
// DONE
void assignBackendGraph(tensor_pool_t *pool,std::vector<execution_node_t *> &nodes);

/* @params  Take execution_node_t which you wanna know
 * @returns the postion of the execution_node_t after verifyIfDAG
 * */
// DONE
int32_t getPosOfNode(execution_node_t *et);

// Prints all data about execution_node_t as well as tensor data
// DONE
void printExecutionNode(execution_node_t *et);
// Previous SIGNATURE
// bool tensor_graph_build(tensor_graph_t *g, tensor_t *t);

/*
 * Evaluate the whole graph forward operation.
 * @params            graph struct for ops sequence
 * @return            boolean status flag
 * */
bool tensor_graph_forward_evaluate(tensor_pool_t *pool_cpu, tensor_pool_t *pool_gpu, std::vector<execution_node_t *> &nodes);

/*
 * Evaluate the whole graph backward operation.
 * @params            graph struct for ops sequence
 * @return            boolean status flag
 * */
bool tensor_graph_backward_evaluate(tensor_graph_t *g);

//////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

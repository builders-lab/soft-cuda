#pragma once
#include <cstdint>
#include <string>
#include <vector>

// #ifdef __cplusplus
// extern "C" {
// #endif

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
// Added this for assignBackend function so that we can adjust according to user requirements
enum class backend_mode {
    GPU,
    CPU,
    HYBRID,
};
// DONE
//  Opaque tensor
typedef struct tensor_instance tensor_t;

// DONE
//  Opaque pool of tensors
typedef struct tensor_pool_instance tensor_pool_t;

typedef struct execution_node execution_node_t;

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
                        uint32_t *dims, void *elems, bool grad_status = true);

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

/*
 * Move the node from device to host
 *
 * @param  node        The node which we want to move from device to host
 * */
// DONE
bool execution_node_to_host(execution_node_t *node);


/*
 * Fetches the data of the tensor.
 * @return           Returns a void pointer to the data array
 * */
// DONE
void *tensor_get_data(tensor_t *t);

/*
 * @return the dimension of the tensor
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

/*
 * @param pool         Takes the tensor pool.
 * @param x            The tensor we want to square
 *
 * @return             Squared tensor object
 * */

// DONE
tensor_t *tensor_square(tensor_pool_t *pool, tensor_t *x);

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
tensor_t *tensor_add_bias(tensor_pool_t *pool, tensor_t *xw, tensor_t *bias);

/*
 * Do matrix subtraction
 * @param out          Pointer to the tensor where result will be stored
 * @param x            Pointer to the tensor which will be subtracted from 
 * @param y            Pointer to the tensor which will be subtracted
 *
 * @return             Returns a tensor object with operation set.
 * */
// DONE
tensor_t *tensor_sub(tensor_pool_t *pool, tensor_t *a, tensor_t *b);

// The activation function
// @params out                     Output tensor
// @param  a                       Input tensor

// DONE
tensor_t *tensor_relu(tensor_pool_t *pool, tensor_t *a);


// Calculates the mean
// @params pool                    Tensor pool
// @a                              Tensor whose mean we want to find
//
// @return                         Returns the mean of the tensor
// DONE
tensor_t *tensor_mean(tensor_pool_t *pool, tensor_t *a);

// Compares result
// return how correct we were b/w 0-1

// DONE
tensor_t *tensor_mse_loss(tensor_pool_t *pool, tensor_t *predictions, tensor_t *target);

// Fills an existing tensor with normally distributed random numbers.
// @params  t                      Tensor we want to fill with random no.
// @params  mean                   The mean around which we want to have the value
// @params  std_dev                Standard deviation

// DONE
bool tensor_fill_random_normal(tensor_t *t, float mean, float std_dev);


// TODO: HAVE TO DO THIS IF WE WANT CLASSIFICATION
// // Fused operation combining Softmax and Cross-Entropy for stability
// tensor_t *tensor_cross_entropy_loss(tensor_pool_t *pool, const tensor_t *predictions,
//                                     const tensor_t *targets);

// ***********************************************************************************
// Evalutes the operation(Forward) with depth=1
// @return             boolean flag for status
//
// @note               This is atomic operations and is used internally
//                     is for CPU computation
// DONE
bool tensor_evaluate( tensor_pool_t *pool,tensor_t *t,  float *d_a = nullptr, float *d_b = nullptr, float *d_res= nullptr);

// Is for GPU computation
// DONE
bool tensor_evaluate_GPU( tensor_pool_t *pool,tensor_t *t,  float *d_a, float *d_b, float *d_res);

//////////////////////
// THE BACKWARD PASS

// Evalutes the operation(BACKWARD) with depth=1
// @return             boolean flag for status
// @note               Inplictly handle the GPU CPU transfer is a single operation
bool tensor_backward(tensor_pool_t *pool, tensor_t *t);

// The Optimizer
// @params nodes       The graph
// @learning_rate      The size of step
//
void tensor_sgd(std::vector<execution_node_t *> &nodes, float learning_rate);
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


/* 
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
 * @params pool_gpu       The VRAM pool where GPU ops data is stored,
 *                    only data is transfered not whole tensor,
 *                    ADVISED/RECOMMENDED/COMPUSLORY to give different memory pool.
 * @params nodes      The vector of execution_node_t in which original data was stored passed by refrence.
 *
 * @return void       Doesn't return anything.
 * */
// DONE
void assignBackendGraph(tensor_pool_t *pool_gpu,std::vector<execution_node_t *> &nodes, backend_mode value = backend_mode::CPU);

/* This function is used for assigining temporary storage on GPU VRAM for GPU -> GPU part so we don't have 
 * transfer overhead and thrashing
 *
 * @params pool_grad_cpu             Pool for CPU grad allocation
 * @params pool_grad_gpu             Pool for GPU grad allocation
 * 
 * @params nodes                     The graph
 * */
//DONE
void assignGradMemory(tensor_pool_t *pool_grad_cpu, tensor_pool_t *pool_grad_gpu, std::vector<execution_node_t *> &nodes);


/* @params  Take execution_node_t which you wanna know
 * @returns the postion of the execution_node_t after verifyIfDAG
 * */
// DONE
int32_t getPosOfNode(execution_node_t *et);

// Prints all data about execution_node_t as well as tensor data
// internally calls tensor print hence prints tensor data too

// DONE
void printExecutionNode(execution_node_t *et);


/*
 * Evaluate the whole graph forward operation.
 * @params            graph struct for ops sequence
 * @return            boolean status flag
 * */
bool tensor_graph_forward_evaluate(tensor_pool_t *pool_cpu, tensor_pool_t *pool_gpu, std::vector<execution_node_t *> &nodes);

//////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// BACKWARD PASS FUNCTIONS

// Takes nodes and then sets the grads data to 0 using memset
void gradInitializer(std::vector<execution_node_t *> &nodes);

// Walk over the graph(nodes) and then run autograd
bool tensor_graph_backward(std::vector<execution_node_t *> &nodes);

// Used to transfer the gradient data from device to host
void autogradGpuMemTranfer(std::vector<execution_node_t *> &nodes);

bool save_model(const std::string& filepath, const std::vector<tensor_t*>& weights);
bool load_model(const std::string& filepath, const std::vector<tensor_t*>& weights);

// #ifdef __cplusplus
// }
// #endif

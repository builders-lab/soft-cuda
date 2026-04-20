/**
 * @file tensor_graph.h
 * @brief Flat-C graph building and training primitives — Python/ctypes compatible.
 *
 * Two-layer API:
 *   Layer 1 — low-level primitives that mirror the internal C++ graph functions.
 *   Layer 2 — high-level convenience wrappers for typical Python usage.
 *
 * Key design: sc_graph_t is an opaque handle that wraps
 * std::vector<execution_node_t*> so Python never sees C++ containers.
 *
 * Include via soft_cuda_python.h — do not include directly.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "tensor_pool.h"
#include "tensor_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Backend mode constants  (mirrors backend_mode enum class)
 * ----------------------------------------------------------------------- */
#define SC_BACKEND_GPU    0
#define SC_BACKEND_CPU    1
#define SC_BACKEND_HYBRID 2

/* -----------------------------------------------------------------------
 * Opaque graph handle
 * Internally holds a std::vector<execution_node_t*> allocated on the heap.
 * ----------------------------------------------------------------------- */
typedef struct sc_graph sc_graph_t;

/* -----------------------------------------------------------------------
 * Layer 1 — low-level graph primitives
 * ----------------------------------------------------------------------- */

/**
 * Allocate a new, empty graph handle.
 * Must be freed with sc_graph_destroy when no longer needed.
 *
 * @return  New sc_graph_t*, or NULL on allocation failure.
 */
sc_graph_t *sc_graph_create(void);

/**
 * Free the graph handle and the internal node vector.
 * Does NOT free the tensors or pools themselves.
 *
 * @param g  Graph to destroy.
 */
void sc_graph_destroy(sc_graph_t *g);

/**
 * Topologically sort the tensor computation graph starting from tensor `t`
 * and populate the graph handle with execution nodes.
 * Detects cycles (returns 0 if a cycle is found).
 *
 * @param meta_pool  Pool used to allocate execution_node_t objects.
 *                   Recommended: a dedicated metadata pool.
 * @param t          Root tensor (e.g. loss node).
 * @param g          Graph handle to populate.
 * @return           Non-zero on success (valid DAG), 0 if cycle detected.
 */
int sc_verify_dag(sc_pool_t *meta_pool, sc_tensor_t *t, sc_graph_t *g);

/**
 * Assign a backend (CPU / GPU / HYBRID) to each node in the graph.
 * For GPU nodes, allocates device memory in pool_gpu.
 *
 * @param pool_gpu  GPU VRAM pool for device memory allocation.
 * @param g         Populated graph handle.
 * @param mode      One of SC_BACKEND_CPU / SC_BACKEND_GPU / SC_BACKEND_HYBRID.
 */
void sc_assign_backend(sc_pool_t *pool_gpu, sc_graph_t *g, int mode);

/**
 * Allocate gradient memory for each trainable node.
 * CPU grad tensors go into pool_grad_cpu; GPU grad buffers into pool_grad_gpu.
 *
 * @param pool_grad_cpu  CPU pool for gradient tensors.
 * @param pool_grad_gpu  GPU pool for gradient device buffers.
 * @param g              Graph handle.
 */
void sc_assign_grad_memory(sc_pool_t *pool_grad_cpu,
                           sc_pool_t *pool_grad_gpu,
                           sc_graph_t *g);

/**
 * Run the full forward pass over the graph (all nodes, in topological order).
 * Handles CPU↔GPU data transfer as needed by the assigned backends.
 *
 * @param pool_cpu  CPU pool (for CPU-side evaluate calls).
 * @param pool_gpu  GPU pool (metadata, used by GPU evaluate).
 * @param g         Graph handle.
 * @return          Non-zero on success.
 */
int sc_graph_forward(sc_pool_t *pool_cpu, sc_pool_t *pool_gpu, sc_graph_t *g);

/**
 * Copy GPU gradient data back to CPU host memory after the forward pass.
 * Must be called before sc_backward if any nodes ran on GPU.
 *
 * @param g  Graph handle.
 */
void sc_autograd_gpu_transfer(sc_graph_t *g);

/**
 * Zero out all gradient buffers in the graph (call at the start of each step).
 *
 * @param g  Graph handle.
 */
void sc_grad_initializer(sc_graph_t *g);

/**
 * Run backward pass (autograd) over the entire graph.
 *
 * @param g  Graph handle.
 * @return   Non-zero on success.
 */
int sc_backward(sc_graph_t *g);

/**
 * Stochastic gradient descent update for all leaf (op == NONE) trainable tensors.
 *
 * @param g             Graph handle.
 * @param learning_rate Step size for parameter update.
 */
void sc_sgd(sc_graph_t *g, float learning_rate);

/**
 * Copy the data of a specific node from device (GPU) to host (CPU).
 * Useful to read a specific intermediate result after the forward pass.
 *
 * @param g         Graph handle.
 * @param node_idx  Index of the node in the topological order (0-based).
 * @return          Non-zero on success.
 */
int sc_node_to_host(sc_graph_t *g, size_t node_idx);

/* -----------------------------------------------------------------------
 * Layer 2 — high-level convenience API
 * ----------------------------------------------------------------------- */

/**
 * Build a ready-to-train graph from a loss tensor in one call.
 * Internally calls sc_verify_dag → sc_assign_backend → sc_assign_grad_memory.
 *
 * Caller must still supply pool_grad_cpu and pool_grad_gpu for gradient
 * memory (they are passed through to sc_assign_grad_memory).
 *
 * @param meta_pool      Pool for execution_node_t allocation.
 * @param pool_gpu       GPU VRAM pool (device data + grad buffers).
 * @param pool_grad_cpu  CPU pool for gradient tensors.
 * @param pool_grad_gpu  GPU pool for gradient device buffers.
 * @param loss           Root loss tensor (usually the output of sc_tensor_mse_loss).
 * @param backend_mode   One of SC_BACKEND_* constants.
 * @return               Ready sc_graph_t*, or NULL on failure.
 */
sc_graph_t *sc_build_graph(sc_pool_t *meta_pool,
                           sc_pool_t *pool_gpu,
                           sc_pool_t *pool_grad_cpu,
                           sc_pool_t *pool_grad_gpu,
                           sc_tensor_t *loss,
                           int backend_mode);

/**
 * Execute a single full training step:
 *   forward → autograd_gpu_transfer → grad_initializer → backward → sgd
 *
 * @param pool_cpu       CPU execution pool.
 * @param pool_gpu       GPU execution pool.
 * @param g              Graph handle.
 * @param learning_rate  SGD step size.
 */
void sc_graph_step(sc_pool_t *pool_cpu,
                   sc_pool_t *pool_gpu,
                   sc_graph_t *g,
                   float learning_rate);

/**
 * Read the scalar loss value from the last node in the graph.
 * Returns NaN if the last node is not a scalar or has not been evaluated.
 *
 * @param g  Graph handle.
 * @return   float loss value.
 */
float sc_graph_get_loss(sc_graph_t *g);

/**
 * Return the number of nodes in the graph (useful for iteration / debugging).
 *
 * @param g  Graph handle.
 * @return   Node count.
 */
size_t sc_graph_size(sc_graph_t *g);

#ifdef __cplusplus
} /* extern "C" */
#endif

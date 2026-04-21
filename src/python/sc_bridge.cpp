/* ═══════════════════════════════════════════════════════════════════════════
 *  Written by  : Antigravity (AI Coding Assistant)
 *  Date        : 2026-04-21  02:12 IST
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @file sc_bridge.cpp
 * @brief Implementation of the soft-cuda Python C bridge.
 *
 * Compiled as C++ (so it can call the internal C++ API that uses
 * std::vector, enum class, etc.), but all exported symbols have C linkage
 * so Python ctypes / cffi can dlopen and call them by name.
 *
 * Build as a shared library:
 *   cmake --build build --target soft_cuda_python
 *
 * Every exported function is declared in:
 *   include/soft-cuda/python/soft_cuda_python.h
 *
 * The internal C++ API lives in:
 *   include/soft-cuda/tensor/api.h   (referenced via internal_header.h)
 */

// Pull in everything: public + private headers & all internal APIs
#include "../include/soft-cuda/tensor/api.h"
#include "../include/soft-cuda/python/soft_cuda_python.h"
#include "internal_header.h"


#include <vector>
#include <cstring>   /* memcpy */
#include <cmath>     /* NAN */
#include <cassert>
#include <cstdlib>   /* malloc/free */
#include <string>

/* ========================================================================
 * sc_graph_t — opaque handle wrapping std::vector<execution_node_t*>
 * ========================================================================
 * We heap-allocate the struct itself via malloc so it has a stable address
 * and no C++ constructors are visible at the ABI boundary.
 */
struct sc_graph {
    std::vector<execution_node_t *> nodes;
};

/* ========================================================================
 * Helper: cast int backend constant → backend_mode enum class
 * ======================================================================== */
static backend_mode to_backend_mode(int mode) {
    switch (mode) {
    case 0:  return backend_mode::GPU;
    case 2:  return backend_mode::HYBRID;
    default: return backend_mode::CPU;
    }
}

/* ========================================================================
 * Helper: cast int dtype constant → tensor_dtype_t enum class
 * ======================================================================== */
static tensor_dtype_t to_tensor_dtype(int d) {
    switch (d) {
    case 0:  return tensor_dtype_t::UINT32_T;
    case 1:  return tensor_dtype_t::INT32_T;
    case 2:  return tensor_dtype_t::UINT64_T;
    case 3:  return tensor_dtype_t::INT64_T;
    case 5:  return tensor_dtype_t::FLOAT64_T;
    default: return tensor_dtype_t::FLOAT32_T;  /* SC_DTYPE_FLOAT32 = 4 */
    }
}

/* ------------------------------------------------------------------------ */
/*  Pool wrappers                                                            */
/* ------------------------------------------------------------------------ */

extern "C" sc_pool_t *sc_pool_create(size_t capacity_bytes, int on_device) {
    return tensor_pool_create(capacity_bytes, (bool)on_device);
}

extern "C" void sc_pool_destroy(sc_pool_t *pool) {
    tensor_pool_destroy(pool);
}

extern "C" void sc_pool_zero(sc_pool_t *pool) {
    tensor_pool_zero(pool);
}

extern "C" void *sc_pool_alloc(sc_pool_t *pool, size_t size, uint32_t *out_id) {
    return tensor_pool_alloc(pool, size, out_id);
}

extern "C" size_t sc_pool_size(sc_pool_t *pool) {
    return tensor_pool_size(pool);
}

extern "C" size_t sc_pool_used(sc_pool_t *pool) {
    return tensor_pool_used(pool);
}

/* ------------------------------------------------------------------------ */
/*  Tensor core wrappers                                                     */
/* ------------------------------------------------------------------------ */

extern "C" sc_tensor_t *sc_tensor_create(sc_pool_t *pool,
                                         int dtype,
                                         uint32_t num_dims,
                                         uint32_t *dims,
                                         void *elems,
                                         int grad) {
    return tensor_create(pool,
                         to_tensor_dtype(dtype),
                         num_dims,
                         dims,
                         elems,
                         (bool)grad);
}

extern "C" uint32_t sc_tensor_id(sc_tensor_t *t) {
    return tensor_id(t);
}

extern "C" void *sc_tensor_get_data(sc_tensor_t *t) {
    return tensor_get_data(t);
}

extern "C" uint8_t sc_tensor_get_ndims(sc_tensor_t *t) {
    return tensor_get_ndims(t);
}

extern "C" uint32_t *sc_tensor_get_dims(sc_tensor_t *t) {
    return tensor_get_dims(t);
}

extern "C" void sc_tensor_print_data(sc_tensor_t *t) {
    tensor_print_data(t);
}

extern "C" int sc_tensor_fill_random_normal(sc_tensor_t *t,
                                            float mean,
                                            float std_dev) {
    return (int)tensor_fill_random_normal(t, mean, std_dev);
}

/* ------------------------------------------------------------------------ */
/*  Op wrappers                                                              */
/* ------------------------------------------------------------------------ */

extern "C" sc_tensor_t *sc_tensor_mul(sc_pool_t *pool,
                                      sc_tensor_t *x,
                                      sc_tensor_t *y) {
    return tensor_mul(pool, x, y);
}

extern "C" sc_tensor_t *sc_tensor_mul_naive(sc_pool_t *pool,
                                            sc_tensor_t *x,
                                            sc_tensor_t *y) {
    return tensor_mul_naive(pool, x, y);
}

extern "C" sc_tensor_t *sc_tensor_add(sc_pool_t *pool,
                                      sc_tensor_t *x,
                                      sc_tensor_t *y) {
    return tensor_add(pool, x, y);
}

extern "C" sc_tensor_t *sc_tensor_add_bias(sc_pool_t *pool,
                                           sc_tensor_t *xw,
                                           sc_tensor_t *bias) {
    return tensor_add_bias(pool, xw, bias);
}

extern "C" sc_tensor_t *sc_tensor_sub(sc_pool_t *pool,
                                      sc_tensor_t *a,
                                      sc_tensor_t *b) {
    return tensor_sub(pool, a, b);
}

extern "C" sc_tensor_t *sc_tensor_relu(sc_pool_t *pool, sc_tensor_t *a) {
    return tensor_relu(pool, a);
}

extern "C" sc_tensor_t *sc_tensor_mean(sc_pool_t *pool, sc_tensor_t *a) {
    return tensor_mean(pool, a);
}

extern "C" sc_tensor_t *sc_tensor_mse_loss(sc_pool_t *pool,
                                           sc_tensor_t *predictions,
                                           sc_tensor_t *target) {
    return tensor_mse_loss(pool, predictions, target);
}

extern "C" sc_tensor_t *sc_tensor_square(sc_pool_t *pool, sc_tensor_t *x) {
    return tensor_square(pool, x);
}

extern "C" sc_tensor_t *sc_tensor_transpose(sc_pool_t *pool, sc_tensor_t *a) {
    return tensor_transpose(pool, a);
}

extern "C" int sc_tensor_evaluate(sc_pool_t *pool, sc_tensor_t *t) {
    /* Pass null device pointers — CPU path */
    return (int)tensor_evaluate(pool, t, nullptr, nullptr, nullptr);
}

extern "C" int sc_tensor_evaluate_gpu(sc_pool_t *pool,
                                      sc_tensor_t *t,
                                      float *d_a,
                                      float *d_b,
                                      float *d_res) {
    return (int)tensor_evaluate_GPU(pool, t, d_a, d_b, d_res);
}

/* ------------------------------------------------------------------------ */
/*  Graph — Layer 1                                                          */
/* ------------------------------------------------------------------------ */

extern "C" sc_graph_t *sc_graph_create(void) {
    sc_graph_t *g = (sc_graph_t *)malloc(sizeof(sc_graph_t));
    if (g == nullptr) return nullptr;
    /* Placement-new to properly construct the std::vector */
    new (&g->nodes) std::vector<execution_node_t *>();
    return g;
}

extern "C" void sc_graph_destroy(sc_graph_t *g) {
    if (g == nullptr) return;
    /* Explicitly destroy the vector before freeing the raw memory */
    g->nodes.~vector();
    free(g);
}

extern "C" int sc_verify_dag(sc_pool_t *meta_pool,
                              sc_tensor_t *t,
                              sc_graph_t *g) {
    if (g == nullptr) return 0;
    return (int)verifyIfDAG(meta_pool, t, g->nodes);
}

extern "C" void sc_assign_backend(sc_pool_t *pool_gpu,
                                  sc_graph_t *g,
                                  int mode) {
    if (g == nullptr) return;
    assignBackendGraph(pool_gpu, g->nodes, to_backend_mode(mode));
}

extern "C" void sc_assign_grad_memory(sc_pool_t *pool_grad_cpu,
                                      sc_pool_t *pool_grad_gpu,
                                      sc_graph_t *g) {
    if (g == nullptr) return;
    assignGradMemory(pool_grad_cpu, pool_grad_gpu, g->nodes);
}

extern "C" int sc_graph_forward(sc_pool_t *pool_cpu,
                                sc_pool_t *pool_gpu,
                                sc_graph_t *g) {
    if (g == nullptr) return 0;
    return (int)tensor_graph_forward_evaluate(pool_cpu, pool_gpu, g->nodes);
}

extern "C" void sc_autograd_gpu_transfer(sc_graph_t *g) {
    if (g == nullptr) return;
    autogradGpuMemTranfer(g->nodes);
}

extern "C" void sc_grad_initializer(sc_graph_t *g) {
    if (g == nullptr) return;
    gradInitializer(g->nodes);
}

extern "C" int sc_backward(sc_graph_t *g) {
    if (g == nullptr) return 0;
    return (int)tensor_graph_backward(g->nodes);
}

extern "C" void sc_sgd(sc_graph_t *g, float learning_rate) {
    if (g == nullptr) return;
    tensor_sgd(g->nodes, learning_rate);
}

extern "C" int sc_node_to_host(sc_graph_t *g, size_t node_idx) {
    if (g == nullptr || node_idx >= g->nodes.size()) return 0;
    return (int)execution_node_to_host(g->nodes[node_idx]);
}

/* ------------------------------------------------------------------------ */
/*  Graph — Layer 2 (convenience wrappers)                                  */
/* ------------------------------------------------------------------------ */

extern "C" sc_graph_t *sc_build_graph(sc_pool_t *meta_pool,
                                      sc_pool_t *pool_gpu,
                                      sc_pool_t *pool_grad_cpu,
                                      sc_pool_t *pool_grad_gpu,
                                      sc_tensor_t *loss,
                                      int backend_mode_val) {
    sc_graph_t *g = sc_graph_create();
    if (g == nullptr) return nullptr;

    if (!verifyIfDAG(meta_pool, loss, g->nodes)) {
        sc_graph_destroy(g);
        return nullptr;
    }

    assignBackendGraph(pool_gpu, g->nodes, to_backend_mode(backend_mode_val));
    assignGradMemory(pool_grad_cpu, pool_grad_gpu, g->nodes);

    return g;
}

extern "C" void sc_graph_step(sc_pool_t *pool_cpu,
                              sc_pool_t *pool_gpu,
                              sc_graph_t *g,
                              float learning_rate) {
    if (g == nullptr) return;
    tensor_graph_forward_evaluate(pool_cpu, pool_gpu, g->nodes);
    autogradGpuMemTranfer(g->nodes);
    gradInitializer(g->nodes);
    tensor_graph_backward(g->nodes);
    tensor_sgd(g->nodes, learning_rate);
}

extern "C" float sc_graph_get_loss(sc_graph_t *g) {
    if (g == nullptr || g->nodes.empty()) return NAN;

    execution_node_t *last = g->nodes.back();
    if (last == nullptr || last->t == nullptr) return NAN;

    /* Bring the loss node back to host if it ran on GPU */
    if (last->t->device == device_type::GPU) {
        execution_node_to_host(last);
    }

    if (last->t->data == nullptr) return NAN;

    /* Loss node must be a scalar (1 element) */
    float val;
    memcpy(&val, last->t->data, sizeof(float));
    return val;
}

extern "C" size_t sc_graph_size(sc_graph_t *g) {
    if (g == nullptr) return 0;
    return g->nodes.size();
}

/* ------------------------------------------------------------------------ */
/*  IO wrappers                                                              */
/* ------------------------------------------------------------------------ */

extern "C" int sc_save_model(const char *path,
                             sc_tensor_t **tensors,
                             size_t count) {
    if (path == nullptr || tensors == nullptr) return 0;
    std::vector<tensor_t *> vec(tensors, tensors + count);
    return (int)save_model(std::string(path), vec);
}

extern "C" int sc_load_model(const char *path,
                             sc_tensor_t **tensors,
                             size_t count) {
    if (path == nullptr || tensors == nullptr) return 0;
    std::vector<tensor_t *> vec(tensors, tensors + count);
    return (int)load_model(std::string(path), vec);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Written by  : Antigravity (AI Coding Assistant)
 *  Date        : 2026-04-21  02:12 IST
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @file tensor_ops.h
 * @brief Flat-C forward ops + evaluate — Python/ctypes compatible.
 *
 * Every function returns a new sc_tensor_t* allocated in the given pool
 * (lazy / define-graph style), except sc_tensor_evaluate* which trigger
 * the actual computation for a single node.
 * Include via soft_cuda_python.h — do not include directly.
 */

#pragma once

#include "tensor_pool.h"
#include "tensor_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Matrix / element-wise ops  (all lazy — return a new op-node tensor)
 * ----------------------------------------------------------------------- */

/** Cache-optimised matrix multiplication (calls tensor_mul internally). */
sc_tensor_t *sc_tensor_mul(sc_pool_t *pool, sc_tensor_t *x, sc_tensor_t *y);

/** Naive O(n³) matrix multiplication (calls tensor_mul_naive). */
sc_tensor_t *sc_tensor_mul_naive(sc_pool_t *pool, sc_tensor_t *x, sc_tensor_t *y);

/** Element-wise / matrix addition. */
sc_tensor_t *sc_tensor_add(sc_pool_t *pool, sc_tensor_t *x, sc_tensor_t *y);

/**
 * Explicit bias-broadcast addition (e.g., Y = XW + b where b has fewer dims).
 * Calls tensor_add_bias internally.
 */
sc_tensor_t *sc_tensor_add_bias(sc_pool_t *pool, sc_tensor_t *xw, sc_tensor_t *bias);

/** Element-wise / matrix subtraction. */
sc_tensor_t *sc_tensor_sub(sc_pool_t *pool, sc_tensor_t *a, sc_tensor_t *b);

/** ReLU activation. */
sc_tensor_t *sc_tensor_relu(sc_pool_t *pool, sc_tensor_t *a);

/** Scalar mean of all elements. Returns a rank-0 (scalar) tensor. */
sc_tensor_t *sc_tensor_mean(sc_pool_t *pool, sc_tensor_t *a);

/**
 * Mean-squared-error loss.
 *
 * @param predictions  Model output tensor.
 * @param target       Ground-truth tensor (same shape as predictions).
 * @return             Scalar MSE tensor.
 */
sc_tensor_t *sc_tensor_mse_loss(sc_pool_t *pool,
                                sc_tensor_t *predictions,
                                sc_tensor_t *target);

/** Element-wise square. */
sc_tensor_t *sc_tensor_square(sc_pool_t *pool, sc_tensor_t *x);

/**
 * Transpose a 2-D matrix.
 * The returned tensor shares no memory with the input; it is a new node.
 */
sc_tensor_t *sc_tensor_transpose(sc_pool_t *pool, sc_tensor_t *a);

/* -----------------------------------------------------------------------
 * Evaluate (eager execution for a single node)
 * ----------------------------------------------------------------------- */

/**
 * Evaluate a single tensor node on the CPU.
 * Internally calls tensor_evaluate(pool, t, NULL, NULL, NULL).
 *
 * @param pool  CPU memory pool.
 * @param t     Tensor node to evaluate.
 * @return      Non-zero on success.
 */
int sc_tensor_evaluate(sc_pool_t *pool, sc_tensor_t *t);

/**
 * Evaluate a single tensor node on the GPU.
 * Caller is responsible for providing device pointers to parent data
 * (d_a, d_b) and a device pointer for the output (d_res).
 *
 * @param pool   CPU pool (used for metadata).
 * @param t      Tensor node to evaluate.
 * @param d_a    GPU pointer to operand A data (float*), or NULL.
 * @param d_b    GPU pointer to operand B data (float*), or NULL.
 * @param d_res  GPU pointer to output buffer (float*).
 * @return       Non-zero on success.
 */
int sc_tensor_evaluate_gpu(sc_pool_t *pool,
                           sc_tensor_t *t,
                           float *d_a,
                           float *d_b,
                           float *d_res);

#ifdef __cplusplus
} /* extern "C" */
#endif

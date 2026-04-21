/* ═══════════════════════════════════════════════════════════════════════════
 *  Written by  : Antigravity (AI Coding Assistant)
 *  Date        : 2026-04-21  02:12 IST
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @file tensor_core.h
 * @brief Flat-C tensor lifecycle API — Python/ctypes compatible.
 *
 * Uses plain int dtype constants instead of enum class tensor_dtype_t
 * so Python ctypes can pass them directly.
 * Include via soft_cuda_python.h — do not include directly.
 */

#pragma once

#include <stdint.h>
#include "tensor_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tensor handle. */
typedef struct tensor_instance sc_tensor_t;

/* -----------------------------------------------------------------------
 * Data-type constants  (mirrors tensor_dtype_t enum class values)
 * ----------------------------------------------------------------------- */
#define SC_DTYPE_UINT32  0
#define SC_DTYPE_INT32   1
#define SC_DTYPE_UINT64  2
#define SC_DTYPE_INT64   3
#define SC_DTYPE_FLOAT32 4
#define SC_DTYPE_FLOAT64 5

/* -----------------------------------------------------------------------
 * Tensor lifecycle
 * ----------------------------------------------------------------------- */

/**
 * Create a tensor.
 *
 * @param pool      Memory pool to allocate from.
 * @param dtype     One of the SC_DTYPE_* constants above.
 * @param num_dims  Rank of the tensor (number of dimensions).
 * @param dims      Array of length num_dims containing each dimension size.
 * @param elems     Initial data buffer, or NULL for a zero-initialised tensor.
 *                  The buffer is copied; the caller retains ownership.
 * @param grad      Non-zero → autograd will track this tensor.
 * @return          New tensor pointer, or NULL on allocation failure.
 */
sc_tensor_t *sc_tensor_create(sc_pool_t *pool,
                              int dtype,
                              uint32_t num_dims,
                              uint32_t *dims,
                              void *elems,
                              int grad);

/**
 * Return the unique identifier for a tensor.
 * IDs are unique within the same pool.
 *
 * @param t  The tensor.
 * @return   Unique uint32 id.
 */
uint32_t sc_tensor_id(sc_tensor_t *t);

/**
 * Fetch a raw pointer to the tensor's data buffer.
 *
 * @param t  The tensor.
 * @return   void* to the underlying data array.
 */
void *sc_tensor_get_data(sc_tensor_t *t);

/**
 * Return the rank (number of dimensions) of the tensor.
 *
 * @param t  The tensor.
 * @return   Number of dimensions (0 = scalar).
 */
uint8_t sc_tensor_get_ndims(sc_tensor_t *t);

/**
 * Return a pointer to the dimension array.
 * The array has sc_tensor_get_ndims(t) valid entries.
 *
 * @param t  The tensor.
 * @return   Pointer to dims[TENSOR_MAX_DIMS+1].
 */
uint32_t *sc_tensor_get_dims(sc_tensor_t *t);

/**
 * Print tensor data to stdout (mirrors tensor_print_data).
 *
 * @param t  The tensor to print.
 */
void sc_tensor_print_data(sc_tensor_t *t);

/**
 * Fill an existing tensor with normally distributed random floats.
 *
 * @param t        Tensor to fill (must be FLOAT32).
 * @param mean     Distribution mean.
 * @param std_dev  Standard deviation.
 * @return         Non-zero on success.
 */
int sc_tensor_fill_random_normal(sc_tensor_t *t, float mean, float std_dev);

/**
 * Fetch a raw pointer to the tensor's gradient data buffer.
 *
 * @param t  The tensor.
 * @return   void* to the underlying gradient data array.
 */
void *sc_tensor_grad_get_data(sc_tensor_t *t);

/**
 * Identify if a tensor is transposed or not.
 *
 * @param t  The tensor.
 * @return   A boolean with true for is transposed and false for not transposed.
 * */
bool sc_tensor_is_transposed(tensor_t *t);

#ifdef __cplusplus
} /* extern "C" */
#endif

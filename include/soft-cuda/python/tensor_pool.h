/* ═══════════════════════════════════════════════════════════════════════════
 *  Written by  : Antigravity (AI Coding Assistant)
 *  Date        : 2026-04-21  02:12 IST
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @file tensor_pool.h
 * @brief Flat-C arena management API — Python/ctypes compatible.
 *
 * All functions use only C-safe primitives: void*, size_t, int.
 * This header is part of the soft_cuda_python bridge layer.
 * Include via soft_cuda_python.h — do not include directly.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque alias so C and Python both see the same type name. */
typedef struct tensor_pool_instance sc_pool_t;

/**
 * Create a new memory arena.
 *
 * @param capacity_bytes  Total bytes to pre-allocate.
 * @param on_device       Non-zero → allocate on GPU VRAM; zero → CPU RAM.
 * @return                Pointer to the new pool, or NULL on failure.
 */
sc_pool_t *sc_pool_create(size_t capacity_bytes, int on_device);

/**
 * Completely destroy the pool and return memory to the system.
 *
 * @param pool  Pool to destroy. Must not be used after this call.
 */
void sc_pool_destroy(sc_pool_t *pool);

/**
 * Reset the bump pointer to zero, invalidating all tensors in the pool
 * without releasing the underlying memory block.
 * Highly efficient for resetting temporary/gradient pools each step.
 *
 * @param pool  Pool to reset.
 */
void sc_pool_zero(sc_pool_t *pool);

/**
 * Allocate raw bytes from the pool.
 *
 * @param pool   The pool to allocate from.
 * @param size   Number of bytes to allocate.
 * @param out_id Output: unique id assigned to this allocation.
 * @return       Pointer to the allocated block, or NULL if exhausted.
 */
void *sc_pool_alloc(sc_pool_t *pool, size_t size, uint32_t *out_id);

/**
 * Return the total capacity of the pool in bytes.
 *
 * @param pool  The pool to query.
 * @return      Capacity in bytes.
 */
size_t sc_pool_size(sc_pool_t *pool);

/**
 * Return the number of bytes currently in use.
 *
 * @param pool  The pool to query.
 * @return      Bytes consumed so far.
 */
size_t sc_pool_used(sc_pool_t *pool);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 *  Written by  : Antigravity (AI Coding Assistant)
 *  Date        : 2026-04-21  02:12 IST
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @file tensor_io.h
 * @brief Flat-C model persistence API — Python/ctypes compatible.
 *
 * Wraps the C++ save_model / load_model (from src/core/graph/saveLoad.cpp)
 * using only C-safe primitives (const char* path, pointer array + count).
 *
 * Include via soft_cuda_python.h — do not include directly.
 */

#pragma once

#include <stddef.h>
#include "tensor_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Save model weights to a binary file.
 *
 * Writes all tensor data sequentially as raw float32 bytes.
 * Tensors must already have been evaluated (data populated) before saving.
 *
 * @param path     Null-terminated file path.
 * @param tensors  Array of tensor pointers to save (parameter tensors).
 * @param count    Number of tensors in the array.
 * @return         Non-zero on success, 0 on failure (e.g. file not writable).
 *
 * @note  Only the raw float data is saved — shape/dtype metadata is NOT
 *        persisted.  The caller must reconstruct tensors of the correct
 *        shape before calling sc_load_model.
 */
int sc_save_model(const char *path, sc_tensor_t **tensors, size_t count);

/**
 * Load model weights from a binary file into pre-allocated tensors.
 *
 * Reads raw float32 bytes and fills each tensor in-order.
 * Tensors must already be created with the correct shapes before loading.
 *
 * @param path     Null-terminated file path.
 * @param tensors  Array of pre-allocated tensor pointers to fill.
 * @param count    Number of tensors in the array.
 * @return         Non-zero on success, 0 on failure (e.g. file not found,
 *                 size mismatch).
 */
int sc_load_model(const char *path, sc_tensor_t **tensors, size_t count);

#ifdef __cplusplus
} /* extern "C" */
#endif

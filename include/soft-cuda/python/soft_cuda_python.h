/* ═══════════════════════════════════════════════════════════════════════════
 *  Written by  : Antigravity (AI Coding Assistant)
 *  Date        : 2026-04-21  02:12 IST
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @file soft_cuda_python.h
 * @brief Master include for the soft-cuda Python/ctypes bridge layer.
 *
 * Include only this header in your Python binding or cffi cdef block.
 * It pulls in the five sub-headers in dependency order:
 *
 *   tensor_pool.h   — arena management (create, destroy, zero, alloc, size, used)
 *   tensor_core.h   — tensor lifecycle (create, id, get_data, get_ndims, get_dims,
 *                                        print_data, fill_random_normal)
 *   tensor_ops.h    — forward ops + evaluate
 *                      (mul, mul_naive, add, add_bias, sub, relu,
 *                       mean, mse_loss, square, transpose, evaluate, evaluate_gpu)
 *   tensor_graph.h  — graph building + training primitives
 *                      Layer 1: verify_dag, assign_backend, assign_grad_memory,
 *                               graph_forward, autograd_gpu_transfer,
 *                               grad_initializer, backward, sgd, node_to_host
 *                      Layer 2: build_graph, graph_step, graph_get_loss, graph_size
 *   tensor_io.h     — persistence (save_model, load_model)
 *
 * All symbols use the "sc_" prefix and expose only C-safe types so that
 * Python ctypes / cffi can parse and call them directly.
 *
 * Shared library: link against  libsoft_cuda_python.so  (or .dll on Windows).
 *
 * Quick Python example:
 * @code
 *   import ctypes, numpy as np
 *   lib = ctypes.CDLL("./libsoft_cuda_python.so")
 *
 *   lib.sc_pool_create.restype  = ctypes.c_void_p
 *   lib.sc_pool_create.argtypes = [ctypes.c_size_t, ctypes.c_int]
 *   pool = lib.sc_pool_create(4 * 1024 * 1024, 0)  # 4 MB CPU pool
 * @endcode
 */

#pragma once

#include "tensor_pool.h"
#include "tensor_core.h"
#include "tensor_ops.h"
#include "tensor_graph.h"
#include "tensor_io.h"

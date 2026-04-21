<!-- ═══════════════════════════════════════════════════════════════════════
     Written by  : Antigravity (AI Coding Assistant)
     Date        : 2026-04-21  02:12 IST
     ═══════════════════════════════════════════════════════════════════════ -->

# soft-cuda Python Bridge — API Reference

> **`include/soft-cuda/python/soft_cuda_python.h`** — the only header you need.
> **`libsoft_cuda_python.so`** (or `.dll`) — the shared library to load from Python.

The Python bridge wraps the internal C++ tensor library behind a flat-C
(`extern "C"`) interface.  Every symbol uses the **`sc_`** prefix and
only C-safe primitives — no `std::vector`, no `enum class`, no default
arguments.  This makes it directly callable from Python via `ctypes` or
`cffi`.

---

## Table of Contents

- [Architecture](#architecture)
- [Building](#building)
- [Headers at a Glance](#headers-at-a-glance)
- [API Reference](#api-reference)
  - [Pool Management — `tensor_pool.h`](#pool-management)
  - [Tensor Lifecycle — `tensor_core.h`](#tensor-lifecycle)
  - [Forward Ops — `tensor_ops.h`](#forward-ops)
  - [Graph & Training — `tensor_graph.h`](#graph--training)
  - [Persistence — `tensor_io.h`](#persistence)
- [Python ctypes Example](#python-ctypes-example)
- [C/C++ Usage Example](#cc-usage-example)
- [Memory Model](#memory-model)
- [Design Decisions](#design-decisions)

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  Python  (ctypes.CDLL / cffi)                                         │
│                                                                        │
│    import ctypes                                                       │
│    lib = ctypes.CDLL("./libsoft_cuda_python.so")                       │
│    pool = lib.sc_pool_create(4*1024*1024, 0)                           │
│                                                                        │
├────────────────────────── extern "C" ABI ──────────────────────────────┤
│                                                                        │
│  sc_bridge.cpp                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ sc_pool_create()     → tensor_pool_create()                     │  │
│  │ sc_tensor_create()   → tensor_create() + enum cast              │  │
│  │ sc_tensor_mul()      → tensor_mul()                             │  │
│  │ sc_build_graph()     → verifyIfDAG + assignBackendGraph + ...   │  │
│  │ sc_graph_step()      → forward + gpu_transfer + grad + bwd + sgd│  │
│  │ sc_graph_t           → wraps std::vector<execution_node_t*>     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
├────────────────────────── C++ internals ───────────────────────────────┤
│                                                                        │
│  soft_lib  (shared library)                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ tensor.h, pool.h, DAGbuild.cpp, assignBackend.cu, train.cpp ... │ │
│  │ backend_cpu/*, backend_gpu/*, saveLoad.cpp                      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Building

```bash
# Configure (once)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build everything (main executable + python bridge)
cmake --build build

# Build only the Python bridge shared library
cmake --build build --target soft_cuda_python
```

The output is:

| Target | File | Description |
|---|---|---|
| `soft` | `build/soft` | Main executable (demo) |
| `soft_lib` | `build/libsoft_lib.so` | Core C++ tensor library |
| `soft_cuda_python` | `build/libsoft_cuda_python.so` | Python-compatible bridge |

---

## Headers at a Glance

```
include/soft-cuda/python/
├── soft_cuda_python.h    ← master include (just #includes the five below)
│
├── tensor_pool.h         ← arena management
│                            sc_pool_create, sc_pool_destroy, sc_pool_zero,
│                            sc_pool_alloc, sc_pool_size, sc_pool_used
│
├── tensor_core.h         ← tensor lifecycle + dtype constants
│                            SC_DTYPE_FLOAT32, SC_DTYPE_INT32, ...
│                            sc_tensor_create, sc_tensor_id, sc_tensor_get_data,
│                            sc_tensor_get_ndims, sc_tensor_get_dims,
│                            sc_tensor_print_data, sc_tensor_fill_random_normal
│
├── tensor_ops.h          ← forward ops + evaluate
│                            sc_tensor_mul, sc_tensor_mul_naive,
│                            sc_tensor_add, sc_tensor_add_bias, sc_tensor_sub,
│                            sc_tensor_relu, sc_tensor_mean, sc_tensor_mse_loss,
│                            sc_tensor_square, sc_tensor_transpose,
│                            sc_tensor_evaluate, sc_tensor_evaluate_gpu
│
├── tensor_graph.h        ← graph building + training
│                            (Layer 1) sc_graph_create, sc_graph_destroy,
│                                      sc_verify_dag, sc_assign_backend,
│                                      sc_assign_grad_memory, sc_graph_forward,
│                                      sc_autograd_gpu_transfer,
│                                      sc_grad_initializer, sc_backward,
│                                      sc_sgd, sc_node_to_host
│                            (Layer 2) sc_build_graph, sc_graph_step,
│                                      sc_graph_get_loss, sc_graph_size
│
└── tensor_io.h           ← persistence
                             sc_save_model, sc_load_model
```

---

## API Reference

### Pool Management

> Defined in `tensor_pool.h`

```c
/* Opaque type — never dereference, just pass the pointer around. */
typedef struct tensor_pool_instance sc_pool_t;

/* Create a new arena. on_device=1 → GPU VRAM, on_device=0 → CPU RAM. */
sc_pool_t *sc_pool_create(size_t capacity_bytes, int on_device);

/* Completely free the arena and return memory to the OS. */
void sc_pool_destroy(sc_pool_t *pool);

/* Reset bump pointer to zero (invalidates all tensors, keeps memory). */
void sc_pool_zero(sc_pool_t *pool);

/* Raw allocation from the pool. Returns NULL if exhausted. */
void *sc_pool_alloc(sc_pool_t *pool, size_t size, uint32_t *out_id);

/* Query capacity and usage. */
size_t sc_pool_size(sc_pool_t *pool);
size_t sc_pool_used(sc_pool_t *pool);
```

> [!TIP]
> Use **separate pools** for data, metadata, GPU memory, and gradients.
> This lets you `sc_pool_zero(pool_grad)` between epochs without
> affecting your weights.

---

### Tensor Lifecycle

> Defined in `tensor_core.h`

#### Data-type Constants

```c
#define SC_DTYPE_UINT32  0
#define SC_DTYPE_INT32   1
#define SC_DTYPE_UINT64  2
#define SC_DTYPE_INT64   3
#define SC_DTYPE_FLOAT32 4     /* ← the one you'll use 99% of the time */
#define SC_DTYPE_FLOAT64 5
```

#### Functions

```c
typedef struct tensor_instance sc_tensor_t;

/* Create a tensor. Pass NULL for elems to get zero-initialised data.
   grad = 1 → autograd will track this tensor (for weights). */
sc_tensor_t *sc_tensor_create(sc_pool_t *pool, int dtype,
                              uint32_t num_dims, uint32_t *dims,
                              void *elems, int grad);

uint32_t      sc_tensor_id(sc_tensor_t *t);
void         *sc_tensor_get_data(sc_tensor_t *t);
uint8_t       sc_tensor_get_ndims(sc_tensor_t *t);
uint32_t     *sc_tensor_get_dims(sc_tensor_t *t);
void          sc_tensor_print_data(sc_tensor_t *t);
int           sc_tensor_fill_random_normal(sc_tensor_t *t, float mean, float std_dev);
```

> [!IMPORTANT]
> Set `grad = 0` for input data and targets.
> Set `grad = 1` for trainable parameters (weights, biases).

---

### Forward Ops

> Defined in `tensor_ops.h`

All ops are **lazy** — they don't compute anything. They create a new
tensor node that records the operation and its operands. Actual
computation happens when the graph's `forward` is called.

```c
/* Arithmetic */
sc_tensor_t *sc_tensor_mul(sc_pool_t *pool, sc_tensor_t *x, sc_tensor_t *y);
sc_tensor_t *sc_tensor_mul_naive(sc_pool_t *pool, sc_tensor_t *x, sc_tensor_t *y);
sc_tensor_t *sc_tensor_add(sc_pool_t *pool, sc_tensor_t *x, sc_tensor_t *y);
sc_tensor_t *sc_tensor_add_bias(sc_pool_t *pool, sc_tensor_t *xw, sc_tensor_t *bias);
sc_tensor_t *sc_tensor_sub(sc_pool_t *pool, sc_tensor_t *a, sc_tensor_t *b);
sc_tensor_t *sc_tensor_square(sc_pool_t *pool, sc_tensor_t *x);
sc_tensor_t *sc_tensor_transpose(sc_pool_t *pool, sc_tensor_t *a);

/* Activation */
sc_tensor_t *sc_tensor_relu(sc_pool_t *pool, sc_tensor_t *a);

/* Reduction */
sc_tensor_t *sc_tensor_mean(sc_pool_t *pool, sc_tensor_t *a);

/* Loss */
sc_tensor_t *sc_tensor_mse_loss(sc_pool_t *pool,
                                sc_tensor_t *predictions,
                                sc_tensor_t *target);

/* Single-node evaluate (for debugging — you don't need these normally) */
int sc_tensor_evaluate(sc_pool_t *pool, sc_tensor_t *t);
int sc_tensor_evaluate_gpu(sc_pool_t *pool, sc_tensor_t *t,
                           float *d_a, float *d_b, float *d_res);
```

> [!NOTE]
> `sc_tensor_mul` internally transposes B and does a cache-optimised
> matmul.  `sc_tensor_mul_naive` is the straightforward O(n³) version,
> safer for small matrices (e.g. XOR).

---

### Graph & Training

> Defined in `tensor_graph.h`

#### Backend Constants

```c
#define SC_BACKEND_GPU    0
#define SC_BACKEND_CPU    1
#define SC_BACKEND_HYBRID 2     /* auto-dispatch per CONFIG.soft */
```

#### Opaque Graph Handle

```c
typedef struct sc_graph sc_graph_t;
```

This hides  `std::vector<execution_node_t*>`  from Python/C.

#### Layer 1 — Low-level Primitives

```c
sc_graph_t *sc_graph_create(void);
void        sc_graph_destroy(sc_graph_t *g);

int         sc_verify_dag(sc_pool_t *meta_pool, sc_tensor_t *t, sc_graph_t *g);
void        sc_assign_backend(sc_pool_t *pool_gpu, sc_graph_t *g, int mode);
void        sc_assign_grad_memory(sc_pool_t *pool_grad_cpu,
                                  sc_pool_t *pool_grad_gpu,
                                  sc_graph_t *g);

int         sc_graph_forward(sc_pool_t *pool_cpu, sc_pool_t *pool_gpu, sc_graph_t *g);
void        sc_autograd_gpu_transfer(sc_graph_t *g);
void        sc_grad_initializer(sc_graph_t *g);
int         sc_backward(sc_graph_t *g);
void        sc_sgd(sc_graph_t *g, float learning_rate);
int         sc_node_to_host(sc_graph_t *g, size_t node_idx);
```

#### Layer 2 — Convenience API

```c
/* One-liner: verify DAG → assign backend → alloc grads. Returns ready graph. */
sc_graph_t *sc_build_graph(sc_pool_t *meta_pool,
                           sc_pool_t *pool_gpu,
                           sc_pool_t *pool_grad_cpu,
                           sc_pool_t *pool_grad_gpu,
                           sc_tensor_t *loss,
                           int backend_mode);

/* One-liner: forward → gpu_transfer → zero_grads → backward → sgd. */
void sc_graph_step(sc_pool_t *pool_cpu,
                   sc_pool_t *pool_gpu,
                   sc_graph_t *g,
                   float learning_rate);

/* Read the scalar loss value (auto-transfers from GPU if needed). */
float sc_graph_get_loss(sc_graph_t *g);

/* Number of nodes in the topological order. */
size_t sc_graph_size(sc_graph_t *g);
```

> [!TIP]
> **For Python users**, the Layer 2 API is almost always what you want.
> Two function calls cover the entire training loop:
> ```python
> graph = lib.sc_build_graph(meta, gpu, gcpu, ggpu, loss, SC_BACKEND_CPU)
> for epoch in range(10000):
>     lib.sc_graph_step(pool, gpu, graph, ctypes.c_float(0.05))
> ```

---

### Persistence

> Defined in `tensor_io.h`

```c
/* Saves raw float32 data sequentially. No shape metadata stored. */
int sc_save_model(const char *path, sc_tensor_t **tensors, size_t count);

/* Loads raw float32 data into pre-allocated tensors (shapes must match). */
int sc_load_model(const char *path, sc_tensor_t **tensors, size_t count);
```

> [!WARNING]
> Only the raw float bytes are persisted — **not** the tensor shapes or
> dtypes.  You must recreate tensors with the correct shapes before
> calling `sc_load_model`.

---

## Python ctypes Example

```python
#!/usr/bin/env python3
"""XOR training using soft-cuda from Python via ctypes."""

import ctypes
import os

# ── Load the shared library ────────────────────────────────────
lib = ctypes.CDLL("./build/libsoft_cuda_python.so")

# ── Declare return types (ctypes defaults to c_int otherwise) ──
lib.sc_pool_create.restype       = ctypes.c_void_p
lib.sc_tensor_create.restype     = ctypes.c_void_p
lib.sc_tensor_get_data.restype   = ctypes.c_void_p
lib.sc_tensor_get_dims.restype   = ctypes.POINTER(ctypes.c_uint32)
lib.sc_tensor_mul_naive.restype  = ctypes.c_void_p
lib.sc_tensor_add.restype        = ctypes.c_void_p
lib.sc_tensor_relu.restype       = ctypes.c_void_p
lib.sc_tensor_sub.restype        = ctypes.c_void_p
lib.sc_tensor_square.restype     = ctypes.c_void_p
lib.sc_tensor_mean.restype       = ctypes.c_void_p
lib.sc_build_graph.restype       = ctypes.c_void_p
lib.sc_graph_get_loss.restype    = ctypes.c_float
lib.sc_graph_size.restype        = ctypes.c_size_t

# ── Constants ──────────────────────────────────────────────────
SC_DTYPE_FLOAT32 = 4
SC_BACKEND_CPU   = 1

# ── Helper to create a uint32 array ───────────────────────────
def dims(*d):
    return (ctypes.c_uint32 * len(d))(*d)

def floats(*f):
    return (ctypes.c_float * len(f))(*f)

# ── 1. Create pools ───────────────────────────────────────────
MB = 1024 * 1024
pool     = lib.sc_pool_create(MB, 0)
meta     = lib.sc_pool_create(MB, 0)
gcpu     = lib.sc_pool_create(MB, 0)
gpu      = lib.sc_pool_create(MB, 1)
ggpu     = lib.sc_pool_create(MB, 1)

# ── 2. Create tensors ─────────────────────────────────────────
X = lib.sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims(4,2),
                         floats(0,0, 0,1, 1,0, 1,1), 0)
Y = lib.sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims(4,1),
                         floats(0, 1, 1, 0), 0)

W1 = lib.sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims(2,4), None, 1)
b1 = lib.sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims(1,4), None, 1)
W2 = lib.sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims(4,1), None, 1)
b2 = lib.sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims(1,1), None, 1)

lib.sc_tensor_fill_random_normal(W1, ctypes.c_float(0.5), ctypes.c_float(0.2))
lib.sc_tensor_fill_random_normal(W2, ctypes.c_float(0.5), ctypes.c_float(0.2))
lib.sc_tensor_fill_random_normal(b1, ctypes.c_float(0.0), ctypes.c_float(0.1))
lib.sc_tensor_fill_random_normal(b2, ctypes.c_float(0.0), ctypes.c_float(0.1))

# ── 3. Define computation graph ───────────────────────────────
H      = lib.sc_tensor_relu(pool, lib.sc_tensor_add(pool,
            lib.sc_tensor_mul_naive(pool, X, W1), b1))
Y_pred = lib.sc_tensor_add(pool, lib.sc_tensor_mul_naive(pool, H, W2), b2)
diff   = lib.sc_tensor_sub(pool, Y_pred, Y)
loss   = lib.sc_tensor_mean(pool, lib.sc_tensor_square(pool, diff))

# ── 4. Build graph (one call) ─────────────────────────────────
graph = lib.sc_build_graph(meta, gpu, gcpu, ggpu, loss, SC_BACKEND_CPU)
assert graph is not None
print(f"Graph built: {lib.sc_graph_size(graph)} nodes")

# ── 5. Train ──────────────────────────────────────────────────
for epoch in range(10001):
    lib.sc_graph_step(pool, gpu, graph, ctypes.c_float(0.05))
    if epoch % 2000 == 0:
        l = lib.sc_graph_get_loss(graph)
        print(f"  epoch {epoch:5d}   loss = {l:.8f}")

# ── 6. Read predictions ──────────────────────────────────────
lib.sc_graph_forward(pool, gpu, graph)
pred_ptr = ctypes.cast(lib.sc_tensor_get_data(Y_pred),
                       ctypes.POINTER(ctypes.c_float))

print("\n  X1   X2   │  Target  │  Predicted")
print("  " + "─" * 38)
inputs = [0,0, 0,1, 1,0, 1,1]
targets = [0, 1, 1, 0]
for i in range(4):
    print(f"  {inputs[i*2]:.0f}    {inputs[i*2+1]:.0f}    │   {targets[i]:.0f}      │   {pred_ptr[i]:.4f}")

# ── 7. Cleanup ────────────────────────────────────────────────
lib.sc_graph_destroy(graph)
for p in [pool, meta, gcpu, ggpu, gpu]:
    lib.sc_pool_destroy(p)
```

---

## C/C++ Usage Example

See **`main.cpp`** in the repository root — it trains an XOR network
using exclusively `sc_*` calls and is heavily commented as a walkthrough.

The key pattern is:

```c
#include "soft-cuda/python/soft_cuda_python.h"   /* single include */

/* 1. Pools */
sc_pool_t *pool = sc_pool_create(1*1024*1024, 0);

/* 2. Tensors  (SC_DTYPE_FLOAT32 = 4,  grad=1 for weights) */
sc_tensor_t *W = sc_tensor_create(pool, SC_DTYPE_FLOAT32, 2, dims, NULL, 1);
sc_tensor_fill_random_normal(W, 0.0f, 0.1f);

/* 3. Lazy ops */
sc_tensor_t *out = sc_tensor_mul_naive(pool, X, W);

/* 4. Build graph (one call) */
sc_graph_t *g = sc_build_graph(meta, gpu, gcpu, ggpu, loss, SC_BACKEND_CPU);

/* 5. Train */
for (int i = 0; i < 10000; i++)
    sc_graph_step(pool, gpu, g, 0.05f);

/* 6. Read loss */
float l = sc_graph_get_loss(g);

/* 7. Cleanup */
sc_graph_destroy(g);
sc_pool_destroy(pool);
```

---

## Memory Model

soft-cuda uses **arena (bump) allocation** — all objects are allocated
from pre-sized pools:

```
┌─────────────────────────────────────────────────────┐
│  sc_pool_t (e.g. 1 MB)                              │
│                                                      │
│  ┌────┬────┬────┬────┬────┬───────────────────────┐ │
│  │ T1 │ T2 │ T3 │ T4 │ T5 │   ← free space →     │ │
│  └────┴────┴────┴────┴────┴───────────────────────┘ │
│  ▲                         ▲                         │
│  base                      bump pointer              │
│                                                      │
│  sc_pool_zero() → resets bump to base (O(1) reset)   │
│  sc_pool_destroy() → returns memory to the OS        │
└─────────────────────────────────────────────────────┘
```

**Recommended pool layout:**

| Pool | `on_device` | Purpose |
|---|---|---|
| `pool` | 0 | Tensor data + op nodes |
| `pool_meta` | 0 | `execution_node_t` objects (graph metadata) |
| `pool_grad_cpu` | 0 | CPU-side gradient tensors |
| `pool_gpu` | 1 | GPU VRAM for forward-pass device data |
| `pool_grad_gpu` | 1 | GPU VRAM for backward-pass gradient buffers |

> [!CAUTION]
> `sc_pool_zero()` invalidates **all** tensors allocated from that pool.
> Use it only on pools whose contents you're sure you no longer need
> (e.g. ephemeral scratch pools, not your weight pool).

---

## Design Decisions

### Why a separate bridge layer (instead of fixing `api.h` directly)?

The existing `api.h` uses C++ features (`enum class`, `std::vector`,
default argument values) that Python's FFI (ctypes, cffi) cannot parse.
Rather than stripping those features from the core library (which would
break the natural C++ ergonomics for C++ users), we added a thin
translation layer.

### Why `sc_graph_t` instead of exposing `std::vector` as an opaque pointer?

A raw `std::vector*` could work, but:

1. We'd need a separate `sc_graph_push` / `sc_graph_get` / `sc_graph_len`
   anyway — the bridge functions need typed access.
2. Bundling the vector inside a named struct gives us a place to add
   additional state later (e.g. caching the loss value, storing pool
   references for automatic cleanup).
3. `typedef struct sc_graph sc_graph_t;` makes the intent clear in the
   header — Python users see "this is a graph handle" not "void pointer
   to something".

### Why `placement-new` in `sc_graph_create`?

`malloc` gives us a C-compatible, ABI-stable allocation.  `placement-new`
properly constructs the `std::vector` inside that allocation.
This avoids exposing `new`/`delete` at the ABI boundary while ensuring
the vector's internal state is correctly initialised.

### Why `SC_DTYPE_*` constants instead of an enum?

Python `ctypes` can't parse C `enum` declarations.  `#define` constants
are just integers — they work everywhere.  The bridge function casts
them back to `tensor_dtype_t` internally.

### Why Layer 1 + Layer 2?

Layer 1 gives full control (e.g. run forward without backward, inspect
intermediate nodes, use a custom optimizer).  Layer 2 covers the 90%
case (training loop) in fewer calls.  Both are available simultaneously.

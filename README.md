# soft-cuda

A from-scratch neural network training library written in C++ and CUDA. It implements tensor operations, automatic differentiation, a DAG-based computation graph, CPU and GPU backends, a hybrid dispatch system, and a flat-C bridge layer for use from Python via `ctypes` or `cffi`.

---

## Overview

soft-cuda provides the core machinery needed to define, execute, and differentiate computation graphs that can run on the CPU, on the GPU, or across both depending on tensor size. It does not depend on any existing deep learning framework. Memory is managed through arena-style bump allocators that the user controls explicitly.

The library is built as two shared libraries:

- `libsoft_lib.so` -- the full internal C++ library containing the tensor core, both backends, and the computation graph engine.
- `libsoft_cuda_python.so` -- a thin flat-C wrapper over `libsoft_lib.so` that exposes every function with `extern "C"` linkage and C-compatible types, making it directly loadable by Python `ctypes` or `cffi`.

A standalone executable `soft` is also built from `main.cpp` as a demonstration that trains a small XOR network using only the flat-C bridge API.

A separate executable `soft_profiler` measures CPU versus GPU throughput for each supported operation at a range of element counts, finds the crossover threshold, and writes a `CONFIG.soft` JSON file. This file drives the HYBRID backend mode in subsequent runs.

---

## Repository Layout

```
soft-cuda/
  CMakeLists.txt                   -- top-level build definition
  make.sh                          -- convenience wrapper around cmake
  main.cpp                         -- XOR demo using the sc_* bridge API
  include/
    soft-cuda/
      tensor/
        api.h                      -- public C++ API for all tensor operations
        debug_api.h                -- debug utilities
        tensor.h                   -- (internal) opaque struct forward-declarations
      python/
        soft_cuda_python.h         -- master include for the flat-C bridge
        tensor_pool.h              -- pool wrappers (sc_pool_*)
        tensor_core.h              -- tensor lifecycle wrappers (sc_tensor_*)
        tensor_ops.h               -- forward op wrappers (sc_tensor_mul, etc.)
        tensor_graph.h             -- graph and training wrappers (sc_graph_*)
        tensor_io.h                -- model save/load wrappers (sc_save_model, etc.)
  src/
    internal_header.h              -- single include that pulls all public and
                                      private declarations together
    core/
      tensor/tensor.cu             -- tensor creation, evaluate dispatch, random fill
      pool/pool.cu                 -- bump allocator (CPU via malloc, GPU via cudaMalloc)
      graph/
        DAGbuild.cpp               -- recursive topological sort + cycle detection
        assignBackend.cu           -- backend assignment per node, VRAM pre-allocation
        train.cpp                  -- tensor_graph_backward, tensor_sgd
        saveLoad.cpp               -- binary flat-float save and load
        CONFIG.soft                -- fallback JSON config (used when no profiled file exists)
      JSON/json_utils.cpp          -- JSON file and string parsing helpers
    backend_cpu/
      math/                        -- scalar CPU implementations of all ops
        add.cpp, sub.cpp, mul.cpp, relu.cpp, mean.cpp, square.cpp,
        scalar.cpp, transpose.cpp, bias_add.cpp, mse.cpp
      backprop/
        backprop_b.cpp             -- CPU backward dispatcher and all gradient functions
        backprop_cuda_bridge.cu    -- thin bridge calling CUDA memset/memcpy helpers
        matmul_b.cpp               -- helper used by the matrix multiply gradient
    backend_gpu/
      math/                        -- CUDA kernel implementations of all ops
        add.cu, sub.cu, relu.cu, square.cu, mean.cu, scalar_mul.cu,
        broadcast_add.cu, matmul.cu
      backprop/
        backprop_gpu.cu            -- CUDA backward kernels + backprop_gpu_dispatch
      kernels/
        sgemm_double_buffer.cuh    -- warp-tiled SGEMM with double-buffered shared
                                      memory using cuda::barrier and cooperative_groups
    python/
      sc_bridge.cpp                -- flat-C implementation of every sc_* function
    init/config/
      profiler_core.cu             -- AOT hardware profiler logic
      profiler.cu                  -- entry point for the soft_profiler binary
      CONFIG.soft                  -- fallback config (same content as core/graph copy)
      soft_init.h / soft_init.cpp  -- initialization stubs
  tests/
    test_ops.cpp                   -- unit tests for forward operations
    test_mul.cpp                   -- unit tests for matrix multiply variants
  benchmarks/
    bench_softcuda.cpp             -- CPU vs GPU timing for add, matmul, full MLP step
    bench_deep_mlp.cpp             -- deeper MLP training benchmark
    bench_pytorch.py               -- equivalent PyTorch baseline for comparison
    run_all.sh                     -- runs all benchmarks in sequence
  docs/
    PYTHON_BRIDGE.md               -- detailed usage guide for the Python bridge
  scripts/
    nsys_easy.sh                   -- wrapper for Nsight Systems profiling
```

---

## Requirements

- CMake 3.16 or later
- A C++17-capable compiler (GCC or Clang)
- CUDA Toolkit (nvcc) with a GPU that supports the compute capability detected at build time (`CMAKE_CUDA_ARCHITECTURES native`)
- cuBLAS (linked by `target_link_libraries(soft_lib PRIVATE cublas)`)
- Python 3 with `ctypes` or `cffi` for the Python bridge (no additional Python packages are required for the bridge itself; `numpy` is used in the benchmark comparison script)

---

## Building

The project uses CMake. The `make.sh` script is a convenience wrapper.

```bash
# Configure and build in Debug mode
bash make.sh -b

# Clean everything and rebuild
bash make.sh -z

# Run the demo binary (stderr suppressed)
bash make.sh -r

# Run with timing output
bash make.sh -v

# Run tests via ctest
bash make.sh -t

# Run benchmarks
bash make.sh -m
```

Manually with CMake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Targets produced:

| Target | Output |
|---|---|
| `soft_lib` | `libsoft_lib.so` -- full C++ library |
| `soft_cuda_python` | `libsoft_cuda_python.so` -- flat-C Python bridge |
| `soft` | demo executable (XOR network) |
| `soft_profiler` | AOT hardware profiler |
| tests (via `add_subdirectory`) | test binaries run by ctest |
| benchmarks (via `add_subdirectory`) | benchmark binaries |

---

## Architecture

### Memory Model

All allocation goes through `tensor_pool_t`, a bump allocator. The same allocator type backs both CPU (`malloc`) and GPU (`cudaMalloc`) arenas, selected by the `isOfDevice` flag at creation time.

```c
tensor_pool_t *pool     = tensor_pool_create(1024 * 1024, false);  // CPU
tensor_pool_t *pool_gpu = tensor_pool_create(1024 * 1024, true);   // VRAM
```

`tensor_pool_zero` resets the bump pointer to zero without releasing memory, making it efficient to reuse an arena across training iterations. `tensor_pool_destroy` frees the underlying block.

Allocations within a pool are 8-byte aligned. Each allocation receives a monotonically increasing integer ID from the pool's `nallocs` counter.

The caller is responsible for creating separate pools for:
- forward-pass tensor data
- graph metadata (execution nodes)
- CPU-side gradients
- GPU-side gradient buffers
- GPU-side forward data

### Tensor

`tensor_t` is an opaque struct with the following logical fields (defined in `src/core/include/tensor/tensor.h`):

- `dtype` -- one of `UINT32_T`, `INT32_T`, `UINT64_T`, `INT64_T`, `FLOAT32_T`, `FLOAT64_T`
- `ndims` -- rank, up to `TENSOR_MAX_DIMS` (8)
- `dims[8]` -- size along each axis, zero-terminated
- `stride[8]` -- row-major strides computed at creation
- `nvalues` -- total element count (product of dims)
- `data` -- pointer to the element buffer inside a pool
- `op` -- which operation produced this tensor (`tensor_op_t` enum)
- `a`, `b` -- pointers to the input tensors for the operation
- `grad` -- pointer to a gradient tensor (allocated separately)
- `grad_compute` -- flag controlling whether this tensor participates in autograd
- `device` -- `CPU` or `GPU`, tracking where the live copy currently resides
- `is_transposed` -- flag set by `tensor_transpose`
- `stateTracker` -- used during DAG traversal (0 = unvisited, 1 = in progress, 2 = done)
- `broadcast_stride[2]` -- strides used for broadcasting in bias-add backward
- `id` -- allocation ID from the pool

Tensors are created with `tensor_create` (or the internal `tensor_dtype_create`) and live entirely within their pool's memory block. There is no individual tensor free; the entire pool is zeroed or destroyed at once.

### Operation Encoding

Each non-leaf tensor records the operation that produced it in `t->op`:

| `tensor_op_t` | Description |
|---|---|
| `NONE` | Leaf (input or weight) |
| `ADD` | Element-wise addition |
| `BROADCAST_ADD` | Addition with broadcast over rows (bias add pattern) |
| `SUB` | Element-wise subtraction |
| `MUL_MATRIX` | Cache-optimised matrix multiplication (calls transpose internally) |
| `NAIVE_MATRIX_MUL` | Straightforward O(n3) matrix multiply |
| `MUL_SCALAR` | Scalar multiplication |
| `TRANSPOSE` | 2-D matrix transpose |
| `RELU` | Rectified linear unit |
| `MEAN` | Scalar mean of all elements |
| `SQUARE` | Element-wise square |
| `CAST` | (stub, not yet implemented) |

All op-creating functions in `api.h` (e.g. `tensor_add`, `tensor_mul`, `tensor_relu`) do not perform any computation. They allocate a new tensor node, set the `op` field, and store pointers to the input tensors in `a` and `b`. Evaluation is deferred until the graph is executed.

### Computation Graph

`verifyIfDAG` performs a recursive depth-first traversal from the root loss tensor, detects cycles using the `stateTracker` field, and produces a topologically sorted sequence of `execution_node_t` objects stored in a `std::vector`.

Each `execution_node_t` holds:
- `t` -- pointer to the original tensor
- `pos` -- index in the sorted sequence
- `id` -- pool allocation ID
- `parent_pos[2]` -- indices of parent nodes in the sequence (-1 if none)
- `to_device_needed` -- whether input data must be copied to VRAM before execution
- `device_ptr` -- pointer to forward data on the GPU (NULL for CPU nodes)
- `device_ptr_grad` -- pointer to gradient data on the GPU
- `backend_fn` -- function pointer, either `tensor_evaluate` or `tensor_evaluate_GPU`

`setUpParentReference` fills `parent_pos` by building a map from tensor ID to node position after the DAG walk.

### Backend Assignment

`assignBackendGraph` iterates the node list and sets `backend_fn` on each node. Three modes are supported:

- `backend_mode::CPU` -- all nodes use `tensor_evaluate` (CPU).
- `backend_mode::GPU` -- all non-leaf nodes use `tensor_evaluate_GPU`; leaf nodes (op == NONE) always stay on CPU.
- `backend_mode::HYBRID` -- each node is assigned individually by `assignDevice`, which reads `CONFIG.soft` to look up the operation name and element count against size-range breakpoints. If the file does not exist at `~/.config/soft-cuda/CONFIG.soft`, the embedded fallback config is used.

After assigning per-node backends, a second pass handles propagation: if a GPU node's parent was assigned to CPU, `to_device_needed` is set and VRAM space is pre-allocated for the parent. If a CPU node has a parent whose data lives on the GPU, the child is promoted to GPU automatically. An assertion pass at the end verifies consistency.

### Forward Execution

`tensor_graph_forward_evaluate` iterates the node list in topological order. For each GPU node, it copies parent data from CPU to VRAM (via `cudaMemcpy`) if not already resident, then calls `tensor_evaluate_GPU` with device pointers. For CPU nodes it calls `tensor_evaluate` with null device pointers. The function returns `true` if no step fails.

`tensor_evaluate` dispatches to the appropriate CPU implementation function based on `t->op`.

`tensor_evaluate_GPU` dispatches to the appropriate CUDA kernel call. Transpose is handled by falling back to the CPU implementation with a round-trip copy, since no GPU transpose kernel exists yet.

### CPU Backend Operations

All CPU math functions operate on contiguous `float*` arrays extracted from `tensor_t`. Each function receives a `tensor_t *t` with `t->a` and `t->b` set, reads from their data buffers, and writes the result into `t->data`. The operations are straightforward scalar loops. The cache-optimised matrix multiply calls `tensor_transpose` on B before computing the product with row-major dot products for better cache access patterns.

### GPU Backend Operations

Each GPU math operation is a CUDA kernel in `src/backend_gpu/math/`. Thread configuration uses a flat 1-D grid with 256 threads per block for element-wise operations, and a 2-D 32x32 block grid for matrix multiply.

The matrix multiply kernel in `src/backend_gpu/math/matmul.cu` calls into the cuBLAS SGEMM interface. The file `src/backend_gpu/kernels/sgemm_double_buffer.cuh` contains a custom warp-tiled SGEMM kernel that uses double-buffered shared memory with `cuda::barrier` and `cooperative_groups::this_thread_block()` for pipelined memory loading. This kernel is instantiated with explicit template parameters for tile sizes (BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN).

### Backward Pass and Autograd

`assignGradMemory` allocates gradient tensors on the CPU pool and, for nodes assigned to GPU, allocates gradient device buffers on the GPU pool. The gradient tensor (`t->grad`) mirrors the shape of the forward tensor.

`gradInitializer` zeroes all gradient buffers at the start of each step and seeds the root node's gradient to 1.0 (representing dL/dL = 1).

`tensor_graph_backward` (implemented as `backprop__`) iterates the node list in reverse topological order. For each node with `device_ptr_grad != NULL`, it calls `backprop_gpu_dispatch`. If the GPU dispatch returns false (operation not implemented on GPU, or missing forward pointers), it falls back to `backprop_cpu` after copying data between host and device as needed.

`backprop_gpu_dispatch` in `src/backend_gpu/backprop/backprop_gpu.cu` implements CUDA backward kernels for: ADD, SUB, RELU, SQUARE, MEAN, MUL_SCALAR, BROADCAST_ADD, NAIVE_MATRIX_MUL, and MUL_MATRIX. The matrix multiply gradients use 32x32 tiled CUDA kernels for dA and dB. TRANSPOSE backward returns false (falls through to CPU).

`backprop_cpu` in `src/backend_cpu/backprop/backprop_b.cpp` implements CPU gradient functions for all operations. Gradients are accumulated with `+=` to support multiple uses of the same tensor.

### Optimizer

`tensor_sgd` iterates leaf nodes (op == NONE) with `grad_compute` enabled. If both weights and gradients are on the GPU, it calls `tensor_sgd_gpu`, a CUDA kernel that applies `w[i] -= lr * g[i]` in parallel. Otherwise, gradients and weights are pulled to CPU, the update is applied, and weights are pushed back to VRAM.

### Model Save and Load

`save_model` writes the raw float data of each weight tensor sequentially to a binary file. `load_model` reads the data back in the same order into existing tensors. Shape metadata is not stored in the file; the caller must recreate tensors with correct shapes before loading.

---

## Public C++ API (api.h)

### Memory Pool

```cpp
tensor_pool_t *tensor_pool_create(size_t capacity_bytes, bool isOfDevice = false);
void           tensor_pool_zero(tensor_pool_t *pool);
void           tensor_pool_destroy(tensor_pool_t *pool);
void          *tensor_pool_alloc(tensor_pool_t *pool, size_t size, uint32_t *id);
size_t         tensor_pool_size(tensor_pool_t *pool);
size_t         tensor_pool_used(tensor_pool_t *pool);
```

### Tensor Creation and Inspection

```cpp
tensor_t *tensor_create(tensor_pool_t *pool, tensor_dtype_t dtype,
                        uint32_t num_dims, uint32_t *dims,
                        void *elems, bool grad_status = true);
uint32_t   tensor_id(tensor_t *t);
void      *tensor_get_data(tensor_t *t);
uint8_t    tensor_get_ndims(tensor_t *t);
uint32_t  *tensor_get_dims(tensor_t *t);
void       tensor_print_data(tensor_t *t);
bool       tensor_fill_random_normal(tensor_t *t, float mean, float std_dev);
```

`tensor_fill_random_normal` uses the Box-Muller transform to generate normally distributed values. It processes two elements per iteration using pairs of uniform random draws.

### Forward Operations (all lazy)

```cpp
tensor_t *tensor_mul(tensor_pool_t *pool, tensor_t *x, tensor_t *y);
tensor_t *tensor_mul_naive(tensor_pool_t *pool, tensor_t *x, tensor_t *y);
tensor_t *tensor_transpose(tensor_pool_t *pool, tensor_t *a);
tensor_t *tensor_add(tensor_pool_t *pool, tensor_t *x, tensor_t *y);
tensor_t *tensor_add_bias(tensor_pool_t *pool, tensor_t *xw, tensor_t *bias);
tensor_t *tensor_sub(tensor_pool_t *pool, tensor_t *a, tensor_t *b);
tensor_t *tensor_relu(tensor_pool_t *pool, tensor_t *a);
tensor_t *tensor_mean(tensor_pool_t *pool, tensor_t *a);
tensor_t *tensor_square(tensor_pool_t *pool, tensor_t *x);
tensor_t *tensor_mse_loss(tensor_pool_t *pool, tensor_t *predictions, tensor_t *target);
```

### Graph

```cpp
bool verifyIfDAG(tensor_pool_t *pool, tensor_t *t,
                 std::vector<execution_node_t *> &seq);
void assignBackendGraph(tensor_pool_t *pool_gpu,
                        std::vector<execution_node_t *> &nodes,
                        backend_mode value = backend_mode::CPU);
void assignGradMemory(tensor_pool_t *pool_grad_cpu,
                      tensor_pool_t *pool_grad_gpu,
                      std::vector<execution_node_t *> &nodes);
bool tensor_graph_forward_evaluate(tensor_pool_t *pool_cpu,
                                   tensor_pool_t *pool_gpu,
                                   std::vector<execution_node_t *> &nodes);
void gradInitializer(std::vector<execution_node_t *> &nodes);
bool tensor_graph_backward(std::vector<execution_node_t *> &nodes);
void autogradGpuMemTranfer(std::vector<execution_node_t *> &nodes);
void tensor_sgd(std::vector<execution_node_t *> &nodes, float learning_rate);
bool execution_node_to_host(execution_node_t *node);
bool save_model(const std::string &filepath, const std::vector<tensor_t *> &weights);
bool load_model(const std::string &filepath, const std::vector<tensor_t *> &weights);
```

---

## Flat-C Python Bridge API (soft_cuda_python.h)

All symbols are prefixed with `sc_`. Types exposed are `sc_pool_t`, `sc_tensor_t`, and `sc_graph_t`, which are typedefs of the corresponding internal opaque structs. All functions have C linkage.

Backend mode constants:

```c
#define SC_BACKEND_GPU    0
#define SC_BACKEND_CPU    1
#define SC_BACKEND_HYBRID 2
```

Data type constants (used for `dtype` parameter):

| Constant | Value | C++ equivalent |
|---|---|---|
| `SC_DTYPE_UINT32` | 0 | `UINT32_T` |
| `SC_DTYPE_INT32` | 1 | `INT32_T` |
| `SC_DTYPE_UINT64` | 2 | `UINT64_T` |
| `SC_DTYPE_INT64` | 3 | `INT64_T` |
| `SC_DTYPE_FLOAT32` | 4 | `FLOAT32_T` |
| `SC_DTYPE_FLOAT64` | 5 | `FLOAT64_T` |

### Layer 1 (low-level primitives)

All graph operations from the C++ API are mirrored 1-to-1:

```c
sc_pool_t   *sc_pool_create(size_t capacity_bytes, int on_device);
void         sc_pool_destroy(sc_pool_t *pool);
void         sc_pool_zero(sc_pool_t *pool);
sc_tensor_t *sc_tensor_create(sc_pool_t *pool, int dtype, uint32_t num_dims,
                               uint32_t *dims, void *elems, int grad);
sc_graph_t  *sc_graph_create(void);
void         sc_graph_destroy(sc_graph_t *g);
int          sc_verify_dag(sc_pool_t *meta_pool, sc_tensor_t *t, sc_graph_t *g);
void         sc_assign_backend(sc_pool_t *pool_gpu, sc_graph_t *g, int mode);
void         sc_assign_grad_memory(sc_pool_t *pool_grad_cpu,
                                   sc_pool_t *pool_grad_gpu, sc_graph_t *g);
int          sc_graph_forward(sc_pool_t *pool_cpu, sc_pool_t *pool_gpu, sc_graph_t *g);
void         sc_grad_initializer(sc_graph_t *g);
int          sc_backward(sc_graph_t *g);
void         sc_sgd(sc_graph_t *g, float learning_rate);
int          sc_node_to_host(sc_graph_t *g, size_t node_idx);
void         sc_autograd_gpu_transfer(sc_graph_t *g);
```

### Layer 2 (convenience wrappers)

```c
sc_graph_t *sc_build_graph(sc_pool_t *meta_pool, sc_pool_t *pool_gpu,
                            sc_pool_t *pool_grad_cpu, sc_pool_t *pool_grad_gpu,
                            sc_tensor_t *loss, int backend_mode);

void  sc_graph_step(sc_pool_t *pool_cpu, sc_pool_t *pool_gpu,
                    sc_graph_t *g, float learning_rate);

float  sc_graph_get_loss(sc_graph_t *g);
size_t sc_graph_size(sc_graph_t *g);

int sc_save_model(const char *path, sc_tensor_t **tensors, size_t count);
int sc_load_model(const char *path, sc_tensor_t **tensors, size_t count);
```

`sc_build_graph` calls `verifyIfDAG`, `assignBackendGraph`, and `assignGradMemory` in sequence and returns a fully prepared graph handle.

`sc_graph_step` executes one complete training step: forward, grad zero, backward, and SGD update.

`sc_graph_get_loss` reads the scalar float value from the last node in the graph. If the last node is on the GPU, it first copies the data to host.

### Python Usage Example

```python
import ctypes

lib = ctypes.CDLL("./libsoft_cuda_python.so")

lib.sc_pool_create.restype  = ctypes.c_void_p
lib.sc_pool_create.argtypes = [ctypes.c_size_t, ctypes.c_int]

pool = lib.sc_pool_create(4 * 1024 * 1024, 0)  # 4 MB CPU pool
```

See `docs/PYTHON_BRIDGE.md` for a complete usage walkthrough.

---

## AOT Hardware Profiler

The `soft_profiler` binary measures actual CPU and GPU throughput for each supported operation at the following element-count breakpoints:

64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216

For each operation and each breakpoint, it runs 10 warmup iterations followed by 30 timed iterations, takes the median, and determines the smallest size at which GPU is faster. It then writes this as a `CONFIG.soft` JSON file to `~/.config/soft-cuda/CONFIG.soft`.

Staleness detection hashes the CUDA device UUID from `cudaDeviceProp`. If an existing `CONFIG.soft` already matches the current device hash, the profiler skips measurement and exits immediately.

The format written is:

```json
{
  "meta": { "soft_version": "0.1.0", "device_hash": "...", "generated_at": "..." },
  "device": { "type": "cuda", "compute_capability": 8.6, "vram_mb": 8192 },
  "ops": {
    "matmul": [
      { "min": 0,   "max": 127, "backend": "cpu" },
      { "min": 128, "max": 4294967295, "backend": "cuda" }
    ],
    "relu": [ { "backend": "cuda" } ]
  }
}
```

When `assignDevice` cannot find the op key in the JSON, or no range matches the element count, it falls back to CPU.

---

## Debug Build

In Debug mode (`-DCMAKE_BUILD_TYPE=Debug`), the CXX flags include `-O0 -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror -pedantic-errors -ggdb`. CUDA files compile with `-O0`.

The macro `SC_DEBUG` enables `cudaDeviceSynchronize` after each GPU kernel launch in `backprop_gpu_dispatch` and `tensor_sgd_gpu`, which makes CUDA errors synchronous and easier to attribute to the correct kernel.

In Release mode, `-O2 -DNDEBUG` is used for both CXX and CUDA.

CUDA separable compilation is enabled (`CUDA_SEPARABLE_COMPILATION ON`) on `soft_lib` and `soft_profiler` to allow device code to be split across translation units.

---

## Known Limitations and In-Progress Items

- Only `float32` is used in practice for all forward and backward computation. The `tensor_dtype_t` enum defines six types but the math implementations assert or assume `float`.
- GPU transpose is not implemented as a CUDA kernel. When a GPU node requires transpose, the data is copied to CPU, transposed there, and copied back.
- The `CAST` operation is defined in the enum but not implemented in either the forward or backward dispatchers.
- The `MUL_MATRIX` path (cache-optimised multiply) calls `tensor_transpose` on B internally and has a known TODO for verifying the GPU backward index when B is already transposed.
- Cross-entropy/softmax loss is commented out in `api.h` as a future addition.
- The HYBRID mode contagion logic (propagating GPU assignment through chains of connected nodes) has a TODO comment noting that the full contiguous-region logic is not yet implemented.
- `tensor_print_data` assumes 2-D layout and will not print higher-rank tensors correctly.
- Model save and load store no shape or dtype metadata, so the caller must manage that information externally.

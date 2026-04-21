// clang-format off
// PUBLIC DECLARATIONS
#include "../include/soft-cuda/tensor/api.h"
#include "../include/soft-cuda/tensor/debug_api.h"
#include "../include/soft-cuda/python/soft_cuda_python.h"

// PRIVATE DECLARATIONS
#include "./core/include/tensor/tensor.h"
#include "./core/include/pool/pool.h"
#include "backend_cpu/include/add.h"
#include "backend_cpu/include/sub.h"
#include "backend_cpu/include/mean.h"
#include "backend_cpu/include/debug.h"
#include "backend_cpu/include/square.h"
#include "backend_cpu/include/mul.h"
#include "backend_cpu/include/scalar.h"
#include "backend_cpu/include/transpose.h"
#include "backend_cpu/include/relu.h"
#include "backend_cpu/include/backprop/backprop_b.h"
#include "./core/include/graph/DAGbuild.h"
#include "./core/include/graph/assignBackend.h"
#include "./core/include/JSON/json_utils.h"
#include "./core/include/third_party/json.hpp"

#include "backend_gpu/include/math/add.h"
#include "backend_gpu/include/math/matmul.h"
// clang-format on

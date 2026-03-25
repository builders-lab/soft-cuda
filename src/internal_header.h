// PUBLIC DECLARATIONS
#include "../include/soft-cuda/tensor/api.h"
#include "../include/soft-cuda/tensor/debug_api.h"


// PRIVATE DECLARATIONS
#include "./core/include/tensor/tensor.h"
#include "./core/include/graph/DAGbuild.h"
#include "./core/include/graph/assignBackend.h"
#include "./core/include/pool/pool.h"
#include "backend_cpu/include/debug.h"
#include "backend_cpu/include/mul.h"
#include "backend_cpu/include/transpose.h"
#include "backend_cpu/include/scalar.h"
#include "backend_cpu/include/add.h"

#include <stdio.h>
#include "internal_header.h"
#include <stdlib.h>
#include <math.h>


__global__
void add(float *x, float *y, float *res, uint32_t size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size ; i += stride) {
    res[i] = x[i] + y[i];
  }
}

bool tensor_add_op_cuda(tensor_t *t,float *d_a, float *d_b, float *d_res) {
  int blockSize = 256;
  int numBlocks = (t->nvalues + blockSize -1) / blockSize;
  add<<<numBlocks, blockSize>>>(d_a,d_b,d_res,t->nvalues);

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
        debug("CUDA Add Kernel Failed: %s\n", cudaGetErrorString(err));
        return false;
  }

  return true;
}

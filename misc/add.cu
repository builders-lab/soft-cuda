#include <stdio.h>
#include <stdlib.h>
#include <math.h>


__global__
void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n ; i += stride) {
    y[i] = x[i] + y[i];
  }
}

int main(void) {
  int N = 1<<20;
  

  float *x,*y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));


  cudaMemLocation loc = {};
  loc.type = cudaMemLocationTypeDevice;
  loc.id   = 0;

  cudaMemPrefetchAsync(x, N*sizeof(float), loc, 0);
  cudaMemPrefetchAsync(y, N*sizeof(float), loc, 0);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  
  int blockSize = 256;
  int numBlocks = (N + blockSize -1) / blockSize;
  add<<<numBlocks, blockSize>>>(N,x,y);

  cudaDeviceSynchronize();

  float maxError = 0.0f;

  for(int i = 0; i <N; i++) {
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  }
  
  printf("Max Error: %f\n", maxError);

  cudaFree(x);
  cudaFree(y);
  return 0;
}

#include "cuda_runtime.h"

void case_good(void *device) {
  cudaStream_t producer;
  cudaStreamCreate(&producer);
  void *allocated;
  cudaMallocAsync(&allocated, 64, producer);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, 0); // label-site
}

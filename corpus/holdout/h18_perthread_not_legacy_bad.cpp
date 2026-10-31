#include "cuda_runtime.h"
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1

void case_bad(void *device) {
  cudaStream_t producer;
  cudaStreamCreate(&producer);
  void *allocated;
  cudaMallocAsync(&allocated, 64, producer);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, 0); // expected-warning{{CL3}}
}

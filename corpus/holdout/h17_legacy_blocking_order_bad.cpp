#include "cuda_runtime.h"

void case_bad(void *device) {
  cudaStream_t producer;
  cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking);
  void *allocated;
  cudaMallocAsync(&allocated, 64, producer);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, 0); // expected-warning{{CL3}}
}

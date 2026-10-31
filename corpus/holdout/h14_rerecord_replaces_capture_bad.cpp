#include "cuda_runtime.h"

void case_bad(void *device) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  void *allocated;
  cudaMallocAsync(&allocated, 64, a);
  cudaEventRecord(ready, a);
  cudaEventRecord(ready, c);
  cudaStreamWaitEvent(b, ready);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, b); // expected-warning{{CL3}}
}

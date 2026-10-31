#include "cuda_runtime.h"

void case_bad(void *device) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  void *allocated;
  cudaEventRecord(ready, a);
  cudaMallocAsync(&allocated, 64, a);
  cudaStreamWaitEvent(b, ready); // expected-warning{{CL4}}
  cudaEventRecord(ready, a);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, b); // expected-warning{{CL3}}
}

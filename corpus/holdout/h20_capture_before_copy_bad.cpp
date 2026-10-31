#include "cuda_runtime.h"

void case_bad(void *device, void *host) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, a);
  cudaMemcpyAsync(device, host, 64, cudaMemcpyHostToDevice, a);
  cudaStreamWaitEvent(b, ready); // expected-warning{{CL4}}
}

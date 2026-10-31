#include "cuda_runtime.h"

void case_good(void *device, void *host) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaMemcpyAsync(device, host, 64, cudaMemcpyHostToDevice, a);
  cudaMemcpyAsync(device, host, 32, cudaMemcpyHostToDevice, a);
  cudaEventRecord(ready, a);
  cudaStreamWaitEvent(b, ready); // label-site
}

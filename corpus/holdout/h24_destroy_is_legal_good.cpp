#include "cuda_runtime.h"

void case_good(void *device, void *host) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaMemcpyAsync(device, host, 64, cudaMemcpyHostToDevice, a);
  cudaEventDestroy(ready); // label-site
  cudaStreamDestroy(a);
}

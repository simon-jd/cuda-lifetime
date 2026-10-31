#include "cuda_runtime.h"

void case_bad(void *device) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  int *host;
  cudaMallocHost((void **)&host, 8 * sizeof(int));
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, a);
  cudaEventRecord(ready, a);
  cudaStreamWaitEvent(b, ready);
  int value = host[0]; // expected-warning{{CL1}}
}

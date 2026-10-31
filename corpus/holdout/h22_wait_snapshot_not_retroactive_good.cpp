#include "cuda_runtime.h"

void case_good(void *device) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  void *allocated;
  cudaMallocAsync(&allocated, 64, a);
  cudaEventRecord(ready, a);
  cudaStreamWaitEvent(b, ready);
  cudaEventRecord(ready, a);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, b); // label-site
}

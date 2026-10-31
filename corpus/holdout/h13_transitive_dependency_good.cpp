#include "cuda_runtime.h"

void case_good(void *device) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  void *allocated;
  cudaEvent_t first, second;
  cudaEventCreate(&first);
  cudaEventCreate(&second);
  cudaMallocAsync(&allocated, 64, a);
  cudaEventRecord(first, a);
  cudaStreamWaitEvent(b, first);
  cudaEventRecord(second, b);
  cudaStreamWaitEvent(c, second);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, c); // label-site
}

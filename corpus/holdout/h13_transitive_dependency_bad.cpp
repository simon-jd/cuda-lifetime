#include "cuda_runtime.h"

void case_bad(void *device) {
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
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, c); // expected-warning{{CL3}}
}

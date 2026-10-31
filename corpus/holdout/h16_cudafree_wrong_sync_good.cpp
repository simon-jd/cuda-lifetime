#include "cuda_runtime.h"

void case_good(void *device) {
  cudaStream_t a, b, c;
  cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
  void *allocated;
  cudaMallocAsync(&allocated, 64, a);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, a);
  cudaStreamSynchronize(a);
  cudaFree(allocated); // label-site
}

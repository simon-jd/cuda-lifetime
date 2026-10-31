#include "cuda_runtime.h"
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1

void case_good(void *device) {
  cudaStream_t producer;
  cudaStreamCreate(&producer);
  void *allocated;
  cudaMallocAsync(&allocated, 64, producer);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, producer);
  cudaStreamWaitEvent(0, ready);
  cudaMemcpyAsync(device, allocated, 64, cudaMemcpyDeviceToDevice, 0); // label-site
}

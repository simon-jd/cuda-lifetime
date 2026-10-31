#include "cuda_runtime.h"

void case_good(void *device, void *host, cudaStream_t stream) {
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaMemcpyAsync(device, host, 64, cudaMemcpyHostToDevice, stream);
  cudaEventRecord(ready, stream);
  cudaEventSynchronize(ready); // label-site
}

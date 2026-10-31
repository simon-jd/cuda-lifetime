#include "cuda_runtime.h"

void case_bad(void *device, void *host, cudaStream_t stream) {
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, stream);
  cudaMemcpyAsync(device, host, 64, cudaMemcpyHostToDevice, stream);
  cudaEventSynchronize(ready); // expected-warning{{CL4}}
}

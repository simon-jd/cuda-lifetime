#include "cuda_runtime.h"

void case_bad(void *device, int *host, cudaStream_t stream) {
  cudaHostRegister(host, 4 * sizeof(int), cudaHostRegisterDefault);
  cudaMemcpyAsync(device, host, 4 * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, stream);
  cudaHostUnregister(host); // expected-warning{{CL2}}
}

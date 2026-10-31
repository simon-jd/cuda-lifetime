#include "cuda_runtime.h"

void case_bad(void *device, cudaStream_t stream) {
  int *host;
  cudaMallocHost((void **)&host, 8 * sizeof(int));
  int *alias = host + 2;
  cudaMemcpyAsync(device, host, 8 * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, stream);
  alias[1] = 5; // expected-warning{{CL1}}
}

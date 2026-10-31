#include "cuda_runtime.h"

void cl1_pinned_h2d_bad(void *device, cudaStream_t producer, cudaStream_t consumer) {
  int *host;
  cudaHostAlloc((void **)&host, sizeof(int), cudaHostAllocDefault);
  cudaMemcpyAsync(device, host, sizeof(int), cudaMemcpyHostToDevice, producer);
  host[0] = 9; // expected-warning{{CL1}}
}

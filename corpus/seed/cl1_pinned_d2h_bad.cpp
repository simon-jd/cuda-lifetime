#include "cuda_runtime.h"

void cl1_pinned_d2h_bad(void *device, cudaStream_t producer, cudaStream_t consumer) {
  int *host;
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, producer);
  int value = host[0]; // expected-warning{{CL1}}
}

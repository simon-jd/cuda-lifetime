#include "cuda_runtime.h"

void case_bad(void *device, cudaStream_t stream) {
  int *host;
  cudaMallocHost((void **)&host, 8 * sizeof(int));
  cudaMemcpyAsync(host + 1, device, 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
  int value = host[2]; // expected-warning{{CL1}}
}

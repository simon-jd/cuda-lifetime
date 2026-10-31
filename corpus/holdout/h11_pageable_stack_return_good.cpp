#include "cuda_runtime.h"

void case_good(void *device, cudaStream_t stream) {
  int values[6] = {};
  cudaMemcpyAsync(values, device, sizeof(values), cudaMemcpyDeviceToHost, stream);
  cudaDeviceSynchronize();
  return; // label-site
}

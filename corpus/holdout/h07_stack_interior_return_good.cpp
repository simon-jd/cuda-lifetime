#include "cuda_runtime.h"

void case_good(void *device, cudaStream_t stream) {
  int data[8] = {};
  cudaHostRegister(data, sizeof(data), cudaHostRegisterDefault);
  cudaMemcpyAsync(device, data + 2, 3 * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaDeviceSynchronize();
  return; // label-site
}

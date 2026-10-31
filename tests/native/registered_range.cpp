#include "cuda_runtime_api.h"
int registered_span(cudaStream_t stream, void *device) {
  int buffer[4];
  cudaHostRegister(buffer + 1, 2 * sizeof(int), 0);
  cudaMemcpyAsync(buffer + 2, device, sizeof(int), 2, stream);
  int value = buffer[2]; // CL1 error: the copy target lies within the registered span
  cudaDeviceSynchronize();
  return value;
}

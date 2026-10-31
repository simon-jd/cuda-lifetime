#include "../native/runtime.h"
int registered_span(cudaStream_t stream, void *device) {
  int buffer[4];
  cudaHostRegister(buffer + 1, 2 * sizeof(int), 0);
  cudaMemcpyAsync(buffer + 2, device, sizeof(int), 2, stream);
  int value = buffer[2]; // review: CL1 error, copy target lies within registered span
  cudaDeviceSynchronize();
  return value;
}

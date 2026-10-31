#include "../native/runtime.h"
int equal_host_alias(int *host, int *alias, void *device, cudaStream_t stream) {
  if (host != alias) return 0;
  cudaHostRegister(host, sizeof(int), 0);
  cudaMemcpyAsync(host, device, sizeof(int), 2, stream);
  return *alias; // review: constrained-equal alias must produce CL1 error
}
void equal_ordered_alias(int *alias) {
  cudaStream_t a, b;
  cudaStreamCreateWithFlags(&a, 1);
  cudaStreamCreateWithFlags(&b, 1);
  int *device;
  cudaMallocAsync((void **)&device, sizeof(int), a);
  if (device != alias) return;
  int host = 1;
  cudaMemcpyAsync(alias, &host, sizeof(int), 1, b); // review: constrained-equal alias must produce CL3 error
  cudaDeviceSynchronize();
}

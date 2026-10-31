#include "../native/runtime.h"
void default_and_nonblocking_are_distinct(void *source) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 1);
  int *device;
  cudaMallocAsync((void **)&device, 4, nullptr);
  cudaMemcpyAsync(device, source, 4, 3, stream); // review: CL3 error, new nonblocking stream differs from default
  cudaDeviceSynchronize();
}

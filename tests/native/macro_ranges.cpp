#include "runtime.h"
#define READ_HOST(pointer) (*static_cast<int *>(pointer))
#define COPY_TO_HOST(host, device, stream) cudaMemcpyAsync(host, device, 4, 2, stream)
int macro_read(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  COPY_TO_HOST(host, device, stream);
  return READ_HOST(host); // CL1 error
}
int macro_completed(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  COPY_TO_HOST(host, device, stream);
  cudaStreamSynchronize(stream);
  return READ_HOST(host);
}

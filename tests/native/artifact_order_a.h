#include "cuda_runtime_api.h"
inline void *enqueue_in_header(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  return host;
}

#include "helpers.h"
template<class T> int template_copy(void *device, cudaStream_t stream) {
  T *host;
  cudaMallocHost((void **)&host, sizeof(T));
  cudaMemcpyAsync(host, device, sizeof(T), 2, stream);
  return *host;
}

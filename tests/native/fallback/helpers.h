// Header functions for fallback_roots (checked by run.py, not lit).
#include "../cuda_runtime_api.h"
inline float read_too_early(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  return host[0]; // the read the fallback root reports
}
// Calls no CUDA function itself: not a fallback root.
inline float wrapper(void *device, cudaStream_t stream) { return read_too_early(device, stream); }

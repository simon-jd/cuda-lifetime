#include "cuda_runtime_api.h"
void ordinary_free_relies_on_implicit_sync(void *source, cudaStream_t stream) {
  int *device;
  cudaMalloc((void **)&device, 4);
  cudaMemcpyAsync(device, source, 4, 3, stream);
  cudaFree(device); // CL3 note: ordinary cudaFree relies on implicit synchronization
}
void ordinary_free_after_completion(void *source, cudaStream_t stream) {
  int *device;
  cudaMalloc((void **)&device, 4);
  cudaMemcpyAsync(device, source, 4, 3, stream);
  cudaStreamSynchronize(stream);
  cudaFree(device); // no diagnostic
}

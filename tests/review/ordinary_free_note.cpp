#include "../native/runtime.h"
void ordinary_free_relies_on_implicit_sync(void *source, cudaStream_t stream) {
  int *device;
  cudaMalloc((void **)&device, 4);
  cudaMemcpyAsync(device, source, 4, 3, stream);
  cudaFree(device); // review: CL3 note, ordinary free never definite or contractual here
}
void ordinary_free_after_completion(void *source, cudaStream_t stream) {
  int *device;
  cudaMalloc((void **)&device, 4);
  cudaMemcpyAsync(device, source, 4, 3, stream);
  cudaStreamSynchronize(stream);
  cudaFree(device); // review: no diagnostic
}

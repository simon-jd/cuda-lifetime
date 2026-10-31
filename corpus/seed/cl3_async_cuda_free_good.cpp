#include "cuda_runtime.h"

void cl3_async_cuda_free_good(void *device, cudaStream_t producer, cudaStream_t consumer) {
  void *allocated;
  cudaMallocAsync(&allocated, 4, producer);
  cudaMemcpyAsync(device, allocated, 4, cudaMemcpyDeviceToDevice, producer);
  cudaStreamSynchronize(producer);
  cudaFree(allocated); // label-site
}

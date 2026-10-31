#include "cuda_runtime.h"

void cl3_cross_stream_alloc_bad(void *device, cudaStream_t producer, cudaStream_t consumer) {
  if (producer == consumer) return;
  void *allocated;
  cudaMallocAsync(&allocated, 4, producer);
  cudaMemcpyAsync(device, allocated, 4, cudaMemcpyDeviceToDevice, consumer); // expected-warning{{CL3}}
}

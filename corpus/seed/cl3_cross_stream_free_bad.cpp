#include "cuda_runtime.h"

void cl3_cross_stream_free_bad(void *device, cudaStream_t producer, cudaStream_t consumer) {
  if (producer == consumer) return;
  void *allocated;
  cudaMallocAsync(&allocated, 4, producer);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, producer);
  cudaStreamWaitEvent(consumer, ready);
  cudaMemcpyAsync(device, allocated, 4, cudaMemcpyDeviceToDevice, consumer);
  cudaFreeAsync(allocated, producer); // expected-warning{{CL3}}
}

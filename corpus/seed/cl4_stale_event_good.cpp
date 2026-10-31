#include "cuda_runtime.h"

void cl4_stale_event_good(void *device, cudaStream_t producer, cudaStream_t consumer) {
  if (producer == consumer) return;
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  void *allocated;
  cudaMallocAsync(&allocated, 4, producer);
  cudaEventRecord(ready, producer);
  cudaStreamWaitEvent(consumer, ready); // label-site
}

#include "cuda_runtime.h"

void cl4_stale_event_bad(void *device, cudaStream_t producer, cudaStream_t consumer) {
  if (producer == consumer) return;
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, producer);
  void *allocated;
  cudaMallocAsync(&allocated, 4, producer);
  cudaStreamWaitEvent(consumer, ready); // expected-warning{{CL4}}
}

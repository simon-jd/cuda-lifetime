#include "cuda_runtime_api.h"
int stream_alias_sync(cudaStream_t stream, cudaStream_t alias, void *device) {
  if (stream != alias) return 0;
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaStreamSynchronize(alias);
  return *host; // no CL1: an equal stream was synchronized
}
int event_alias_sync(cudaStream_t stream, cudaEvent_t event, cudaEvent_t alias, void *device) {
  if (event != alias) return 0;
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(event, stream);
  cudaEventSynchronize(alias); // no CL4: an equal event has a capture
  return *host; // no CL1
}

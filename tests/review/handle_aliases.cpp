#include "../native/runtime.h"
int stream_alias_sync(cudaStream_t stream, cudaStream_t alias, void *device) {
  if (stream != alias) return 0;
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaStreamSynchronize(alias);
  return *host; // review: no CL1, equal stream was synchronized
}
int event_alias_sync(cudaStream_t stream, cudaEvent_t event, cudaEvent_t alias, void *device) {
  if (event != alias) return 0;
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(event, stream);
  cudaEventSynchronize(alias); // review: no CL4, equal event has capture
  return *host; // review: no CL1
}

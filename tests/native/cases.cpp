#include "runtime.h"
int early_read(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  return *static_cast<int *>(host); // CL1
}
int synchronized_read(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaStreamSynchronize(stream);
  return *static_cast<int *>(host);
}
void early_free(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaFreeHost(host); // CL2
}
void wrong_stream(void *device) {
  cudaStream_t one, two;
  cudaStreamCreateWithFlags(&one, 1);
  cudaStreamCreateWithFlags(&two, 1);
  void *allocated;
  cudaMallocAsync(&allocated, 4, one);
  cudaMemcpyAsync(device, allocated, 4, 3, two); // CL3
}
void event_dependency(void *device) {
  cudaStream_t one, two;
  cudaStreamCreateWithFlags(&one, 1);
  cudaStreamCreateWithFlags(&two, 1);
  cudaEvent_t event;
  cudaEventCreate(&event);
  void *allocated;
  cudaMallocAsync(&allocated, 4, one);
  cudaEventRecord(event, one);
  cudaStreamWaitEvent(two, event, 0);
  cudaMemcpyAsync(device, allocated, 4, 3, two);
}
void empty_event(cudaStream_t stream) {
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaStreamWaitEvent(stream, event, 0); // CL4
}
void stale_allocation_event(cudaStream_t one, cudaStream_t two) {
  if (one == two) return;
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, one);
  void *allocated;
  cudaMallocAsync(&allocated, 4, one);
  cudaStreamWaitEvent(two, event, 0); // CL4
}
void reuse_after_free(void *device, cudaStream_t stream) {
  void *allocated;
  cudaMallocAsync(&allocated, 4, stream);
  cudaFreeAsync(allocated, stream);
  cudaMemcpyAsync(device, allocated, 4, 3, stream); // CL3
}
void unknown_flags(void *device, unsigned flags) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, flags);
  void *allocated;
  cudaMallocAsync(&allocated, 4, stream);
  cudaMemcpyAsync(device, allocated, 4, 3, nullptr); // CL3
}
void blocking_flags(void *device) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 0);
  void *allocated;
  cudaMallocAsync(&allocated, 4, stream);
  cudaMemcpyAsync(device, allocated, 4, 3, nullptr);
}
void ordinary_free_relies_on_sync(void *device, cudaStream_t stream) {
  void *allocated;
  cudaMalloc(&allocated, 4);
  cudaMemcpyAsync(device, allocated, 4, 3, stream);
  cudaFree(allocated); // CL3
}
int checked_event_query(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(ready, stream);
  if (cudaEventQuery(ready) == 0) return *static_cast<int *>(host);
  return 0;
}
int ignored_event_query(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(ready, stream);
  cudaEventQuery(ready);
  return *static_cast<int *>(host); // CL1
}
int checked_stream_query(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  if (cudaStreamQuery(stream) == 0) return *static_cast<int *>(host);
  return 0;
}
void *allocate_in_callee(cudaStream_t stream) {
  void *allocated;
  cudaMallocAsync(&allocated, 4, stream);
  return allocated;
}
void callee_allocation_wrong_stream(void *device) {
  cudaStream_t one, two;
  cudaStreamCreateWithFlags(&one, 1);
  cudaStreamCreateWithFlags(&two, 1);
  void *allocated = allocate_in_callee(one);
  cudaMemcpyAsync(device, allocated, 4, 3, two); // CL3
}

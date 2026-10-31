#include "../native/runtime.h"
int wait_snapshot_survives_rerecord(void *device) {
  cudaStream_t a, b;
  cudaStreamCreateWithFlags(&a, 1);
  cudaStreamCreateWithFlags(&b, 1);
  cudaEvent_t event;
  cudaEventCreate(&event);
  int *first, *later;
  cudaMallocHost((void **)&first, 4);
  cudaMallocHost((void **)&later, 4);
  cudaMemcpyAsync(first, device, 4, 2, a);
  cudaEventRecord(event, a);
  cudaStreamWaitEvent(b, event, 0);
  cudaMemcpyAsync(later, device, 4, 2, a);
  cudaEventRecord(event, a);
  cudaStreamSynchronize(b);
  int x = *first; // review: completed by old capture
  int y = *later; // review: CL1 error, re-record cannot strengthen prior wait
  cudaDeviceSynchronize();
  return x + y;
}
int event_transitive_completion(void *device) {
  cudaStream_t a, b;
  cudaStreamCreateWithFlags(&a, 1);
  cudaStreamCreateWithFlags(&b, 1);
  cudaEvent_t first, second;
  cudaEventCreate(&first);
  cudaEventCreate(&second);
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, a);
  cudaEventRecord(first, a);
  cudaStreamWaitEvent(b, first, 0);
  cudaEventRecord(second, b);
  cudaEventSynchronize(second);
  return *host; // review: no diagnostic, second event captures dependency on a
}
int wait_is_not_host_completion(void *device) {
  cudaStream_t a, b;
  cudaStreamCreateWithFlags(&a, 1);
  cudaStreamCreateWithFlags(&b, 1);
  cudaEvent_t event;
  cudaEventCreate(&event);
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, a);
  cudaEventRecord(event, a);
  cudaStreamWaitEvent(b, event, 0);
  int value = *host; // review: CL1 error, no host completion
  cudaDeviceSynchronize();
  return value;
}

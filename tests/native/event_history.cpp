#include "cuda_runtime_api.h"
// Events whose history the path does not show, and waits already ordered.
void event_from_caller(cudaStream_t stream, cudaEvent_t event) {
  cudaStreamWaitEvent(stream, event, 0);
}
struct Holder { cudaEvent_t event; };
void event_from_object(cudaStream_t stream, Holder *holder) {
  cudaEventSynchronize(holder->event);
  cudaStreamWaitEvent(stream, holder->event, 0);
}
void created_but_never_recorded(cudaStream_t stream) {
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaStreamWaitEvent(stream, event, 0); // CL4 warning
}
void later_work_is_on_the_waiter() {
  cudaStream_t one, two;
  cudaStreamCreateWithFlags(&one, 1);
  cudaStreamCreateWithFlags(&two, 1);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  void *allocated;
  cudaMalloc(&allocated, 16);
  cudaEventRecord(a, one);
  cudaStreamWaitEvent(two, a, 0);
  cudaEventRecord(b, two);
  cudaMemsetAsync(allocated, 0, 16, one);
  cudaStreamWaitEvent(one, b, 0);
  cudaStreamSynchronize(one);
}
void later_work_on_another_stream() {
  cudaStream_t one, two;
  cudaStreamCreateWithFlags(&one, 1);
  cudaStreamCreateWithFlags(&two, 1);
  cudaEvent_t event;
  cudaEventCreate(&event);
  void *allocated;
  cudaMalloc(&allocated, 16);
  cudaEventRecord(event, one);
  cudaMemsetAsync(allocated, 0, 16, one);
  cudaStreamWaitEvent(two, event, 0); // CL4 warning
  cudaStreamSynchronize(one);
}

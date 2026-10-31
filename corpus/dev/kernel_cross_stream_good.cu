#include <cuda_runtime.h>

__global__ void consume_good(int *pointer) { pointer[0] = 42; }

void kernel_cross_stream_good(cudaStream_t producer, cudaStream_t consumer) {
  if (producer == consumer) return;
  int *device;
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaMallocAsync((void **)&device, sizeof(int), producer);
  cudaEventRecord(ready, producer);
  cudaStreamWaitEvent(consumer, ready);
  consume_good<<<1, 1, 0, consumer>>>(device); // label-site
}

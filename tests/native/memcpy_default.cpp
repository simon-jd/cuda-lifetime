#include "cuda_runtime_api.h"
// cudaMemcpyDefault (kind 4) takes its direction from the operands' memory kinds.
int default_d2h_pinned(cudaStream_t stream) {
  void *device;
  int *host;
  cudaMalloc(&device, sizeof(int));
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), 4, stream);
  return *host; // CL1 error: device to pinned host, read before completion
}
int default_d2h_pageable(cudaStream_t stream) {
  void *device;
  int host[1];
  cudaMalloc(&device, sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), 4, stream);
  int value = host[0]; // CL1 warning: device to pageable host, read before completion
  cudaStreamSynchronize(stream);
  return value;
}
int default_h2d_pinned_write(cudaStream_t stream) {
  void *device;
  int *host;
  cudaMalloc(&device, sizeof(int));
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(device, host, sizeof(int), 4, stream);
  *host = 1; // CL1 error: the H2D source is overwritten before completion
  cudaStreamSynchronize(stream);
  return 0;
}
int default_h2d_pinned_read(cudaStream_t stream) {
  void *device;
  int *host;
  cudaMalloc(&device, sizeof(int));
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(device, host, sizeof(int), 4, stream);
  int value = *host; // no CL1: reading an H2D source is allowed
  cudaStreamSynchronize(stream);
  return value;
}
int default_d2h_synchronized(cudaStream_t stream) {
  void *device;
  int *host;
  cudaMalloc(&device, sizeof(int));
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), 4, stream);
  cudaStreamSynchronize(stream);
  return *host; // no CL1: the stream was synchronized
}

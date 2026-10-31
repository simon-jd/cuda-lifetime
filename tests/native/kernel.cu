#include "cuda_runtime_api.h"
struct dim3 { unsigned x, y, z; constexpr dim3(unsigned x=1, unsigned y=1, unsigned z=1):x(x),y(y),z(z){} };
extern int __cudaPushCallConfiguration(dim3, dim3, unsigned long=0, cudaStream_t=0);
__attribute__((global)) void use(int *);
void bad() {
  cudaStream_t one, two;
  cudaStreamCreateWithFlags(&one, 1);
  cudaStreamCreateWithFlags(&two, 1);
  void *memory;
  cudaMallocAsync(&memory, 4, one);
  use<<<1, 1, 0, two>>>(static_cast<int *>(memory)); // CL3
}
void good() {
  cudaStream_t one;
  cudaStreamCreateWithFlags(&one, 1);
  void *memory;
  cudaMallocAsync(&memory, 4, one);
  use<<<1, 1, 0, one>>>(static_cast<int *>(memory));
}
void reserved_legacy_bad() {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 1);
  void *memory;
  cudaMallocAsync(&memory, 4, stream);
  use<<<1, 1, 0, (cudaStream_t)1>>>(static_cast<int *>(memory)); // CL3
}
void reserved_per_thread_bad() {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 1);
  void *memory;
  cudaMallocAsync(&memory, 4, stream);
  use<<<1, 1, 0, (cudaStream_t)2>>>(static_cast<int *>(memory)); // CL3
}
void reserved_same_stream_good() {
  void *legacy, *perThread, *defaultStream;
  cudaMallocAsync(&legacy, 4, (cudaStream_t)1);
  use<<<1, 1, 0, (cudaStream_t)1>>>(static_cast<int *>(legacy));
  cudaMallocAsync(&perThread, 4, (cudaStream_t)2);
  use<<<1, 1, 0, (cudaStream_t)2>>>(static_cast<int *>(perThread));
  cudaMallocAsync(&defaultStream, 4, nullptr);
  use<<<1, 1, 0, nullptr>>>(static_cast<int *>(defaultStream));
}

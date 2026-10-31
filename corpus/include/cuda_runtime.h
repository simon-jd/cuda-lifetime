#ifndef CUDA_LIFETIME_TEST_CUDA_RUNTIME_H
#define CUDA_LIFETIME_TEST_CUDA_RUNTIME_H

// Declaration-only analysis fixture. This is not the CUDA SDK and must never be
// used to establish real toolkit compatibility or runtime behavior.
#include <stddef.h>

struct CUstream_st;
struct CUevent_st;
typedef CUstream_st *cudaStream_t;
typedef CUevent_st *cudaEvent_t;
typedef int cudaError_t;
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};
enum { cudaSuccess = 0, cudaStreamDefault = 0, cudaStreamNonBlocking = 1,
       cudaHostAllocDefault = 0, cudaHostRegisterDefault = 0 };
#define cudaStreamLegacy ((cudaStream_t)1)
#define cudaStreamPerThread ((cudaStream_t)2)

extern "C" {
cudaError_t cudaMallocHost(void **, size_t);
cudaError_t cudaHostAlloc(void **, size_t, unsigned);
cudaError_t cudaHostRegister(void *, size_t, unsigned);
cudaError_t cudaHostUnregister(void *);
cudaError_t cudaFreeHost(void *);
cudaError_t cudaMalloc(void **, size_t);
cudaError_t cudaMallocAsync(void **, size_t, cudaStream_t);
cudaError_t cudaFree(void *);
cudaError_t cudaFreeAsync(void *, cudaStream_t);
cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind,
                            cudaStream_t = 0);
cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t = 0);
cudaError_t cudaStreamCreate(cudaStream_t *);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0);
cudaError_t cudaEventDestroy(cudaEvent_t);
void *malloc(size_t);
void free(void *);
}
#endif

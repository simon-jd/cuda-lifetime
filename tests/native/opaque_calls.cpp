#include "runtime.h"
// Calls CSA cannot see into, and calls it inlines.
extern void reads(const void *);
extern void writes(void *);
extern void logs(const char *, ...);
struct View { float *data; };
extern void consume(const View *);
inline const float *address_of(const float *p, int i) { return p + i; }
inline float first(const float *p) { return p[0]; } // CL1 error
void const_call_on_pending_h2d(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(device, host, 16, 1, stream);
  reads(host);  // read/read: the copy only reads the source too
  cudaStreamSynchronize(stream);
}
void const_call_on_pending_d2h(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(host, device, 16, 2, stream);
  reads(host); // CL1 warning
  cudaStreamSynchronize(stream);
}
void mutable_call_on_pending_h2d(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(device, host, 16, 1, stream);
  writes(host); // CL1 warning
  cudaStreamSynchronize(stream);
}
void variadic_call_on_pending_h2d(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(device, host, 16, 1, stream);
  logs("%p", host); // CL1 warning
  cudaStreamSynchronize(stream);
}
void indirect_escape(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(host, device, 16, 2, stream);
  View view{host};
  consume(&view); // CL1 warning
  cudaStreamSynchronize(stream);
}
void inlined_address_only(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(host, device, 16, 2, stream);
  const float *second = address_of(host, 1);
  cudaStreamSynchronize(stream);
  (void)second;
}
float inlined_read(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(host, device, 16, 2, stream);
  return first(host);
}

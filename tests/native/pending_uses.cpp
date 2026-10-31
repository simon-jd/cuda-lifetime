#include "runtime.h"
struct Buffer { int value; void opaque_member(); };
extern void opaque_argument(void *);
void *escaped;
void pending_argument(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  opaque_argument(host); // CL1 warning
  cudaStreamSynchronize(stream);
}
void completed_argument(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaStreamSynchronize(stream);
  opaque_argument(host);
}
void pending_member(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, sizeof(Buffer));
  cudaMemcpyAsync(host, device, sizeof(Buffer), 2, stream);
  static_cast<Buffer *>(host)->opaque_member(); // CL1 warning
  cudaStreamSynchronize(stream);
}
void completed_member(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, sizeof(Buffer));
  cudaMemcpyAsync(host, device, sizeof(Buffer), 2, stream);
  cudaStreamSynchronize(stream);
  static_cast<Buffer *>(host)->opaque_member();
}
void pending_escape(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  escaped = host; // CL1 warning
  cudaStreamSynchronize(stream);
}
void completed_escape(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaStreamSynchronize(stream);
  escaped = host;
}
struct cudaDeviceProp { cudaDeviceProp(); };
extern void cuda_device_id();
void non_api_names() {
  cudaDeviceProp prop;
  cuda_device_id();
  cudaDeviceSynchronize();
}

// RUN: %native %s --sarif %t.sarif --coverage %t.json -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
#include "cuda_runtime_api.h"
// A stream wrapper whose accessors are defined in another translation unit,
// as RMM 26's cuda_stream_view: each call reads the wrapper's only member,
// so a copy queued on view.value() and a wait on the same view match.
class stream_handle {
public:
  cudaStream_t value() const;
  operator cudaStream_t() const;
  cudaStream_t take();
private:
  cudaStream_t stream_;
};
float accessor_then_conversion(void *device, stream_handle stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream.value());
  cudaStreamSynchronize(stream);
  return host[0];
}
float other_stream(void *device, stream_handle stream, stream_handle other) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream.value());
  cudaStreamSynchronize(other.value());
  return host[0]; // CL1 error
}
// A non-const member may return anything: the wait is not known to cover the copy.
float non_const_accessor(void *device, stream_handle stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream.value());
  cudaStreamSynchronize(stream.take());
  return host[0]; // CL1 error
}

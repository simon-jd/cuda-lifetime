// RUN: %native %s --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
#include "host_containers.h"
// A host container's destructor releases the buffer data() returned.
void vector_destroyed(void *device, cudaStream_t stream) {
  std::vector<int> host(4);
  cudaMemcpyAsync(device, host.data(), 16, 1, stream);
  return; // CL2 warning
}
void vector_synchronized(void *device, cudaStream_t stream) {
  std::vector<int> host(4);
  cudaMemcpyAsync(device, host.data(), 16, 1, stream);
  cudaStreamSynchronize(stream);
}
// With cudaMemcpyDefault the container makes the source known host memory.
void block_scoped_vector(void *device, cudaStream_t stream) {
  {
    std::vector<int> host(4);
    cudaMemcpyAsync(device, host.data(), 16, 4, stream);
  } // CL2 warning
  cudaStreamSynchronize(stream);
}
// Through a view constructed from the container, as with cuDF's host_span.
void through_view(void *device, stream_view stream) {
  std::vector<int> host(4);
  copy_span_async<int>(device, std::span<const int>(host), stream);
  return; // CL2 warning
}
void through_view_synchronized(void *device, stream_view stream) {
  std::vector<int> host(4);
  copy_span_async<int>(device, std::span<const int>(host), stream);
  cudaStreamSynchronize(stream.value());
}
// A derived container through a view whose data() is inherited.
void derived_container_inherited_data(void *device, stream_view stream) {
  host_vector<char> buffer(64);
  copy_host_span_async<char>(device, cudf::host_span<const char>(buffer), stream);
  return; // CL2 warning
}
// The view comes from the container's conversion operator, as in cuDF.
void view_from_conversion(void *device, stream_view stream) {
  host_vector<char> buffer(64);
  const host_vector<char> &view = buffer;
  copy_host_span_async<char>(device, view, stream);
  return; // CL2 warning
}
void view_through_braces_and_copies(void *device, stream_view stream, bool copy) {
  if (copy) {
    host_vector<long> buffer(8);
    copy_container_async(device, buffer, stream);
  } // CL2 warning
  cudaStreamSynchronize(stream.value());
}
void derived_container_synchronized(void *device, stream_view stream) {
  host_vector<char> buffer(64);
  copy_host_span_async<char>(device, cudf::host_span<const char>(buffer), stream);
  cudaStreamSynchronize(stream.value());
}
// A view of memory the caller owns ends nothing.
void view_of_caller_memory(void *device, int *values, stream_view stream) {
  copy_span_async<int>(device, std::span<const int>(values, 4), stream);
}
// A container that outlives the function is not released here.
void caller_container(void *device, std::vector<int> &host, cudaStream_t stream) {
  cudaMemcpyAsync(device, host.data(), 16, 1, stream);
}
// Returning or moving a container passes its buffer on: the moved-from
// object releases nothing, and the new owner's end is what counts.
std::vector<int> returned_by_value(void *device, cudaStream_t stream) {
  std::vector<int> out(4);
  cudaMemcpyAsync(out.data(), device, 16, 2, stream);
  return out;
}
void caller_destroys_returned(void *device, cudaStream_t stream) {
  std::vector<int> host = returned_by_value(device, stream);
  return; // CL2 warning
}
void caller_synchronizes_returned(void *device, cudaStream_t stream) {
  std::vector<int> host = returned_by_value(device, stream);
  cudaStreamSynchronize(stream);
}
void moved_into_longer_lived(void *device, std::vector<int> &keep, cudaStream_t stream) {
  std::vector<int> host(4);
  cudaMemcpyAsync(device, host.data(), 16, 1, stream);
  keep = static_cast<std::vector<int> &&>(host);
}

// The other translation unit of cross_tu.cpp: copy helpers whose bodies the
// caller's translation unit does not contain.
#include "host_containers.h"
namespace {
int queue(void *dst, const void *src, unsigned long count, cudaStream_t stream) {
  return cudaMemcpyAsync(dst, src, count, 4, stream);
}
}
void copy_to_device_async(void *dst, const void *src, unsigned long bytes, stream_view stream) {
  if (bytes == 0) return;
  queue(dst, src, bytes, stream.value());
}
// Synchronizes after copying, so it gets no summary.
void copy_and_wait(void *dst, const void *src, unsigned long bytes, stream_view stream) {
  queue(dst, src, bytes, stream.value());
  cudaStreamSynchronize(stream.value());
}

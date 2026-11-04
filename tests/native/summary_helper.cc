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
[[noreturn]] void fail(int status);
// An error path that synchronizes and then throws does not end the copy for the caller.
// Like cuDF's throw_cuda_error: not declared noreturn, but every path throws.
void throw_error(int status) {
  cudaDeviceSynchronize();
  if (status > 1) throw status;
  else fail(status);
}
void checked_copy(void *dst, const void *src, unsigned long bytes, stream_view stream) {
  if (int status = queue(dst, src, bytes, stream.value())) throw_error(status);
}
// Synchronizes after copying: its summary is the synchronization.
void copy_and_wait(void *dst, const void *src, unsigned long bytes, stream_view stream) {
  queue(dst, src, bytes, stream.value());
  cudaStreamSynchronize(stream.value());
}
// Like cuDF's memcpy_batch_async: arrays of operands.
int copy_batch(void **dsts, const void **srcs, const unsigned long *sizes, unsigned long count, stream_view stream) {
  if (count > 1) return cudaMemcpyBatchAsync(dsts, srcs, sizes, count, nullptr, nullptr, 0, stream.value());
  for (unsigned long i = 0; i < count; ++i) cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], 4, stream.value());
  return 0;
}
// Like cuDF's memcpy_async: a single copy through the batch helper.
void copy_one(void *dst, const void *src, unsigned long count, stream_view stream) {
  copy_batch(&dst, &src, &count, 1, stream);
}
// Synchronizes the receiver's stream on every path that returns.
void stream_view::synchronize() const {
  if (int status = cudaStreamSynchronize(stream_)) throw_error(status);
}
// Returns before its throw when the status is success, like RAFT's
// check_cuda_error: not an error path that never returns.
void check_status(int status) {
  if (status == 0) return;
  throw status;
}
// Synchronizes on one branch; the other checks a status and returns: no summary.
void sync_unless_failed(stream_view stream, int status) {
  if (status) check_status(status);
  else cudaStreamSynchronize(stream.value());
}
// Synchronizes on some paths only: no summary.
void maybe_sync(stream_view stream, bool wait) {
  if (wait) cudaStreamSynchronize(stream.value());
}
void device_buffer::copy_async(const void *src, unsigned long bytes) {
  cudaMemcpyAsync(data_, src, bytes, 4, stream().value());
}

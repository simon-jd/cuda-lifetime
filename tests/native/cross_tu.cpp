// RUN: rm -rf %t.models && mkdir -p %t.models
// RUN: %native %S/summary_helper.cc --summarize %t.models/helper.json -- -std=c++17
// RUN: %python %S/expect_models.py %t.models/helper.json "copy_to_device_async dst=0 src=1 size=2 stream=3 kind=4" "copy_batch dst=0[] src=1[] size=2[] count=3 stream=4 kind=4" "copy_one dst=0 src=1 size=2 stream=3 kind=4" "checked_copy dst=0 src=1 size=2 stream=3 kind=4" "stream_view::synchronize stream=this" "copy_and_wait stream=3"
// RUN: %native %s --model %t.models --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
#include "host_containers.h"
// copy_to_device_async is defined in another translation unit. Its summary,
// generated from that TU, queues a cudaMemcpyDefault copy from parameter 1.
void stack_source(void *device, stream_view stream) {
  int value = 7;
  copy_to_device_async(device, &value, sizeof value, stream);
  return; // CL2 warning
}
void stack_source_synchronized(void *device, stream_view stream) {
  int value = 7;
  copy_to_device_async(device, &value, sizeof value, stream);
  cudaStreamSynchronize(stream.value());
}
// The wrapper names the same stream as the handle it holds.
void stack_source_synchronized_raw(void *device, cudaStream_t raw) {
  stream_view stream{raw};
  int value = 7;
  copy_to_device_async(device, &value, sizeof value, stream);
  cudaStreamSynchronize(raw);
}
void helper_that_waits(void *device, stream_view stream) {
  int value = 7;
  copy_and_wait(device, &value, sizeof value, stream);
}
// A container method CSA does not inline is applied through its body's summary.
void element_from_block_scope(device_vector<int> &column, stream_view stream) {
  int single = 1;
  column.set_element_async(1, single, stream);
  return; // CL2 warning
}
void element_synchronized(device_vector<int> &column, stream_view stream) {
  int single = 1;
  column.set_element_async(1, single, stream);
  cudaStreamSynchronize(stream.value());
}
// A copy through a batched helper, as cuDF's memcpy_async does.
void through_batch_helper(void *device, stream_view stream) {
  int value = 7;
  copy_one(device, &value, sizeof value, stream);
  return; // CL2 warning
}
// cudaMemcpyBatchAsync takes arrays of operands; the first entry is modeled.
void batch_entry(void *device, cudaStream_t stream) {
  int value = 7;
  void *dsts[1] = {device};
  const void *srcs[1] = {&value};
  unsigned long sizes[1] = {sizeof value};
  cudaMemcpyBatchAsync(dsts, srcs, sizes, 1, nullptr, nullptr, 0, stream);
  return; // CL2 warning
}
// A synchronization defined in another TU, on the receiver, as RMM's
// cuda_stream_view::synchronize.
void synchronized_through_member(void *device, stream_view stream) {
  int value = 7;
  copy_to_device_async(device, &value, sizeof value, stream);
  stream.synchronize();
}
void maybe_synchronized(void *device, stream_view stream, bool wait) {
  int value = 7;
  copy_to_device_async(device, &value, sizeof value, stream);
  maybe_sync(stream, wait);
  return; // CL2 warning
}

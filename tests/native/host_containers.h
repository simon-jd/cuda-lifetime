// Minimal stand-ins for library types the checker models by name: an owning
// host container (std::vector), a view (std::span), a stream wrapper holding
// the handle as its only member, and a device container whose element setter
// queues a copy from the caller's value. CSA does not inline methods of
// container classes defined in headers.
#pragma once
#include "cuda_runtime_api.h"
namespace std {
template <class T> class vector {
  T *begin_ = nullptr;
public:
  explicit vector(unsigned long count);
  vector(const vector &);
  ~vector();
  T *data();
  T *begin();
  unsigned long size() const;
};
template <class T> class span {
  T *data_;
  unsigned long size_;
public:
  template <class U> span(vector<U> &values);
  span(T *data, unsigned long size);
  T *data() const;
  T *begin() const;
  unsigned long size() const;
};
}
// Like cuDF's host_vector and host_span: a derived container, and a view
// whose data() is inherited from a base class.
template <class T> struct span_base {
  T *data_;
  T *data() const;
  T *begin() const;
  unsigned long size() const;
};
namespace cudf {
template <class T> struct host_span : span_base<T> {
  template <class C> host_span(C &values);
};
}
template <class T> class host_vector : public std::vector<T> {
public:
  explicit host_vector(unsigned long count);
  ~host_vector();
  operator cudf::host_span<const T>() const { return cudf::host_span<const T>(*this); }
};
struct stream_view {
  cudaStream_t stream_;
  cudaStream_t value() const { return stream_; }
};
template <class T> class device_vector {
  T *data_;
public:
  T *begin() { return data_; }
  void set_element_async(unsigned long index, const T &value, stream_view stream) {
    cudaMemcpyAsync(data_ + index, &value, sizeof(value), 4, stream.value());
  }
};
// Defined in summary_helper.cc and summarized from that translation unit.
void copy_to_device_async(void *dst, const void *src, unsigned long bytes, stream_view stream);
void copy_and_wait(void *dst, const void *src, unsigned long bytes, stream_view stream);
void copy_one(void *dst, const void *src, unsigned long count, stream_view stream);
// Like cuDF's cuda_memcpy_async(device_span, host_span, stream).
template <class T> void copy_span_async(void *dst, std::span<const T> src, stream_view stream) {
  cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(T), 4, stream.value());
}
template <class T> void copy_host_span_async(void *dst, cudf::host_span<const T> src, stream_view stream) {
  cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(T), 4, stream.value());
}

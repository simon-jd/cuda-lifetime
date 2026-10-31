#include "../cuda_runtime_api.h"
inline int header_copy(void *device, cudaStream_t stream) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  return *host;
}
struct HeaderDtor { ~HeaderDtor() { cudaDeviceSynchronize(); } };
struct Virtual { virtual void call() = 0; };
#define DEFINE_CUDA_FUNCTION() int macro_copy(void *device, cudaStream_t stream) { return header_copy(device, stream); }
struct HeaderCtor {
  int value;
  HeaderCtor() : value((cudaDeviceSynchronize(), 0)) {}
};
struct MemberDtor { HeaderDtor member; };
struct DerivedDtor : HeaderDtor {};

// A CUDA-named type's ordinary constructor is not a CUDA Runtime API call.
struct cudaDeviceProp { int value; cudaDeviceProp() : value(0) {} };
int host_constructor_only() { cudaDeviceProp prop; return prop.value; }

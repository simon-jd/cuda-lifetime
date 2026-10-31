#include "helpers.h"
int uses_header(void *device, cudaStream_t stream) { return header_copy(device, stream); }

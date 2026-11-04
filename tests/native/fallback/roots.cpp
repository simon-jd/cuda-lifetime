// A caller whose loop runs more iterations than CSA follows on one path:
// every path stops in the loop, so the header function that queues the copy
// is never inlined, and run.py expects it to be analyzed as a fallback root.
#include "helpers.h"
extern int step(int);
float caller(void *device, cudaStream_t stream) {
  int sum = 0;
  for (int i = 0; i < 64; ++i)
    sum += step(i);
  return wrapper(device, stream) + sum;
}

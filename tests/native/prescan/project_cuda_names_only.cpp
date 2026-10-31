// Only project functions with CUDA-like names: no CUDA API call is reachable.
int cudaGraphsManual(int value);
float cuGet(int value) { return static_cast<float>(value); }
int project_only() { return cudaGraphsManual(1) + static_cast<int>(cuGet(2)); }

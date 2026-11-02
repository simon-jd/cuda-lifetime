// nvcc's crt/math_functions.hpp declares max and min for float and double as
// __host__ __device__. Clang's CUDA wrapper headers keep only their __device__
// declarations, so host code that calls them, such as simpleMultiCopy.cu's
// timing summary, does not compile under host-only Clang ("no matching
// function for call to 'max'"). These host overloads do what nvcc's do.
// Force-included (-include) in every mutant and original command alike.
#pragma once
#include <cmath>

static inline __host__ float max(const float a, const float b) { return std::fmax(a, b); }
static inline __host__ double max(const double a, const double b) { return std::fmax(a, b); }
static inline __host__ float min(const float a, const float b) { return std::fmin(a, b); }
static inline __host__ double min(const double a, const double b) { return std::fmin(a, b); }

// Stand-in for the header gpe's build generates from kernels/*.slang
// (cmake/CompileSlang.cmake, gpe_write_kernel_header). It has the same
// declarations for the two kernels CMakeLists.txt compiles (scale, display),
// with one-byte blobs: slangc, which produces the blobs, is not installed,
// and the host code analyzed only uses the table's types and lookup.
#pragma once

#include <cstddef>
#include <string_view>

namespace gpe::kernels {

inline constexpr unsigned char k_scale[] = {0x00};

inline constexpr unsigned char k_display[] = {0x00};

/// One compiled kernel, embedded.
struct Blob {
    std::string_view     name;
    std::string_view     entry;
    const unsigned char* data;
    size_t               size;
};

inline constexpr Blob kAll[] = {
    {"scale", "scaleMain", k_scale, sizeof(k_scale)},
    {"display", "displayMain", k_display, sizeof(k_display)},
};

[[nodiscard]] inline const Blob* find(std::string_view name) {
    for (const Blob& blob : kAll) {
        if (blob.name == name) {
            return &blob;
        }
    }
    return nullptr;
}

}   // namespace gpe::kernels

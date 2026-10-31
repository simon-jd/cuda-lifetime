# Reproduction environment; CUDA driver/GPU access is not needed for static work.
# The 13.2.2 devel image tag is unavailable; install the verified official package.
FROM ubuntu:24.04 AS toolchain
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg xz-utils zstd git ninja-build gcc-13 g++-13 \
    python3 python3-venv python3-dev zlib1g-dev libzstd-dev libedit-dev \
    libblas-dev liblapack-dev libxml2-dev time && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-13-2=13.2.2-1 \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v4.4.4/cmake-4.4.4-linux-x86_64.tar.gz -o /tmp/cmake.tar.gz \
    && echo 'e5bb807f7728cb60cd8b27ebc97a2edb469b68655f21e844a600c3575b76f5bb  /tmp/cmake.tar.gz' | sha256sum -c - \
    && tar -xf /tmp/cmake.tar.gz -C /opt && rm /tmp/cmake.tar.gz
ENV PATH=/opt/cmake-4.4.4-linux-x86_64/bin:/opt/llvm-build/bin:/usr/local/cuda-13.2/bin:$PATH
RUN curl -fsSL https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.3/llvm-project-23.1.3.src.tar.xz -o /tmp/llvm.tar.xz \
    && echo 'c44186a7762ed28954be72e5ff6df9808e0779d4f1bf014ecc4e7e211d31ee34  /tmp/llvm.tar.xz' | sha256sum -c - \
    && tar -xf /tmp/llvm.tar.xz -C /opt && rm /tmp/llvm.tar.xz \
    && cmake -S /opt/llvm-project-23.1.3.src/llvm -B /opt/llvm-build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
       '-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra' -DLLVM_TARGETS_TO_BUILD=X86 \
       -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DCLANG_LINK_CLANG_DYLIB=ON \
       -DLLVM_INCLUDE_BENCHMARKS=OFF \
    && cmake --build /opt/llvm-build --target clang clang-cpp clang-tidy llvm-config -j ${BUILD_JOBS}
FROM toolchain AS project
ARG BUILD_JOBS=8
ARG CUDA_LIFETIME_SOURCE_DIR=.
RUN apt-get update && apt-get install -y --no-install-recommends python3-jsonschema \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work/cuda-lifetime
COPY ${CUDA_LIFETIME_SOURCE_DIR}/ .
RUN python3 -c 'import json; assert json.load(open("/usr/local/cuda-13.2/version.json"))["cuda"]["version"] == "13.2.2"' \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_DIR=/opt/llvm-build/lib/cmake/llvm -DClang_DIR=/opt/llvm-build/lib/cmake/clang \
    && cmake --build build -j ${BUILD_JOBS} \
    && python3 -m unittest discover -s tests -p 'test_*.py' \
    && ctest --test-dir build --output-on-failure
ENTRYPOINT ["python3", "-m", "tool.cli"]

#!/usr/bin/env bash
# Build the standalone C++ benchmark binary using CMake
set -e

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo ""
echo "Binary ready: build/cuda_bench"
echo "Run with: ./build/cuda_bench"

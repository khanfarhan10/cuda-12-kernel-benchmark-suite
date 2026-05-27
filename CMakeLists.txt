cmake_minimum_required(VERSION 3.18)
project(cuda_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)   # Ampere — change for your GPU

find_package(CUDAToolkit 12.3 REQUIRED)

include_directories(src/cuda src/cpp)

set(CUDA_SOURCES
    src/cuda/vector_add.cu
    src/cuda/tiled_matmul.cu
    src/cuda/reduction.cu
    src/cuda/image_preprocess.cu
)

add_executable(cuda_bench
    src/cpp/main.cpp
    ${CUDA_SOURCES}
)

target_link_libraries(cuda_bench CUDA::cudart)

# Enable line info for Nsight profiling
target_compile_options(cuda_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --use_fast_math>
)

# Optional: build shared library for Python ctypes usage
add_library(cuda_kernels_lib SHARED ${CUDA_SOURCES})
target_link_libraries(cuda_kernels_lib CUDA::cudart)

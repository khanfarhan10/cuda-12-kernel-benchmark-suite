#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Ceiling division helper
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Print device info
inline void print_device_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s | SM: %d.%d | Global mem: %.1f GB | Shared mem/block: %zu KB\n",
           prop.name, prop.major, prop.minor,
           (float)prop.totalGlobalMem / (1 << 30),
           prop.sharedMemPerBlock / 1024);
}

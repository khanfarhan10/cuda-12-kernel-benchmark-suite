#pragma once
#include <cuda_runtime.h>

// Tiled matrix multiplication: C = A * B
// A: (M x K), B: (K x N), C: (M x N)
void launch_tiled_matmul(const float* d_A, const float* d_B, float* d_C,
                          int M, int K, int N);

constexpr int TILE_SIZE = 16;

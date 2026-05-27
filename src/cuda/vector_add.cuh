#pragma once
#include <cuda_runtime.h>

// Launch wrapper: element-wise vector addition C = A + B
void launch_vector_add(const float* d_A, const float* d_B, float* d_C,
                       int N, int block_size = 256);

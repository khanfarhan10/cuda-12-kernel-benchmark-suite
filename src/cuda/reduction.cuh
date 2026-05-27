#pragma once
#include <cuda_runtime.h>

// Parallel reduction: sum all N elements of d_in into *d_out.
void launch_reduction(const float* d_in, float* d_out, int N);

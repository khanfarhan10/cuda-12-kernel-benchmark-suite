#include "vector_add.cuh"
#include "cuda_utils.cuh"

// Each thread processes one element. Memory-access pattern is coalesced:
// consecutive threads read/write consecutive addresses in global memory.
__global__ void vector_add_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void launch_vector_add(const float* d_A, const float* d_B, float* d_C,
                       int N, int block_size) {
    int grid_size = ceil_div(N, block_size);
    vector_add_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
}

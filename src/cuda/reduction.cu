#include "reduction.cuh"
#include "cuda_utils.cuh"

// Tree-based parallel reduction using shared memory and warp-level __syncthreads.
// Each block reduces BLOCK_SIZE elements to one partial sum.
// Host launches ceil(N/BLOCK_SIZE) blocks, then either recurses or uses atomicAdd.
//
// Memory coalescing: threads 0..blockDim.x-1 read consecutive global addresses.
// Shared memory bank conflicts avoided by using stride-1 access after each step.
#define BLOCK_SIZE 256

__global__ void reduction_kernel(const float* __restrict__ in, float* __restrict__ out, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < N) ? in[gid] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(out, sdata[0]);
}

void launch_reduction(const float* d_in, float* d_out, int N) {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    int grid = ceil_div(N, BLOCK_SIZE);
    reduction_kernel<<<grid, BLOCK_SIZE>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}

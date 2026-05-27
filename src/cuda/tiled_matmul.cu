#include "tiled_matmul.cuh"
#include "cuda_utils.cuh"

// Tiled matmul using shared memory to reduce global memory bandwidth.
// Each thread block loads a TILE_SIZE x TILE_SIZE tile of A and B into
// shared memory, computes partial dot products, then accumulates.
// Key optimization: shared memory eliminates redundant global reads;
// each element of A and B is loaded only K/TILE_SIZE times instead of N or M.
__global__ void tiled_matmul_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < ceil_div(K, TILE_SIZE); ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void launch_tiled_matmul(const float* d_A, const float* d_B, float* d_C,
                          int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(ceil_div(N, TILE_SIZE), ceil_div(M, TILE_SIZE));
    tiled_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

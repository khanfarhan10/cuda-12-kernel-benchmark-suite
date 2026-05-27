#include <torch/extension.h>
#include <cuda_runtime.h>

// Vector addition kernel exposed as a PyTorch C++/CUDA extension.
// Demonstrates PyTorch CUDA tensor interop and custom kernel integration.
__global__ void vec_add_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

// Tiled matmul kernel (simplified, reuses logic from src/cuda/tiled_matmul.cu)
#define TILE 16
__global__ void tiled_mm_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int K, int N) {
    __shared__ float sA[TILE][TILE], sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        sA[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K)
                                       ? A[row*K + t*TILE+threadIdx.x] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (t*TILE+threadIdx.y < K && col < N)
                                       ? B[(t*TILE+threadIdx.y)*N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = acc;
}

// --- Python-callable wrappers ---

torch::Tensor cuda_vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "Shape mismatch");
    auto c = torch::empty_like(a);
    int N = a.numel();
    int block = 256, grid = (N + block - 1) / block;
    vec_add_kernel<<<grid, block>>>(a.data_ptr<float>(),
                                    b.data_ptr<float>(),
                                    c.data_ptr<float>(), N);
    return c;
}

torch::Tensor cuda_tiled_matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    tiled_mm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                     C.data_ptr<float>(), M, K, N);
    return C;
}

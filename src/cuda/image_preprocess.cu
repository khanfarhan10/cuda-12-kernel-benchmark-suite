#include "image_preprocess.cuh"
#include "cuda_utils.cuh"

// Normalize: out[c][h][w] = (in[h][w][c] / 255.0 - mean[c]) / std[c]
// Converts HWC uint8 input to CHW float output in one kernel pass.
// Using __restrict__ hints the compiler that pointers don't alias,
// enabling better memory access optimization.
__global__ void normalize_kernel(const unsigned char* __restrict__ in,
                                  float* __restrict__ out,
                                  int H, int W, int C,
                                  const float* __restrict__ mean,
                                  const float* __restrict__ std_inv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // col
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row

    if (x >= W || y >= H) return;

    for (int c = 0; c < C; ++c) {
        float val = static_cast<float>(in[(y * W + x) * C + c]) / 255.0f;
        out[c * H * W + y * W + x] = (val - mean[c]) * std_inv[c];
    }
}

void launch_image_normalize(const unsigned char* d_in, float* d_out,
                             int H, int W, int C,
                             const float* d_mean, const float* d_std) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(W, 16), ceil_div(H, 16));
    normalize_kernel<<<grid, block>>>(d_in, d_out, H, W, C, d_mean, d_std);
    CUDA_CHECK(cudaGetLastError());
}

// Swap channels 0 and 2 in a CHW float tensor (BGR<->RGB)
__global__ void bgr2rgb_kernel(float* __restrict__ img, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx0 = 0 * H * W + y * W + x;
    int idx2 = 2 * H * W + y * W + x;
    float tmp = img[idx0];
    img[idx0] = img[idx2];
    img[idx2] = tmp;
}

void launch_bgr2rgb(float* d_img, int H, int W) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(W, 16), ceil_div(H, 16));
    bgr2rgb_kernel<<<grid, block>>>(d_img, H, W);
    CUDA_CHECK(cudaGetLastError());
}

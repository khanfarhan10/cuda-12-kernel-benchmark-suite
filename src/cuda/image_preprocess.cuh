#pragma once
#include <cuda_runtime.h>

// Normalize uint8 image [0,255] to float [0,1] with per-channel mean/std.
// Input: HxWxC uint8, Output: HxWxC float32 (CHW layout optional via flag).
void launch_image_normalize(const unsigned char* d_in, float* d_out,
                             int H, int W, int C,
                             const float* d_mean, const float* d_std);

// BGR -> RGB channel swap in-place on a float CHW tensor.
void launch_bgr2rgb(float* d_img, int H, int W);

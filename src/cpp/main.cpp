#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "../cuda/cuda_utils.cuh"
#include "../cuda/vector_add.cuh"
#include "../cuda/tiled_matmul.cuh"
#include "../cuda/reduction.cuh"
#include "timer.hpp"

static void bench_vector_add(int N, int reps) {
    size_t bytes = N * sizeof(float);
    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    CudaTimer t;
    t.tic();
    for (int i = 0; i < reps; ++i)
        launch_vector_add(d_A, d_B, d_C, N);
    float ms = t.toc() / reps;

    double gb = 3.0 * bytes / 1e9;
    printf("VectorAdd  N=%-10d  avg %.3f ms  BW %.1f GB/s\n", N, ms, gb / (ms * 1e-3));

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; ++i) if (fabsf(h_C[i] - 3.0f) > 1e-4f) { ok = false; break; }
    printf("  Correctness: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

static void bench_reduction(int N, int reps) {
    std::vector<float> h_in(N, 1.0f);
    float *d_in, *d_out, h_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    CudaTimer t;
    t.tic();
    for (int i = 0; i < reps; ++i)
        launch_reduction(d_in, d_out, N);
    float ms = t.toc() / reps;

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = fabsf(h_out - (float)N) < 1.0f;
    printf("Reduction  N=%-10d  avg %.3f ms  result=%.0f  %s\n", N, ms, h_out, ok ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
}

int main() {
    print_device_info();
    printf("\n--- Vector Addition Benchmarks ---\n");
    bench_vector_add(1 << 20, 100);
    bench_vector_add(1 << 24, 50);

    printf("\n--- Reduction Benchmarks ---\n");
    bench_reduction(1 << 20, 100);
    bench_reduction(1 << 24, 50);

    return 0;
}

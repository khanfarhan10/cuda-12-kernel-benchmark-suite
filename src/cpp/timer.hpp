#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <cstdio>

// CUDA event-based timer for accurate GPU kernel timing (excludes CPU overhead).
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    void tic() { cudaEventRecord(start); }
    float toc() {   // returns elapsed ms
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Wall-clock timer for CPU measurements
struct CpuTimer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> t0;
    void tic() { t0 = Clock::now(); }
    double toc_ms() {
        auto t1 = Clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

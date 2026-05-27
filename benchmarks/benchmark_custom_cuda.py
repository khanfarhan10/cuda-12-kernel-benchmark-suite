"""Custom CUDA kernel benchmarks via the PyTorch C++ extension."""
import torch
import time
from tabulate import tabulate

try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../extension"))
    import cuda_kernels
    HAVE_EXT = True
except ImportError:
    HAVE_EXT = False
    print("Warning: cuda_kernels extension not built. Build with: cd extension && pip install -e . --no-build-isolation")


def bench_gpu(fn, reps=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def run():
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    results = []

    if HAVE_EXT:
        for N in [1 << 20, 1 << 24]:
            a = torch.randn(N, device=device, dtype=torch.float32)
            b = torch.randn(N, device=device, dtype=torch.float32)
            ms = bench_gpu(lambda: cuda_kernels.vector_add(a, b))
            bw = 3 * N * 4 / 1e9 / (ms * 1e-3)
            results.append(["Custom VectorAdd", N, f"{ms:.3f} ms", f"{bw:.1f} GB/s"])

        for M, K, N in [(512, 512, 512), (1024, 1024, 1024)]:
            A = torch.randn(M, K, device=device, dtype=torch.float32)
            B = torch.randn(K, N, device=device, dtype=torch.float32)
            ms = bench_gpu(lambda: cuda_kernels.tiled_matmul(A, B), reps=50)
            tflops = 2.0 * M * K * N / 1e12 / (ms * 1e-3)
            results.append(["Custom TiledMatMul", f"{M}x{K}x{N}", f"{ms:.3f} ms", f"{tflops:.2f} TFLOPS"])
    else:
        results.append(["Extension not built", "—", "—", "—"])

    print("\n=== Custom CUDA Kernel Benchmarks ===")
    print(tabulate(results, headers=["Kernel", "Size", "Time", "Throughput"], tablefmt="github"))


if __name__ == "__main__":
    run()

"""PyTorch GPU baseline benchmarks (uses torch CUDA ops, not custom kernels)."""
import torch
import time
from tabulate import tabulate


def bench_gpu(fn, reps=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000  # ms


def run():
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []

    for N in [1 << 20, 1 << 24]:
        a = torch.randn(N, device=device, dtype=torch.float32)
        b = torch.randn(N, device=device, dtype=torch.float32)
        ms = bench_gpu(lambda: a + b)
        bw = 3 * N * 4 / 1e9 / (ms * 1e-3)
        results.append(["VectorAdd", N, f"{ms:.3f} ms", f"{bw:.1f} GB/s"])

    for M, K, N in [(512, 512, 512), (1024, 1024, 1024), (4096, 4096, 4096)]:
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)
        ms = bench_gpu(lambda: torch.mm(A, B), reps=50)
        tflops = 2.0 * M * K * N / 1e12 / (ms * 1e-3)
        results.append(["MatMul", f"{M}x{K}x{N}", f"{ms:.3f} ms", f"{tflops:.1f} TFLOPS"])

    for N in [1 << 20, 1 << 24]:
        a = torch.randn(N, device=device, dtype=torch.float32)
        ms = bench_gpu(lambda: a.sum())
        results.append(["Reduction", N, f"{ms:.3f} ms", "—"])

    print("\n=== PyTorch GPU Baselines ===")
    print(tabulate(results, headers=["Kernel", "Size", "Time", "Throughput"], tablefmt="github"))


if __name__ == "__main__":
    run()

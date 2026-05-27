"""CPU NumPy baseline benchmarks for vector add, matmul, and reduction."""
import time
import numpy as np
from tabulate import tabulate


def bench(fn, reps=50):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000  # ms


def run():
    results = []
    for N in [1 << 20, 1 << 24]:
        a = np.random.randn(N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)
        ms = bench(lambda: a + b)
        results.append(["VectorAdd", N, f"{ms:.3f} ms"])

    for M, K, N in [(512, 512, 512), (1024, 1024, 1024)]:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        ms = bench(lambda: A @ B, reps=10)
        results.append(["MatMul", f"{M}x{K}x{N}", f"{ms:.3f} ms"])

    for N in [1 << 20, 1 << 24]:
        a = np.random.randn(N).astype(np.float32)
        ms = bench(lambda: a.sum())
        results.append(["Reduction", N, f"{ms:.3f} ms"])

    print("\n=== NumPy CPU Baselines ===")
    print(tabulate(results, headers=["Kernel", "Size", "Time"], tablefmt="github"))


if __name__ == "__main__":
    run()

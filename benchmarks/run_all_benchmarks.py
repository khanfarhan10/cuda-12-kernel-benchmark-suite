"""Run all benchmarks sequentially and print a combined results table."""
import subprocess, sys

scripts = [
    "benchmark_numpy.py",
    "benchmark_torch.py",
    "benchmark_custom_cuda.py",
]

for s in scripts:
    print(f"\n{'='*60}\nRunning {s}\n{'='*60}")
    subprocess.run([sys.executable, s], check=False,
                   cwd=__file__.rsplit("/", 1)[0] if "/" in __file__ else ".")

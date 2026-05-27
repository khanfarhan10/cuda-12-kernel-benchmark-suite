# Benchmark Methodology

## Timing approach

| Backend | Timer | Rationale |
|---------|-------|-----------|
| NumPy CPU | `time.perf_counter` | Wall-clock, sufficient for CPU ops |
| PyTorch GPU | `time.perf_counter` + `torch.cuda.synchronize()` before/after | GPU work is async; synchronize ensures accurate wall time |
| Custom CUDA (C++) | `cudaEvent` elapsed time | Measures only kernel time, excludes CPU launch overhead |

## Warmup rounds

All benchmarks run 10–50 warmup iterations before timing to:
- Allow GPU clock boost to stabilize
- Populate L2/texture caches to a steady state
- JIT-compile any lazy PyTorch graphs

## Repetition count

| Size | Reps | Rationale |
|------|------|-----------|
| Small (≤ 1M) | 100 | Low per-call time needs more samples |
| Large (≥ 16M) | 10–50 | Longer per-call time, fewer reps needed |

## Metrics reported

- **Time (ms):** average over repetitions
- **Bandwidth (GB/s):** `(bytes_read + bytes_written) / time` — for memory-bound kernels
- **TFLOPS:** `2 * M * K * N / time` — for matrix multiply

## Environment notes

Results depend heavily on:
- GPU model (A100 ≫ RTX 3090 ≫ Jetson Orin for raw throughput)
- CUDA version (12.3+ recommended)
- Power/thermal limits (Jetson is thermally throttled)
- Host memory bandwidth (pinned vs pageable affects H2D/D2H)

Always report GPU model and CUDA version alongside benchmark results.

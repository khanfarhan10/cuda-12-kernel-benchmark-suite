# Profiling Guide

## Nsight Systems — System-level timeline

Use Nsight Systems to get a bird's-eye view: kernel launches, memory transfers, CPU↔GPU synchronization, stream activity.

```bash
# Profile the C++ benchmark binary
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=nsys_output \
    --force-overwrite=true \
    ./build/cuda_bench

# Open the GUI
nsys-ui nsys_output.nsys-rep
```

**What to look for:**
- Are memory copies (HtoD/DtoH) overlapping with kernel execution? If not, consider CUDA streams + pinned memory.
- Are kernels launching back-to-back without gaps? A gap indicates CPU overhead or synchronization.
- Is `cudaDeviceSynchronize()` called too frequently? Each call stalls the GPU pipeline.

---

## Nsight Compute — Kernel-level microarchitecture profiling

Use Nsight Compute to dive deep into one kernel's performance: memory efficiency, occupancy, pipeline stalls.

```bash
# Profile vector_add_kernel specifically
ncu \
    --target-processes all \
    --kernel-name vector_add_kernel \
    --set full \
    --output ncu_vec_add \
    ./build/cuda_bench

ncu-ui ncu_vec_add.ncu-rep
```

**Key metrics and what they mean:**

| Metric | Good sign | Bad sign |
|--------|-----------|----------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | >60% | <20% (kernel launches too small or too many stalls) |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` (coalescing) | ~100% | <50% (strided/random access pattern) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` (occupancy) | >50% | <20% (too many registers or too much shared mem per block) |
| `l1tex__t_sector_hit_rate.pct` (L1 hit rate) | >80% | <30% |
| `dram__bytes_read.sum / time` | Near device peak BW | Much lower → compute bound |

---

## Compute Sanitizer — Memory and race condition checks

```bash
# Memory error detection
compute-sanitizer --tool memcheck ./build/cuda_bench

# Race condition detection (slower, for shared-memory kernels)
compute-sanitizer --tool racecheck ./build/cuda_bench

# Initialize memory to catch uninitialized reads
compute-sanitizer --tool initcheck ./build/cuda_bench
```

---

## Profiling from Python (PyTorch)

```python
import torch

# Simple CUDA event timing
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
output = my_kernel(inputs)
end.record()
torch.cuda.synchronize()
print(f"Kernel time: {start.elapsed_time(end):.3f} ms")

# PyTorch profiler (richer, integrates with TensorBoard)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    output = my_kernel(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## Roofline model intuition

Ask: is the kernel **memory-bound** or **compute-bound**?

```
Arithmetic Intensity (AI) = FLOPs / Bytes transferred
```

- **Low AI (< ~10 FLOP/byte):** memory-bound. Optimize memory coalescing, use shared memory to reduce DRAM traffic.
- **High AI (> ~10 FLOP/byte):** compute-bound. Optimize instruction throughput, use tensor cores (FP16/BF16), increase occupancy.

For vector add: AI = 1 FLOP / 12 bytes ≈ 0.083 → heavily memory-bound. Bottleneck is always DRAM BW.
For matmul (N=4096): AI = 2N³ / 3N² bytes = 2N/3 ≈ 2730 FLOP/byte → compute-bound on large matrices.

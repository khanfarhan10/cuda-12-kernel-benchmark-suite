# CUDA 12.3+ Interview & Performance Engineering Guide
### Farhan Hai Khan | Prepared for Turing CUDA with C++ and Python Developer Role

---

## 1. CUDA Mental Model

### CPU vs GPU

| | CPU | GPU |
|--|-----|-----|
| Cores | 8–64 powerful cores | Thousands of small cores (A100: 6912 CUDA cores) |
| Design goal | Low latency for serial tasks | High throughput for parallel tasks |
| Cache | Large per-core L1/L2/L3 | Smaller per-SM L1; large shared bandwidth |
| Thread model | OS threads, heavyweight | CUDA threads, ultra-lightweight |
| Context switch | ~microseconds | Near-zero (warp scheduling) |

### Host vs Device

- **Host** = CPU + its RAM (DRAM connected to CPU via PCIe)
- **Device** = GPU + its VRAM (HBM2e on A100, GDDR6 on RTX 30xx)
- `cudaMalloc` allocates **device** memory
- `cudaMemcpy` transfers between host and device
- Kernels execute on the device; host code launches them

### Kernel launch syntax

```cpp
my_kernel<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(args...);
```

- `grid_dim`: number of blocks (dim3 or int)
- `block_dim`: threads per block (dim3 or int, max 1024)
- `shared_mem_bytes`: dynamic shared memory (optional, default 0)
- `stream`: CUDA stream (optional, default 0 = default stream)

### Thread hierarchy

```
Grid
└── Block[0]  Block[1]  ...  Block[gridDim.x-1]
     └── Thread[0]  Thread[1]  ...  Thread[blockDim.x-1]
```

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID (1D)
```

2D grids and blocks:
```cpp
dim3 block(16, 16);                                // 256 threads per block
dim3 grid(ceil_div(W, 16), ceil_div(H, 16));
int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
```

### Warp basics

- A **warp** = 32 consecutive threads that execute the same instruction simultaneously (SIMT = Single Instruction Multiple Threads)
- If threads in a warp take different branches (`if/else`), both paths serialize: **warp divergence** (avoid in hot paths)
- Warp size is always 32 on all NVIDIA hardware to date

---

## 2. CUDA Memory Hierarchy

### Overview (fast → slow, small → large)

| Memory | Location | Scope | Latency | Size |
|--------|----------|-------|---------|------|
| Registers | On-chip | Per-thread | 1 cycle | 255 regs/thread |
| Shared memory | On-chip | Per-block | ~4–8 cycles | 48–164 KB/SM |
| L1 cache | On-chip | Per-SM | ~20–30 cycles | shared with smem |
| L2 cache | On-chip | Device-wide | ~100–200 cycles | 40 MB (A100) |
| Global memory (DRAM/HBM) | Off-chip | Device-wide | ~400–800 cycles | GBs |
| Constant memory | Off-chip | Device-wide (RO) | ~1 cycle (cached) | 64 KB |
| Texture memory | Off-chip | Device-wide (RO) | cached | varies |
| Pinned (host) memory | Host RAM | Host-accessible | slow from GPU | GBs |
| Unified memory | Host+Device | Both | varies | virtual |

### Shared memory

```cpp
__shared__ float tile[16][16];
// Declared inside kernel; allocated per block
// Lifetime = block lifetime
// All threads in the block share it
```

Must `__syncthreads()` after writing before reading (within a block).

### Global memory access patterns

```cpp
// GOOD: coalesced — consecutive threads access consecutive addresses
A[tid]       // tid = blockIdx.x * blockDim.x + threadIdx.x

// BAD: strided — wastes memory transactions
A[tid * stride]  // each thread is far from its neighbor
```

Memory transactions are 128-byte cache lines. If 32 threads access 32 × 4 = 128 consecutive bytes → 1 transaction. If strided → up to 32 transactions.

### Unified memory

```cpp
cudaMallocManaged(&ptr, size);  // accessible from both host and device
// Page migration happens automatically; expensive if pattern is CPU-heavy then GPU-heavy
```

### Pinned (page-locked) memory

```cpp
cudaMallocHost(&h_ptr, size);   // pinned host memory
// Enables async H2D/D2H transfers; can overlap with kernel execution using streams
// Don't over-allocate — pins physical pages and starves the OS
```

---

## 3. Performance Concepts

### Memory coalescing

32 threads in a warp access memory in one instruction. If accesses are contiguous, the GPU merges them into the minimum number of transactions (ideal: 1 per warp for 128-byte access).

**Rule:** thread `t` should access `base + t` (stride 1).

### Occupancy

Occupancy = active warps per SM / max warps per SM.

Higher occupancy helps hide latency (when one warp stalls on memory, another warp runs). But occupancy alone doesn't guarantee performance — a compute-bound kernel with low occupancy can still be fast.

Factors limiting occupancy:
- Register usage per thread (more regs → fewer resident threads)
- Shared memory per block (more smem → fewer concurrent blocks)
- Block size too small (few warps per block → SM underutilized)

Check with `ncu --query-metrics sm__warps_active.avg.pct_of_peak_sustained_active`.

### Arithmetic intensity

```
AI = FLOPs / Bytes_transferred_to/from_DRAM
```

- Vector add: AI ≈ 0.083 FLOP/byte → memory-bound
- Large matmul: AI ≈ 2N/3 FLOP/byte for N×N × N×N → compute-bound

### Bandwidth vs compute bound

- **Memory-bound:** kernel time dominated by DRAM transfers. Fix: improve coalescing, use shared memory, reduce redundant loads.
- **Compute-bound:** kernel time dominated by arithmetic. Fix: use tensor cores (FP16/BF16), increase ILP, use `--use_fast_math`.

### Latency hiding

GPUs hide memory latency by switching to other warps while waiting for data. Enough warps must be "in-flight." If occupancy is too low, the SM idles waiting for memory.

### Shared memory bank conflicts

Shared memory is divided into 32 banks (one per lane). If multiple threads in a warp access the same bank simultaneously (different addresses, same bank) → serialized = **bank conflict** (2-way, 4-way, etc.).

**Avoid:** pad shared memory arrays by 1 element per row.
```cpp
__shared__ float tile[16][16 + 1];  // +1 padding avoids bank conflicts in transposed access
```

---

## 4. Core CUDA Problems

### Vector addition

```cpp
__global__ void vec_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
// Launch: vec_add<<<(N+255)/256, 256>>>(d_A, d_B, d_C, N);
```

Key points: boundary check `if (i < N)`, coalesced access, memory-bound kernel.

### Tiled matrix multiplication

Standard matmul: each element C[i][j] reads a full row of A and column of B from global memory → O(N³) global reads.

Tiled: load TILE×TILE submatrices of A and B into shared memory → each element read only N/TILE times from DRAM → ~TILE× bandwidth reduction.

```cpp
__shared__ float sA[TILE][TILE], sB[TILE][TILE];
// Each thread loads one element of sA and one of sB
// __syncthreads() after loading
// Compute partial dot product
// __syncthreads() before loading next tile
```

### Parallel reduction

Naive: have thread 0 sum everything. O(N) sequential.

Tree reduction: each step halves active threads.
```
Step 1: T[0] += T[128], T[1] += T[129], ...  (stride=128)
Step 2: T[0] += T[64],  T[1] += T[65],  ...  (stride=64)
...
Step 8: T[0] += T[1]                          (stride=1)
```

Result in T[0]. Need `__syncthreads()` between steps for correctness.

For global reduction across blocks: use `atomicAdd` to accumulate partial sums from each block into a single output.

### Prefix sum (scan)

Hillis-Steele (inclusive scan):
```
Pass 1: out[i] = in[i] + in[i-1]   (stride 1)
Pass 2: out[i] = in[i] + in[i-2]   (stride 2)
Pass k: out[i] = in[i] + in[i-2^(k-1)]
```
O(N log N) work but O(log N) depth — optimal for GPU.

### Image preprocessing kernel

Normalize HWC uint8 to CHW float32 in one kernel:
```cpp
out[c * H * W + y * W + x] = (in[(y * W + x) * C + c] / 255.0f - mean[c]) / std[c];
```

Block: 16×16 2D. Grid: ceil(W/16) × ceil(H/16).

---

## 5. CUDA Runtime and NVCC

### Compiling .cu files

```bash
# Basic compilation
nvcc -o my_program my_kernel.cu main.cpp

# With architecture targeting (Ampere = sm_86)
nvcc -arch=sm_86 -o my_program my_kernel.cu main.cpp

# Generate PTX + cubin for multiple architectures
nvcc --generate-code=arch=compute_80,code=sm_80 \
     --generate-code=arch=compute_86,code=sm_86 \
     -o my_program my_kernel.cu main.cpp

# With optimizations
nvcc -O3 --use_fast_math -lineinfo -arch=sm_86 ...
```

### Architecture flags

| GPU Family | Compute Capability | Flag |
|------------|-------------------|------|
| Jetson Orin (Ampere) | 8.7 | sm_87 |
| RTX 30xx (Ampere) | 8.6 | sm_86 |
| A100 (Ampere) | 8.0 | sm_80 |
| V100 (Volta) | 7.0 | sm_70 |
| T4 (Turing) | 7.5 | sm_75 |

### Common NVCC errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot call __device__ from __host__` | Host code calls GPU-only function | Mark as `__host__ __device__` or call from kernel only |
| `nvcc fatal: Unsupported gpu architecture` | Wrong `-arch` flag | Check GPU with `nvidia-smi` and use correct sm_XX |
| `identifier undefined` | Header not included or wrong namespace | Add `#include` |
| `ptxas error: Entry function ... uses too much shared data` | Exceeded shared mem limit | Reduce tile size or use dynamic shared mem |

### Linking CUDA with C++

```cmake
find_package(CUDAToolkit REQUIRED)
target_link_libraries(my_target CUDA::cudart)
```

### Checking CUDA installation

```bash
nvcc --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## 6. PyTorch + CUDA

### Checking availability

```python
import torch
print(torch.cuda.is_available())          # True if GPU is available
print(torch.cuda.device_count())          # Number of GPUs
print(torch.cuda.get_device_name(0))      # GPU name
print(torch.version.cuda)                 # CUDA version PyTorch was compiled against
print(torch.backends.cudnn.version())     # cuDNN version
```

### Writing PyTorch GPU benchmarks

```python
import torch, time

def bench(fn, device="cuda", warmup=20, reps=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000  # ms
```

### Writing PyTorch C++/CUDA extensions

Structure:
```
extension/
├── kernels.cu      # CUDA kernels + host wrappers
├── bindings.cpp    # pybind11 module
└── setup.py        # torch.utils.cpp_extension.CUDAExtension
```

`setup.py`:
```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup
setup(name="my_ext",
      ext_modules=[CUDAExtension("my_ext", ["kernels.cu", "bindings.cpp"])],
      cmdclass={"build_ext": BuildExtension})
```

`bindings.cpp`:
```cpp
#include <torch/extension.h>
torch::Tensor my_op(torch::Tensor x);   // forward declaration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_op", &my_op, "My custom op");
}
```

`kernels.cu`:
```cpp
#include <torch/extension.h>
__global__ void kernel(float* data, int N) { ... }
torch::Tensor my_op(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be on GPU");
    auto out = torch::empty_like(x);
    kernel<<<(x.numel()+255)/256, 256>>>(x.data_ptr<float>(), x.numel());
    return out;
}
```

### When custom CUDA beats PyTorch — and when it doesn't

**Custom beats PyTorch:**
- Fused kernels (e.g., normalize + transpose + copy in one pass)
- Memory layout conversions (HWC → CHW)
- Domain-specific reductions not exposed in PyTorch
- Operations requiring custom synchronization patterns

**PyTorch wins:**
- Standard GEMM/matmul (backed by cuBLAS — heavily optimized)
- Standard convolutions (cuDNN)
- Anything where PyTorch's memory reuse and graph optimization kicks in

---

## 7. TensorRT / ONNX / Jetson / Edge AI

### PyTorch → ONNX → TensorRT pipeline

```python
# Step 1: Export to ONNX
import torch
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}})

# Step 2: Build TensorRT engine (trtexec CLI)
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

# Step 3: Load and run TRT engine (Python, using tensorrt package)
import tensorrt as trt
import pycuda.driver as cuda
```

Or use `torch2trt` / `torch_tensorrt` for tighter integration:
```python
import torch_tensorrt
trt_model = torch_tensorrt.compile(model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.float, torch.half})
```

### FP16 and INT8

| Precision | Memory | Speed | Accuracy |
|-----------|--------|-------|----------|
| FP32 | baseline | baseline | Full |
| FP16 | 0.5× | 2× on Tensor Cores | Minimal loss |
| INT8 | 0.25× | 4× on Tensor Cores | Needs calibration |

**INT8 calibration:** run a calibration dataset through the model, collect activation statistics, determine scale factors per layer. TensorRT does this automatically with `IInt8Calibrator`.

### Jetson Orin constraints

- 12-core ARM Cortex-A78AE CPU + 1792-core Ampere GPU (shared power budget)
- Default TDP: 15–60 W (configurable with `nvpmodel`)
- Shared memory between CPU and GPU (unified memory architecture)
- Use `jetson_clocks` to maximize clocks during benchmarking
- NVDEC accelerates video decoding without using CUDA cores

### GStreamer/NVDEC role in video inference

```
Camera/RTSP → GStreamer pipeline → NVDEC (hardware video decoder)
    → CUDA surface / NV12 buffer → CUDA preprocessing kernel
    → AI inference (TensorRT) → post-processing
```

Without NVDEC: software decoding occupies CPU cores (90%+ CPU).
With NVDEC: video decoding offloaded to dedicated hardware, freeing CPU and CUDA cores.
Result: ~45% CPU, 4 FPS throughput (vs 90% CPU, ~1 FPS without).

GStreamer pipeline example:
```bash
gst-launch-1.0 rtspsrc location=rtsp://... ! rtph264depay ! h264parse \
  ! nvv4l2decoder ! nvvideoconvert ! appsink
```

### CPU/GPU bottlenecks in video analytics

Common bottlenecks:
1. **Video decode (CPU):** fix with NVDEC
2. **H2D memory transfer:** use pinned memory and CUDA streams
3. **Preprocessing (CPU):** move to CUDA kernel
4. **Small batch size:** batch multiple frames; use TRT dynamic shapes
5. **Post-processing (CPU):** NMS, tracking on CPU — move to CUDA where possible

---

## 8. Profiling

### Nsight Systems

- **What it does:** full system timeline — CPU threads, CUDA kernels, memory copies, OS events, NVTX markers
- **When to use:** first pass to understand end-to-end behavior

Key questions to answer:
1. Are H2D/D2H copies overlapping with kernel execution? (look for memcpy + kernel on same row)
2. Are there CPU gaps between kernels? (indicates synchronization or CPU bottleneck)
3. Is the default stream used everywhere? (prevents overlap — use multiple streams)

### Nsight Compute

- **What it does:** deep per-kernel hardware metrics: memory throughput, occupancy, instruction efficiency, pipeline utilization
- **When to use:** after identifying the slow kernel in Nsight Systems

**Metrics to inspect:**

```
Memory throughput:
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum     # Global load bytes
  dram__bytes_read.sum                              # DRAM read bytes
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct  # Coalescing efficiency

Occupancy:
  sm__warps_active.avg.pct_of_peak_sustained_active

Compute utilization:
  sm__throughput.avg.pct_of_peak_sustained_elapsed

Shared memory:
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum  # Bank conflicts
```

### How to explain bottlenecks in an interview

> "I ran Nsight Systems first to get the timeline. The vector_add kernel showed 98% memory bandwidth utilization and 2% compute — classic memory-bound kernel. Nsight Compute confirmed coalescing was near 100%, so the bottleneck was fundamental: the arithmetic intensity of A+B is ~0.08 FLOP/byte, which is well below the GPU's ridge point. No amount of kernel optimization will fix that; the bottleneck is DRAM bandwidth, not code quality."

---

## 9. Debugging

### cudaGetLastError + cudaDeviceSynchronize

```cpp
// Launch kernel
my_kernel<<<grid, block>>>(args);

// Check for launch errors (wrong arguments, too many resources)
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
    fprintf(stderr, "Launch error: %s\n", cudaGetErrorString(err));

// Synchronize and check for execution errors
err = cudaDeviceSynchronize();
if (err != cudaSuccess)
    fprintf(stderr, "Execution error: %s\n", cudaGetErrorString(err));
```

### Compute Sanitizer

```bash
compute-sanitizer --tool memcheck ./cuda_bench      # illegal memory access
compute-sanitizer --tool racecheck ./cuda_bench     # shared memory races
compute-sanitizer --tool synccheck ./cuda_bench     # sync violations
compute-sanitizer --tool initcheck ./cuda_bench     # uninitialized reads
```

### Common bugs and fixes

| Bug | Symptom | Fix |
|-----|---------|-----|
| Missing `if (idx < N)` | Garbage output, out-of-bounds | Always guard thread index |
| Missing `__syncthreads()` | Wrong reduction result, non-deterministic | Add sync after shared mem write |
| Wrong `__syncthreads()` placement inside `if` | Deadlock (some threads skip sync) | Never put `__syncthreads()` inside divergent `if` |
| Host pointer passed to kernel | Segfault / illegal address | Use `cudaMalloc` for device pointers |
| Race on global atomics | Wrong output, non-deterministic | Use `atomicAdd` correctly; ensure no double-counting |
| Shared memory bank conflict | Correct output but slow | Pad arrays: `float sA[16][17]` |

---

## 10. Interview Q&A — 50 CUDA / C++ / Python / PyTorch Questions

### CUDA fundamentals

**Q1. What is a warp and why does its size matter?**
A warp is the fundamental scheduling unit of 32 threads that execute the same instruction in lockstep (SIMT). Its size matters because: (1) memory accesses from a warp are coalesced as a unit; (2) warp divergence (branching within a warp) causes serialization; (3) occupancy is measured in warps.

**Q2. What is the difference between `__global__`, `__device__`, and `__host__` in CUDA?**
- `__global__`: called from host, executed on device (a kernel)
- `__device__`: called from device only, executed on device
- `__host__`: called and executed on host (default for regular C++ functions)
- Can combine `__host__ __device__` to compile for both

**Q3. Why must `__syncthreads()` never appear inside a divergent branch?**
`__syncthreads()` is a barrier for all threads in a block. If some threads enter a branch containing `__syncthreads()` and others don't, the threads that don't enter the branch never hit the barrier → **deadlock**. Every thread in the block must reach the same `__syncthreads()`.

**Q4. What is memory coalescing and how do you achieve it?**
Coalescing merges the global memory accesses of threads in a warp into the minimum number of 128-byte transactions. Achieve it by ensuring thread `t` accesses address `base + t` (stride-1 access). Strided or random access patterns cause multiple transactions.

**Q5. What is occupancy and how do you improve it?**
Occupancy = active warps / max warps per SM. Improve by: (1) reducing register usage per thread (`--maxrregcount` flag); (2) reducing shared memory per block; (3) choosing a block size that's a multiple of warp size (32) and large enough (128–256 minimum).

**Q6. What is the difference between shared memory and L1 cache?**
Both are on-chip and fast. Shared memory is explicitly managed by the programmer (`__shared__`). L1 cache is hardware-managed, caches global memory accesses automatically. On modern GPUs they share the same physical SRAM, configurable in split.

**Q7. What is a CUDA stream?**
A stream is a sequence of CUDA operations (kernels, memcpy) that execute in order within the stream but can overlap with operations in other streams. Multiple streams enable kernel-kernel overlap and kernel-memcpy overlap (requires pinned memory).

```cpp
cudaStream_t s1, s2;
cudaStreamCreate(&s1); cudaStreamCreate(&s2);
kernel_A<<<g, b, 0, s1>>>(d_A);
kernel_B<<<g, b, 0, s2>>>(d_B);  // can overlap with kernel_A
cudaStreamSynchronize(s1);
```

**Q8. What is `atomicAdd` and when do you need it?**
`atomicAdd(&address, val)` adds `val` to the value at `address` atomically — no race condition. Needed when multiple threads write to the same memory location (e.g., final reduction across blocks). Slower than regular writes; avoid in hot inner loops.

**Q9. What is pinned memory and when is it useful?**
Pinned (page-locked) host memory (`cudaMallocHost`) cannot be paged out by the OS, enabling DMA (direct memory access) for faster H2D/D2H transfers. Also enables asynchronous transfers (`cudaMemcpyAsync`) that can overlap with kernel execution. Overuse starves the OS of physical pages.

**Q10. What is the difference between `cudaMemcpy` and `cudaMemcpyAsync`?**
- `cudaMemcpy`: synchronous — blocks CPU until copy completes
- `cudaMemcpyAsync`: asynchronous — returns immediately; requires pinned host memory and a stream to overlap with kernel execution

---

### Memory and performance

**Q11. What causes shared memory bank conflicts and how do you avoid them?**
Shared memory is divided into 32 banks. A bank conflict occurs when multiple threads in a warp access different addresses in the same bank simultaneously → serialized accesses. Fix: pad array rows by 1 element (`float arr[16][17]`) to change the stride, spreading accesses across different banks.

**Q12. What is the roofline model?**
Plots kernel performance (FLOP/s) vs arithmetic intensity (FLOP/byte). The "roof" is the lower of: peak compute (horizontal line) and peak memory bandwidth × AI (diagonal line). Kernels below the roof are either memory-bound (left of ridge point) or compute-bound (right). Tells you which resource to optimize.

**Q13. How does tiled matrix multiplication reduce memory bandwidth?**
Without tiling: each element C[i][j] = dot(row i of A, col j of B) requires reading A[i][0..K-1] and B[0..K-1][j] from global memory = O(MN × 2K) reads total.

With TILE×TILE tiling: load TILE×TILE blocks of A and B into shared memory once per tile, then compute TILE² outputs. Each global element is read exactly N/TILE or M/TILE times → ~TILE× bandwidth reduction.

**Q14. What is occupancy tuning?**
Selecting block size and shared memory usage to maximize the number of active warps per SM. More warps → better latency hiding for memory-bound kernels. Use `cudaOccupancyMaxPotentialBlockSize()` for automatic tuning, or Nsight Compute's "Occupancy" section.

**Q15. When does increasing occupancy NOT help performance?**
When the kernel is compute-bound: SM execution units are already fully utilized. Adding more warps just queues them — no benefit. Also when register pressure is high: halving block size to increase concurrent blocks may increase occupancy but degrade performance due to poor data reuse.

---

### Debugging and tools

**Q16. How do you debug a CUDA kernel that gives wrong results?**
1. Add `cudaDeviceSynchronize()` + `cudaGetLastError()` after the kernel call
2. Use `CUDA_LAUNCH_BLOCKING=1` to serialize launches for cleaner error messages
3. Run `compute-sanitizer --tool memcheck` for illegal memory access
4. Run `compute-sanitizer --tool racecheck` for shared memory races
5. Add printf inside the kernel for small inputs (expensive but sometimes necessary)
6. Reduce to a minimal test case (single block, single warp)

**Q17. What does `cudaGetLastError()` do and why should you always call it?**
It clears and returns the last CUDA error code. Kernels are launched asynchronously, so without calling it (and checking), errors can go silently undetected. The error may actually originate from a previous operation and accumulate until checked.

**Q18. What is Compute Sanitizer and how is it different from Nsight Compute?**
- **Compute Sanitizer**: detects bugs at runtime (illegal memory access, uninitialized reads, race conditions). Like Valgrind for CUDA.
- **Nsight Compute**: performance profiling — measures bandwidth, occupancy, instruction throughput. Not for bug detection.

---

### TensorRT and Inference

**Q19. What is TensorRT and why is it faster than native PyTorch?**
TensorRT is NVIDIA's inference optimizer. It: (1) fuses adjacent layers into single kernels; (2) selects the fastest kernel implementation for your specific GPU; (3) optimizes memory layout between layers; (4) supports FP16/INT8 for higher throughput. A PyTorch model re-run through TRT can be 2–5× faster at inference.

**Q20. What is FP16 and what are its tradeoffs?**
FP16 (half precision) uses 16-bit floats instead of 32-bit. Benefits: 2× memory bandwidth, 2× throughput on FP16 tensor cores, smaller model footprint. Risks: reduced dynamic range (max ~65504 vs ~3.4×10³⁸ for FP32) → loss spikes or NaN in training. For inference, FP16 is usually safe with minimal accuracy loss.

**Q21. What is INT8 quantization and what is calibration?**
INT8 maps float values to 8-bit integers using a scale factor per tensor. Calibration involves running representative input data through the model, recording activation ranges, and computing optimal scale factors to minimize quantization error. TensorRT automates this with `IInt8Calibrator`.

**Q22. What is the PyTorch → ONNX → TensorRT pipeline?**
1. `torch.onnx.export()` converts PyTorch model to ONNX (Open Neural Network Exchange) format
2. `trtexec --onnx=model.onnx --saveEngine=model.trt` builds a TRT engine
3. Load `.trt` engine, create execution context, bind input/output GPU buffers, run `execute_async_v2`
Result: fastest possible inference on the target GPU.

---

### LLM code review (Turing-specific)

**Q23. How do you evaluate the correctness of AI-generated CUDA code?**
1. Check memory bounds: does every thread index guard `if (idx < N)`?
2. Check synchronization: is `__syncthreads()` placed correctly (never in divergent branch)?
3. Check memory access pattern: is global access coalesced?
4. Check for race conditions: any unprotected writes to shared global state?
5. Check launch configuration: grid/block math correct? Block size multiple of 32?
6. Run with Compute Sanitizer to catch runtime bugs
7. Compare output against CPU reference with tolerance (`np.allclose`)

**Q24. What are common mistakes AI models make when generating CUDA kernels?**
- Missing boundary check `if (idx < N)` → out-of-bounds access
- `__syncthreads()` inside `if` → potential deadlock
- Non-coalesced memory access (row-major vs column-major confusion)
- Using `threadIdx.x` where `blockIdx.x * blockDim.x + threadIdx.x` is needed
- Incorrect tile size leading to shared memory overflow
- Missing `cudaDeviceSynchronize()` before reading results back to host

**Q25. How do you write a correctness test for a CUDA kernel?**
```python
import torch, numpy as np

def test_vector_add():
    N = 1 << 20
    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    c_custom = cuda_kernels.vector_add(a, b)   # custom kernel
    c_ref    = a + b                            # PyTorch reference
    assert torch.allclose(c_custom, c_ref, atol=1e-5, rtol=1e-4), \
        f"Max error: {(c_custom - c_ref).abs().max():.2e}"
```

---

### C++ and Python

**Q26. What is `__restrict__` in CUDA C++?**
Tells the compiler that two pointers don't alias (point to the same memory). Enables more aggressive optimizations: the compiler can reorder loads/stores and use vectorized instructions. Always add to kernel parameters when pointers are guaranteed non-aliasing.

**Q27. What is Cython and when would you use it?**
Cython is a superset of Python that compiles to C/C++ and then to a Python extension module. Use when: (1) a Python function is a bottleneck; (2) you want to call C/C++ code from Python; (3) you need type annotations for performance without writing full C++. In the document AI project, Cython accelerated OCR preprocessing pipelines significantly.

**Q28. What is `torch.compile()` in PyTorch 2.x?**
`torch.compile(model)` uses `TorchDynamo` and `TorchInductor` to JIT-compile the model to optimized CUDA code at runtime. Typically 1.5–2× speedup without any code changes. Alternative to TensorRT for Python-first workflows.

**Q29. How do you profile Python GPU code?**
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True, with_stack=True
) as prof:
    output = model(input)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Q30. What is the GIL and does it affect GPU code?**
The Python Global Interpreter Lock prevents true multi-threading for CPU-bound Python code. For GPU code, the GIL is mostly not a bottleneck: CUDA calls are asynchronous, so the GPU runs independently. For CPU preprocessing, use `multiprocessing` or move preprocessing to CUDA.

---

### More CUDA and performance

**Q31. What is warp divergence and how do you minimize it?**
Warp divergence occurs when threads in a warp take different branches (if/else). The GPU serializes both paths: threads in the "false" branch are masked out but still wait. Minimize by: (1) ensuring all threads in a warp follow the same path; (2) restructuring data to group similar-path inputs together; (3) using predication instead of branching for small divergent sections.

**Q32. What is the difference between `cudaMalloc` and `cudaMallocManaged`?**
- `cudaMalloc`: allocates device memory only. Must explicitly copy data with `cudaMemcpy`.
- `cudaMallocManaged`: allocates unified memory accessible from both CPU and GPU. Page migration handled automatically by the CUDA runtime. Simpler but may have performance overhead for streaming access patterns.

**Q33. What are CUDA streams and how do you use multiple streams for overlap?**
Multiple streams can overlap H2D copies, kernel execution, and D2H copies:
```cpp
// Stream A: input chunk 0
cudaMemcpyAsync(d_in0, h_in0, bytes, cudaMemcpyH2D, s0);
kernel<<<grid, block, 0, s0>>>(d_in0, d_out0, N);
cudaMemcpyAsync(h_out0, d_out0, bytes, cudaMemcpyD2H, s0);

// Stream B: input chunk 1 (overlaps with stream A)
cudaMemcpyAsync(d_in1, h_in1, bytes, cudaMemcpyH2D, s1);
kernel<<<grid, block, 0, s1>>>(d_in1, d_out1, N);
```
Requires pinned host memory for async copies.

**Q34. What is the thread block size limit and how do you choose it?**
Max 1024 threads per block. Choose block size to: (1) be a multiple of 32 (full warps); (2) provide enough warps for latency hiding (128–256 often optimal); (3) not exceed register or shared memory limits. For 1D: `block_size = 256`. For 2D images: `dim3(16, 16) = 256`.

**Q35. What is cooperative groups?**
CUDA cooperative groups (CUDA 9+) allow flexible synchronization scopes: warp, block, multi-block, or entire grid. Useful for advanced reduction patterns where you want to synchronize only a subset of threads.
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
auto block = cg::this_thread_block();
block.sync();  // equivalent to __syncthreads()
auto warp = cg::tiled_partition<32>(block);
warp.sync();   // synchronize just the warp
```

**Q36. What is `-lineinfo` and why add it to nvcc?**
`-lineinfo` embeds source file and line number information in the compiled binary. Nsight Compute and Nsight Systems can then map GPU performance counters back to specific source lines, making it much easier to identify exactly which line of CUDA code is the bottleneck.

**Q37. What is `--use_fast_math` in nvcc?**
Enables approximate math intrinsics (e.g., `__sinf` instead of `sinf`). Typically 2–4× faster for transcendentals with ~1 ULP error. Safe for most ML/inference workloads; not safe for applications requiring IEEE-754 precision.

**Q38. Explain `dim3` in CUDA.**
`dim3` is a 3-component struct `{x, y, z}` for specifying grid/block dimensions. Unspecified components default to 1.
```cpp
dim3 block(16, 16);      // 16×16×1 = 256 threads per block
dim3 grid(W/16, H/16);   // ceil(W/16) × ceil(H/16) blocks
kernel<<<grid, block>>>(...);
```

**Q39. What are CUDA events and how are they used for timing?**
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
my_kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
// ms = time between start and stop events (GPU-side, accurate)
```
More accurate than host timing because it measures GPU time directly.

**Q40. What is global memory coalescing at the hardware level?**
When a warp issues a global memory access, the memory controller checks the 32 addresses. If they fall within the same 128-byte cache line sector(s), they are merged into 1–4 transactions. If they're scattered across different cache lines, each address may require a separate transaction (up to 32×). For float32, ideal coalescing = 32 × 4 = 128 bytes = 1 transaction per warp.

---

### Advanced and edge AI

**Q41. What are tensor cores and when do they activate?**
Tensor cores are specialized units in Volta+ GPUs that compute 4×4 or 16×16 matrix multiply-accumulate (MMA) in one instruction. Activated for `torch.mm`, `torch.conv2d`, and TensorRT layers when inputs are FP16, BF16, or INT8 and dimensions are multiples of 8/16. Provide 8–16× higher throughput than regular CUDA cores for matmul.

**Q42. How does NVDEC work?**
NVDEC is a dedicated fixed-function hardware decoder on NVIDIA GPUs. It decodes H.264/H.265/AV1 video streams directly into GPU memory (NV12 format) without using CUDA cores or CPU. In GStreamer, `nvv4l2decoder` (Jetson) or `nvdec` (desktop) enables this.

**Q43. What is NV12 format?**
NV12 is a planar YUV 4:2:0 format: full-resolution Y (luma) plane followed by interleaved UV (chroma) plane at half resolution. NVDEC outputs NV12. Must convert to RGB/BGR for typical CV models using `nvvideoconvert` (GStreamer) or a custom CUDA kernel.

**Q44. How do you optimize a TensorRT model for Jetson Orin?**
1. Use FP16: `trtexec --onnx=model.onnx --saveEngine=model.trt --fp16`
2. Set explicit batch size matching your inference batch
3. Use dynamic shapes sparingly (adds overhead)
4. Use DLA (Deep Learning Accelerator) for supported layers to free GPU for other tasks
5. Use NVDEC for video input rather than CPU decode
6. Profile with `trtexec --loadEngine=model.trt --iterations=100 --duration=30`

**Q45. What is the difference between synchronous and asynchronous execution in CUDA?**
CUDA API calls are generally asynchronous (return to CPU immediately while GPU works). Exceptions: `cudaDeviceSynchronize()`, `cudaMemcpy` (synchronous variant), and kernel calls in blocking mode. Async execution enables overlap but also means bugs can appear asynchronously — always check errors after sync.

---

### Python/ML ecosystem

**Q46. What is `torch.no_grad()` and when is it important for inference?**
Disables gradient tracking. For inference: (1) reduces memory by not storing activations for backprop; (2) faster because gradient computation is skipped. Always use during inference:
```python
with torch.no_grad():
    output = model(input)
```

**Q47. How do you measure GPU memory usage in PyTorch?**
```python
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.max_memory_allocated() / 1e9, "GB peak")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved by caching allocator")
torch.cuda.reset_peak_memory_stats()
```

**Q48. What is `torch.cuda.empty_cache()` and when would you call it?**
Releases unused cached GPU memory back to the OS. The PyTorch caching allocator keeps memory in a pool for fast reuse; `empty_cache()` forces release. Call between unrelated inference batches if you need the memory for another process. Does NOT free memory that's still referenced by live tensors.

**Q49. What is the difference between `model.half()` and `torch.autocast`?**
- `model.half()`: permanently converts all model parameters to FP16. Simple but less numerically stable.
- `torch.autocast`: automatically applies mixed precision where beneficial, keeping FP32 for numerically sensitive ops (e.g., softmax, log). Better accuracy tradeoff.

```python
with torch.autocast(device_type="cuda"):
    output = model(input)
```

**Q50. What is cuBLAS and when does PyTorch use it?**
cuBLAS is NVIDIA's GPU-accelerated BLAS (Basic Linear Algebra Subprograms) library. PyTorch uses it under the hood for `torch.mm`, `torch.bmm`, `F.linear`, etc. It automatically selects the fastest algorithm for your GPU and uses tensor cores when applicable. This is why custom naive matmul kernels are slower than `torch.mm` — cuBLAS is a highly engineered library.

---

## 11. Delivery Review Preparation

### How to explain the GitHub proof project

> "I built a CUDA 12.3+ benchmark suite that implements custom kernels for four operations: vector addition, tiled matrix multiplication, parallel reduction, and image preprocessing. The vector add kernel demonstrates memory coalescing — stride-1 access pattern for maximum DRAM bandwidth. The tiled matmul uses 16×16 shared memory tiles to reduce global memory traffic by ~16×. The reduction uses a tree-based approach with `atomicAdd` for cross-block aggregation. The image preprocessing kernel converts HWC uint8 to CHW float32 in one pass, which is exactly what you need before TensorRT inference.
>
> I also built a PyTorch C++/CUDA extension using pybind11 so the kernels are callable from Python, and I added correctness tests comparing against NumPy/PyTorch references with float tolerances. For profiling, I have Nsight Systems and Nsight Compute scripts with notes on what metrics to inspect."

### What to demo

1. Run `./build/cuda_bench` and show output (vector add bandwidth, reduction correctness)
2. Show `python benchmarks/benchmark_torch.py` comparing PyTorch GPU vs custom
3. Show `pytest tests/ -v` — all passing
4. Open Nsight Compute report and point to: coalescing efficiency, occupancy, memory throughput

### Benchmark explanation script

> "For vector add at N=4M floats, our custom kernel hits ~420 GB/s effective bandwidth on an RTX 3090. The device's peak memory bandwidth is ~936 GB/s. We're at ~45% efficiency — expected, because vector add is a pure bandwidth kernel and we're bandwidth-bound, not compute-bound. The PyTorch baseline shows similar numbers because PyTorch also uses a coalesced kernel internally. The point here isn't to beat PyTorch — it's to demonstrate I understand memory access patterns, can write correct CUDA C++, and know how to measure performance properly."

### Common follow-up questions

- **"Why is your custom matmul slower than PyTorch?"** — PyTorch uses cuBLAS, which is NVIDIA's hand-tuned library using tensor cores. My tiled kernel uses FP32 CUDA cores with 16×16 tiles. For a fairer comparison, I'd use FP16 and larger tiles, but even then cuBLAS is the gold standard.
- **"Can you add FP16 support?"** — Yes, change `float` to `__half`, use `__hmul`/`__hadd` intrinsics, and add `--generate-code=arch=compute_86,code=sm_86` with FP16 enabled.
- **"What would you change to support batch processing?"** — Add a batch dimension, launch with an extra grid dimension for batch index, or use CUDA streams for pipeline parallelism across batches.

---

## 12. Seven-Day Crash Plan

### Day 1 — CUDA fundamentals + vector add

**Morning (3h):**
- Read sections 1–3 of this guide (mental model, memory hierarchy, performance concepts)
- Write vector_add.cu from scratch without looking at the solution
- Verify output correctness

**Afternoon (3h):**
- Set up the GitHub project locally (`bash scripts/setup_env.sh`)
- Build and run `./build/cuda_bench`
- Run `pytest tests/test_vector_add.py`

**Evening (1h):**
- Answer Q1–Q10 aloud without notes
- Write down anything you got wrong

---

### Day 2 — Tiled matmul + shared memory

**Morning (3h):**
- Study shared memory, bank conflicts, tiling from sections 2–3
- Implement `tiled_matmul.cu` from scratch (16×16 tiles)
- Test correctness vs `numpy.dot`

**Afternoon (3h):**
- Run Nsight Compute on the matmul kernel
- Find the bandwidth and occupancy numbers
- Explain why custom matmul is slower than cuBLAS

**Evening (1h):**
- Answer Q11–Q20 aloud
- Document what metrics you'd check in an interview

---

### Day 3 — Reduction + parallel patterns

**Morning (3h):**
- Implement tree reduction from scratch
- Add correctness test for non-power-of-two sizes
- Understand prefix sum (section 4)

**Afternoon (3h):**
- Implement image preprocessing kernel (normalize + BGR↔RGB)
- Verify against torchvision transforms
- Profile with Nsight Systems

**Evening (1h):**
- Answer Q21–Q30 aloud

---

### Day 4 — PyTorch extension + benchmarks

**Morning (3h):**
- Build and test the PyTorch C++/CUDA extension (`bash scripts/build_extension.sh`)
- Run `benchmark_custom_cuda.py`
- Compare timing with `benchmark_torch.py` and `benchmark_numpy.py`

**Afternoon (3h):**
- Read sections 6 (PyTorch + CUDA) and 8 (Profiling)
- Run Nsight Compute on vector_add_kernel
- Write down the 5 key metrics and their values for your GPU

**Evening (1h):**
- Answer Q31–Q40 aloud

---

### Day 5 — TensorRT, ONNX, Jetson, LLM code review

**Morning (3h):**
- Read section 7 (TensorRT / ONNX / Jetson / Edge AI)
- Export a small PyTorch model to ONNX
- Convert to TensorRT with trtexec if available

**Afternoon (3h):**
- Practice reviewing a sample AI-generated CUDA kernel for common bugs (Q23–Q25)
- Reread section 9 (Debugging)
- Run Compute Sanitizer on the benchmark binary

**Evening (1h):**
- Answer Q41–Q50 aloud

---

### Day 6 — Mock interview simulation

**Morning (2h):**
- Whiteboard / explain tiled matmul in under 5 minutes
- Explain the GPU memory hierarchy without notes
- Explain the roofline model and classify your benchmark kernels

**Afternoon (3h):**
- Review your GitHub project code as if you're presenting it
- Prepare the delivery review demo (section 11)
- Write down 3 things you'd improve about each kernel

**Evening (2h):**
- Full mock technical Q&A (pick 20 random questions from section 10 and answer cold)
- Time yourself — aim for 1–2 minutes per question

---

### Day 7 — Polish + submission

**Morning (2h):**
- Push the GitHub project to `github.com/khanfarhan10/cuda-12-kernel-benchmark-suite`
- Update resume with GitHub link and final CUDA project bullet
- Send updated resume to Bhargavi

**Afternoon (2h):**
- Final review of interview Q&A weak spots
- Re-read the delivery review preparation section
- Prepare environment (IDE, terminal, Nsight tools ready for demo)

**Evening (1h):**
- Rest. You're ready.

---

## Export to PDF

```bash
# Using pandoc (recommended)
pandoc docs/CUDA_12_3_Learning_Guide.md \
  -o Farhan_CUDA_12_3_Learning_Guide.pdf \
  --pdf-engine=pdflatex \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --toc

# Alternative: weasyprint (Python-based, better HTML rendering)
pip install weasyprint markdown
python -c "
import markdown, weasyprint
with open('docs/CUDA_12_3_Learning_Guide.md') as f:
    html = markdown.markdown(f.read(), extensions=['tables', 'fenced_code'])
weasyprint.HTML(string='<html><body>' + html + '</body></html>').write_pdf('Farhan_CUDA_12_3_Learning_Guide.pdf')
"
```

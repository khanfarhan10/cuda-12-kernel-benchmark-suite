# CUDA 12.3+ Kernel Benchmark Suite

[![CUDA 12.3+](https://img.shields.io/badge/CUDA-12.3%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org)

Custom CUDA C++ kernels for **vector addition**, **tiled matrix multiplication**, **parallel reduction**, and **image preprocessing**, benchmarked against NumPy CPU and PyTorch GPU baselines, with profiling support via Nsight Systems and Nsight Compute.

---

## Why this project exists

This project demonstrates hands-on CUDA 12.3+ GPU programming skills:
- Custom CUDA kernels with correct memory access patterns, shared memory usage, synchronization, and kernel launch configuration
- Performance benchmarking methodology (CPU vs GPU vs custom kernel)
- PyTorch C++/CUDA extension integration
- Profiling workflow for identifying memory-bound vs compute-bound bottlenecks

---

## CUDA concepts demonstrated

| Concept | Where |
|---------|-------|
| Thread/block/grid model | All `.cu` files |
| Memory coalescing | `vector_add.cu` — stride-1 global access |
| Shared memory tiling | `tiled_matmul.cu` — 16×16 tiles reduce DRAM traffic |
| Tree reduction + atomics | `reduction.cu` |
| Synchronization (`__syncthreads`) | `tiled_matmul.cu`, `reduction.cu` |
| `__restrict__` hint | `vector_add.cu`, `image_preprocess.cu` |
| CUDA event timing | `src/cpp/timer.hpp` |
| PyTorch C++/CUDA extension | `extension/kernels.cu`, `extension/bindings.cpp` |
| Nsight Systems profiling | `scripts/profile_nsight_systems.sh` |
| Nsight Compute profiling | `scripts/profile_nsight_compute.sh` |

---

## Repository structure

```
cuda-12-kernel-benchmark-suite/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── setup.py
├── CMakeLists.txt
├── src/
│   ├── cpp/
│   │   ├── main.cpp          # C++ benchmark runner
│   │   └── timer.hpp         # CUDA event + CPU timers
│   └── cuda/
│       ├── cuda_utils.cuh    # CUDA_CHECK macro, device info
│       ├── vector_add.cu/h   # Element-wise addition kernel
│       ├── tiled_matmul.cu/h # Shared-memory tiled matmul
│       ├── reduction.cu/h    # Parallel tree reduction
│       └── image_preprocess.cu/h  # Normalize + BGR↔RGB
├── extension/
│   ├── kernels.cu            # PyTorch-callable CUDA kernels
│   ├── bindings.cpp          # pybind11 module definition
│   ├── setup.py              # Extension build script
│   └── __init__.py
├── benchmarks/
│   ├── benchmark_numpy.py    # NumPy CPU baselines
│   ├── benchmark_torch.py    # PyTorch GPU baselines
│   ├── benchmark_custom_cuda.py  # Custom kernel benchmarks
│   └── run_all_benchmarks.py
├── tests/
│   ├── test_vector_add.py
│   ├── test_matmul.py
│   ├── test_reduction.py
│   └── test_image_preprocess.py
├── scripts/
│   ├── setup_env.sh
│   ├── build_cpp.sh
│   ├── build_extension.sh
│   ├── run_tests.sh
│   ├── profile_nsight_systems.sh
│   └── profile_nsight_compute.sh
└── docs/
    ├── CUDA_12_3_Learning_Guide.md    # Comprehensive CUDA study guide
    ├── profiling_guide.md
    ├── benchmark_methodology.md
    └── troubleshooting.md
```

---

## Requirements

- CUDA Toolkit 12.3+ with NVCC
- NVIDIA GPU (Ampere or newer recommended; Jetson Orin supported)
- Python 3.9+
- PyTorch 2.1+ (CUDA build)
- CMake 3.18+
- GCC 9+ / Clang 10+

---

## CUDA 12.3+ setup

### Check existing installation
```bash
nvcc --version        # should show release 12.3 or higher
nvidia-smi            # shows driver version and GPU
```

### Install CUDA Toolkit 12.3
```bash
# Ubuntu 22.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-3

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Install PyTorch with CUDA 12.3
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu123
```

---

## Build instructions

### Option A: C++ benchmark binary (CMake)
```bash
bash scripts/build_cpp.sh
# Produces: build/cuda_bench
./build/cuda_bench
```

### Option B: PyTorch C++/CUDA extension
```bash
bash scripts/build_extension.sh
# Or manually:
cd extension && pip install -e . --no-build-isolation
```

---

## Run tests
```bash
pip install -r requirements.txt
bash scripts/run_tests.sh
# Or: pytest tests/ -v
```

---

## Run benchmarks
```bash
# All benchmarks
python benchmarks/run_all_benchmarks.py

# Individual
python benchmarks/benchmark_numpy.py
python benchmarks/benchmark_torch.py
python benchmarks/benchmark_custom_cuda.py
```

---

## Profiling

### Nsight Systems (system-level timeline)
```bash
bash scripts/profile_nsight_systems.sh ./build/cuda_bench
nsys-ui nsys_profile.nsys-rep
```

### Nsight Compute (kernel-level metrics)
```bash
bash scripts/profile_nsight_compute.sh ./build/cuda_bench vector_add_kernel
ncu-ui ncu_profile.ncu-rep
```

---

## Expected benchmark results

Results will vary by GPU. Representative numbers on an RTX 3090:

| Kernel | Size | NumPy CPU | PyTorch GPU | Custom CUDA |
|--------|------|-----------|-------------|-------------|
| Vector Add | 4M floats | ~8 ms | ~0.15 ms | ~0.12 ms |
| Vector Add | 64M floats | ~130 ms | ~1.8 ms | ~1.5 ms |
| Matrix Mul | 1024×1024 | ~40 ms | ~0.4 ms | ~1.2 ms* |
| Reduction | 4M floats | ~4 ms | ~0.08 ms | ~0.10 ms |

\* Custom tiled kernel uses 16×16 tiles; PyTorch uses cuBLAS (highly optimized). The goal here is learning, not beating cuBLAS.

---

## How to interpret results

- **Vector add is memory-bound:** bandwidth ≈ 3 × N × 4 bytes / time. Compare to GPU peak BW.
- **Matmul is compute-bound at large sizes:** compare TFLOPS to GPU peak.
- **Custom kernel vs PyTorch:** custom kernels won't beat cuBLAS/cuDNN for standard ops. Value is in demonstrating the programming model and optimization knowledge.

---

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for common CUDA/NVCC/PyTorch setup issues.

---

## Resume bullet

> Built a CUDA 12.3+ benchmark suite implementing custom CUDA C++ kernels for vector operations, tiled matrix multiplication, parallel reduction, and image preprocessing; benchmarked against NumPy/PyTorch baselines; built PyTorch C++/CUDA extension bindings; profiled memory access, occupancy, latency, and throughput using Nsight Systems and Nsight Compute.

---

## Author

**Farhan Hai Khan** — CUDA / GPU Performance Engineer | C++ | Python | PyTorch | TensorRT | Edge AI  
GitHub: [khanfarhan10](https://github.com/khanfarhan10) | Email: farhan.ai.engineer@gmail.com

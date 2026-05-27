# Troubleshooting CUDA / NVCC / PyTorch CUDA

## 1. `nvcc: command not found`

NVCC is not on PATH. Fix:
```bash
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```
Add to `~/.bashrc` to persist.

## 2. `CUDA driver version is insufficient`

NVIDIA driver is too old for the CUDA Toolkit version. Run `nvidia-smi` to see driver version.
- CUDA 12.3 requires driver ≥ 525.85.12
- Install/update driver: `sudo apt install nvidia-driver-525`

## 3. `torch.cuda.is_available()` returns False

Common causes:
- PyTorch CPU-only wheel installed. Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu123`
- CUDA_VISIBLE_DEVICES='' set in environment. Unset it.
- Driver mismatch — run `python -c "import torch; print(torch.version.cuda)"` vs `nvcc --version`

## 4. Extension build fails: `fatal error: torch/extension.h: No such file`

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
# Add that path to CMAKE_PREFIX_PATH
```

Or simply build from the extension directory:
```bash
cd extension && pip install -e . --no-build-isolation
```

## 5. `CUDA error: device-side assert triggered`

Turn on full error info:
```bash
CUDA_LAUNCH_BLOCKING=1 python your_script.py
```
Then read the traceback — common cause is an out-of-bounds index or invalid tensor shape.

## 6. `cudaErrorIllegalAddress` / segfault in kernel

- Pointer passed to kernel is host memory, not device memory.
- Accessing past array bounds — check grid/block math.
- Use Compute Sanitizer: `compute-sanitizer --tool memcheck ./cuda_bench`

## 7. Kernel produces wrong results silently

- Forgot `__syncthreads()` between shared-memory write and read.
- Race condition on global atomics.
- Thread boundary check missing (`if (idx < N)`).
- Run with `compute-sanitizer --tool racecheck ./cuda_bench`.

## 8. `CMake Error: Could not find CUDA`

```bash
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc
```

## 9. Jetson / embedded: `insufficient shared memory`

Jetson Orin has 48 KB shared memory per SM (same as desktop Ampere). Reduce `TILE_SIZE` from 32 to 16 if needed.

## 10. Nsight Compute requires sudo

```bash
sudo ncu --target-processes all ./cuda_bench
```
Or add the user to the `nvidia` group and set `kernel.perf_event_paranoid = 0`.

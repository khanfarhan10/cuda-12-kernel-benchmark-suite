"""
cuda_kernels — PyTorch C++/CUDA extension.

Build with:
    cd extension && pip install -e . --no-build-isolation

Then use:
    import cuda_kernels
    c = cuda_kernels.vector_add(a, b)     # a, b: torch.float32 CUDA tensors
    C = cuda_kernels.tiled_matmul(A, B)   # A: (M,K), B: (K,N) CUDA tensors
"""
try:
    from cuda_kernels import vector_add, tiled_matmul  # noqa: F401
except ImportError:
    raise ImportError(
        "cuda_kernels extension not built. Run: cd extension && pip install -e . --no-build-isolation"
    )

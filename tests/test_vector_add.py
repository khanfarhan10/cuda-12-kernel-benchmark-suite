"""Correctness tests for custom CUDA vector addition."""
import pytest
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../extension"))

try:
    import cuda_kernels
    HAVE_EXT = True
except ImportError:
    HAVE_EXT = False


@pytest.mark.skipif(not HAVE_EXT, reason="cuda_kernels extension not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVectorAdd:
    def test_basic(self):
        a = torch.ones(1024, device="cuda")
        b = torch.ones(1024, device="cuda") * 2
        c = cuda_kernels.vector_add(a, b)
        assert torch.allclose(c, torch.full((1024,), 3.0, device="cuda"))

    def test_large(self):
        N = 1 << 22
        a = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        c_custom = cuda_kernels.vector_add(a, b)
        c_ref    = a + b
        assert torch.allclose(c_custom, c_ref, atol=1e-5), \
            f"Max error: {(c_custom - c_ref).abs().max()}"

    def test_zeros(self):
        a = torch.zeros(256, device="cuda")
        b = torch.zeros(256, device="cuda")
        c = cuda_kernels.vector_add(a, b)
        assert torch.all(c == 0)

    def test_non_multiple_of_block(self):
        N = 777
        a = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        assert torch.allclose(cuda_kernels.vector_add(a, b), a + b, atol=1e-5)

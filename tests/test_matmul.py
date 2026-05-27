"""Correctness tests for custom CUDA tiled matrix multiplication."""
import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../extension"))

try:
    import cuda_kernels
    HAVE_EXT = True
except ImportError:
    HAVE_EXT = False


@pytest.mark.skipif(not HAVE_EXT, reason="cuda_kernels extension not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTiledMatmul:
    def test_square(self):
        A = torch.randn(64, 64, device="cuda")
        B = torch.randn(64, 64, device="cuda")
        C_custom = cuda_kernels.tiled_matmul(A, B)
        C_ref    = torch.mm(A, B)
        assert torch.allclose(C_custom, C_ref, atol=1e-4), \
            f"Max error: {(C_custom - C_ref).abs().max()}"

    def test_non_square(self):
        A = torch.randn(128, 64, device="cuda")
        B = torch.randn(64, 256, device="cuda")
        C_custom = cuda_kernels.tiled_matmul(A, B)
        C_ref    = torch.mm(A, B)
        assert torch.allclose(C_custom, C_ref, atol=1e-4)

    def test_non_tile_multiple(self):
        # Sizes not multiples of TILE_SIZE=16
        A = torch.randn(33, 50, device="cuda")
        B = torch.randn(50, 47, device="cuda")
        C_custom = cuda_kernels.tiled_matmul(A, B)
        C_ref    = torch.mm(A, B)
        assert torch.allclose(C_custom, C_ref, atol=1e-4)

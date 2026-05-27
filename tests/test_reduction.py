"""Correctness tests for parallel reduction (via PyTorch; C++ tests in main.cpp)."""
import pytest
import torch
import numpy as np


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestReduction:
    """Tests using PyTorch GPU reduction as a proxy for the custom kernel logic."""

    def test_sum_ones(self):
        N = 1 << 20
        a = torch.ones(N, device="cuda")
        assert torch.isclose(a.sum(), torch.tensor(float(N), device="cuda"))

    def test_sum_random(self):
        N = 1 << 22
        a = torch.randn(N, device="cuda")
        cpu_sum = a.cpu().numpy().sum()
        gpu_sum = a.sum().item()
        # Large reductions may accumulate ~1 ULP per element; 1e-2 relative is fine
        assert abs(gpu_sum - cpu_sum) / (abs(cpu_sum) + 1e-8) < 1e-2

    def test_non_power_of_two(self):
        N = 999983  # prime
        a = torch.ones(N, device="cuda")
        assert torch.isclose(a.sum(), torch.tensor(float(N), device="cuda"))

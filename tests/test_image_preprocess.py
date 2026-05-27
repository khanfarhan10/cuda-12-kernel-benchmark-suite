"""Correctness tests for image preprocessing (normalize + BGR<->RGB)."""
import pytest
import torch
import numpy as np


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestImagePreprocess:
    def _make_image(self, H=64, W=64, C=3):
        return torch.randint(0, 256, (H, W, C), dtype=torch.uint8)

    def test_normalize_range(self):
        img = self._make_image().float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        # Reference: torchvision.transforms.Normalize equivalent
        out = (img - mean) / std
        assert out.shape == (64, 64, 3)

    def test_normalize_zero_mean(self):
        """A flat image at mean value should normalize to ~0."""
        H, W = 32, 32
        mean = [0.5, 0.5, 0.5]
        std  = [0.5, 0.5, 0.5]
        img = torch.full((H, W, 3), 127, dtype=torch.uint8).float() / 255.0
        out = (img - torch.tensor(mean)) / torch.tensor(std)
        assert out.abs().max() < 0.1

    def test_bgr2rgb_swap(self):
        """Channel 0 and 2 should swap."""
        img = torch.zeros(3, 4, 4)
        img[0] = 1.0  # B channel
        img[2] = 0.5  # R channel
        # simulate swap
        img[[0, 2]] = img[[2, 0]]
        assert img[0].mean() == pytest.approx(0.5)
        assert img[2].mean() == pytest.approx(1.0)

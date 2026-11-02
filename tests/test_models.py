"""Unit tests for U-Net and DeepLabV3+ segmentation architectures."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from surfaceguard.models.unet import UNet
from surfaceguard.models.deeplabv3 import DeepLabV3Plus


class TestUNet:
    def test_unet_output_shapes(self):
        model = UNet(in_channels=3, num_classes=1, base_features=16)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        assert "logits" in out
        assert "mask" in out
        assert out["logits"].shape == (2, 1, 128, 128)
        assert out["mask"].shape == (2, 1, 128, 128)
        assert (out["mask"] >= 0.0).all() and (out["mask"] <= 1.0).all()

    def test_cpu_device_compatibility(self):
        model = UNet(in_channels=3, num_classes=1, base_features=16).to(torch.device("cpu"))
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out["mask"].device.type == "cpu"


class TestDeepLabV3Plus:
    def test_deeplabv3_output_shapes(self):
        model = DeepLabV3Plus(in_channels=3, num_classes=1, base_channels=16)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        assert "logits" in out
        assert "mask" in out
        assert out["logits"].shape == (2, 1, 128, 128)
        assert out["mask"].shape == (2, 1, 128, 128)
        assert (out["mask"] >= 0.0).all() and (out["mask"] <= 1.0).all()

"""Unit tests for SurfaceSegmentor inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from surfaceguard.inference.segmentor import SurfaceSegmentor
from surfaceguard.models.unet import UNet


class TestSurfaceSegmentor:
    def test_predict_on_numpy_image(self):
        model = UNet(in_channels=3, num_classes=1, base_features=16)
        segmentor = SurfaceSegmentor(model=model, image_size=(64, 64), threshold=0.5)

        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        res = segmentor.predict(dummy_img)

        assert res.defect_probability_map.shape == (100, 100)
        assert res.binary_mask.shape == (100, 100)
        assert res.latency_ms > 0.0

    def test_predict_on_pil_image(self):
        model = UNet(in_channels=3, num_classes=1, base_features=16)
        segmentor = SurfaceSegmentor(model=model, image_size=(64, 64))

        pil_img = Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8))
        res = segmentor.predict(pil_img)

        assert res.defect_probability_map.shape == (80, 80)

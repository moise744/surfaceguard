"""Unit tests for paired segmentation transforms and dataset loading."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from surfaceguard.data.dataset import SurfaceDefectDataset
from surfaceguard.data.transforms import PairedSegmentationTransforms


class TestSegmentationTransforms:
    def test_paired_transform_shapes(self):
        tfm = PairedSegmentationTransforms(image_size=(128, 128), is_training=True)
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        mask = Image.fromarray(np.random.randint(0, 255, (200, 200), dtype=np.uint8))

        img_t, mask_t = tfm(img, mask)

        assert img_t.shape == (3, 128, 128)
        assert mask_t.shape == (1, 128, 128)
        assert mask_t.dtype == torch.float32
        # Binary values in mask
        unique_vals = torch.unique(mask_t).tolist()
        assert set(unique_vals).issubset({0.0, 1.0})


class TestDatasetLoader:
    def test_empty_directory_handling(self, tmp_path):
        ds = SurfaceDefectDataset(root=tmp_path, category="metal", split="test")
        assert len(ds) == 0

    def test_mock_defect_directory(self, tmp_path):
        cat_dir = tmp_path / "bottle"
        split_dir = cat_dir / "test" / "broken"
        gt_dir = cat_dir / "ground_truth" / "broken"
        split_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)

        dummy_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        dummy_mask = Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 255)

        dummy_img.save(split_dir / "001.png")
        dummy_mask.save(gt_dir / "001_mask.png")

        ds = SurfaceDefectDataset(root=tmp_path, category="bottle", split="test")
        assert len(ds) == 1
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["is_defective"] is True

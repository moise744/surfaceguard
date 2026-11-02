"""
Inference engine for industrial surface defect segmentation.

Extracts pixel-level defect masks, separates distinct defect blobs,
and extracts geometric attributes (area, bounding box, perimeter).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from surfaceguard.models.unet import UNet

logger = logging.getLogger(__name__)


@dataclass
class DefectRegion:
    """Attributes of a connected defective region."""
    region_id: int
    area_pixels: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class SegmentationResult:
    """Complete defect segmentation outcome for an image."""
    has_defect: bool
    defect_probability_map: np.ndarray  # (H, W) float32 [0.0, 1.0]
    binary_mask: np.ndarray              # (H, W) uint8 {0, 255}
    defect_regions: List[DefectRegion]
    total_defect_area_pixels: int
    latency_ms: float


class SurfaceSegmentor:
    """Production inference engine for surface defect segmentation models.

    Args:
        model: Pre-constructed PyTorch segmentation model.
        checkpoint_path: Path to model checkpoint .pth file.
        device: 'cpu', 'cuda', or 'auto'.
        image_size: Spatial dimensions for inference.
        threshold: Decision cutoff for defect probability.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str | Path] = None,
        device: str | torch.device = "auto",
        image_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.threshold = threshold

        if isinstance(device, torch.device):
            self.device = device
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is None:
            self.model = UNet(in_channels=3, num_classes=1)
        else:
            self.model = model

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device, weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                elif isinstance(state, dict):
                    self.model.load_state_dict(state)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    @torch.no_grad()
    def predict(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> SegmentationResult:
        """Run segmentation inference on input image.

        Args:
            image: Path to image, RGB numpy array, or PIL Image.

        Returns:
            SegmentationResult instance.
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_w, orig_h = pil_img.size

        # Preprocessing
        resized = TF.resize(pil_img, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(resized)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std).unsqueeze(0).to(self.device)

        # Forward pass & latency timing
        t0 = time.perf_counter()
        output = self.model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(output, dict):
            prob_map = output["mask"].squeeze().cpu().numpy()
        else:
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

        # Resize probability map back to original image dimensions
        prob_map_orig = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        bin_mask = (prob_map_orig >= self.threshold).astype(np.uint8) * 255

        # Separate connected defect regions
        regions = self._extract_defect_regions(bin_mask)
        total_defect_area = int(sum(r.area_pixels for r in regions))

        return SegmentationResult(
            has_defect=len(regions) > 0,
            defect_probability_map=prob_map_orig,
            binary_mask=bin_mask,
            defect_regions=regions,
            total_defect_area_pixels=total_defect_area,
            latency_ms=round(latency_ms, 2),
        )

    @staticmethod
    def _extract_defect_regions(bin_mask: np.ndarray) -> List[DefectRegion]:
        """Identify connected defect blobs and calculate spatial characteristics."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
        regions: List[DefectRegion] = []

        # Index 0 is background; iterate from 1 to num_labels - 1
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 8:  # filter noise blobs smaller than 8 pixels
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[i][0])
            cy = float(centroids[i][1])

            regions.append(
                DefectRegion(
                    region_id=i,
                    area_pixels=area,
                    centroid=(cx, cy),
                    bbox=(x, y, w, h),
                )
            )

        return regions

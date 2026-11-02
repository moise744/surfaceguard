"""
Dataset loader for Industrial Surface Defect Segmentation.

Loads paired images and pixel-level ground truth binary masks.
Compatible with standard MVTec AD defect directories and custom surface inspection sets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from surfaceguard.data.transforms import PairedSegmentationTransforms

logger = logging.getLogger(__name__)


class SurfaceDefectDataset(Dataset):
    """Loads paired images and pixel-level defect masks.

    Directory structure expected:
        <root>/<category>/test/<defect_type>/<img_name>.png
        <root>/<category>/ground_truth/<defect_type>/<img_name>_mask.png (or .png)

    Also handles normal (good) images with automatic all-zero empty masks.
    """

    def __init__(
        self,
        root: str | Path,
        category: str = "bottle",
        split: str = "test",
        transform: Optional[PairedSegmentationTransforms] = None,
        include_good: bool = True,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform or PairedSegmentationTransforms(is_training=False)
        self.include_good = include_good

        self.samples: List[Dict[str, object]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Scan directories and build list of paired image & mask samples."""
        cat_dir = self.root / self.category
        split_dir = cat_dir / self.split
        gt_dir = cat_dir / "ground_truth"

        if not split_dir.exists():
            logger.warning("Split directory not found: %s", split_dir)
            return

        for defect_folder in sorted(split_dir.iterdir()):
            if not defect_folder.is_dir():
                continue

            defect_name = defect_folder.name
            is_good = defect_name in ("good", "normal")

            if is_good and not self.include_good:
                continue

            for img_path in sorted(defect_folder.glob("*.*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue

                mask_path = None
                if not is_good and gt_dir.exists():
                    gt_defect_folder = gt_dir / defect_name
                    # MVTec format: image is '001.png', mask is '001_mask.png' or '001.png'
                    candidate_1 = gt_defect_folder / f"{img_path.stem}_mask.png"
                    candidate_2 = gt_defect_folder / f"{img_path.name}"
                    if candidate_1.exists():
                        mask_path = candidate_1
                    elif candidate_2.exists():
                        mask_path = candidate_2

                self.samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "defect_name": defect_name,
                    "is_defective": not is_good,
                })

        logger.info(
            "SurfaceDefectDataset loaded %d samples for %s/%s",
            len(self.samples), self.category, self.split
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | bool]:
        item = self.samples[idx]
        img_path = Path(str(item["image_path"]))
        image = Image.open(img_path).convert("RGB")

        # Load mask or generate blank mask for normal products
        if item["mask_path"] is not None and Path(str(item["mask_path"])).exists():
            mask = Image.open(str(item["mask_path"])).convert("L")
        else:
            mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))

        img_tensor, mask_tensor = self.transform(image, mask)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "defect_name": str(item["defect_name"]),
            "is_defective": bool(item["is_defective"]),
            "image_path": str(img_path),
        }

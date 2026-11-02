"""
Paired transformations for image and ground-truth segmentation masks.

Ensures spatial augmentations (flips, crops, rotations) are synchronously
applied to both the input RGB image and the corresponding binary mask.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class PairedSegmentationTransforms:
    """Synchronous transformation pipeline for paired image and mask."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.is_training = is_training
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image and mask.

        Args:
            image: PIL RGB image.
            mask: PIL Grayscale binary mask (0 = good, 255 = defective).

        Returns:
            Tuple of (image_tensor, mask_tensor):
            - image_tensor: (3, H, W) normalized float32
            - mask_tensor: (1, H, W) binary float32 in {0.0, 1.0}
        """
        # Resize both with appropriate interpolation
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_training:
            # Synchronous random horizontal flip
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Synchronous random vertical flip
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Photometric augmentation (applied only to image, never mask)
            if random.random() < 0.3:
                image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.85, 1.15))

        # Convert image to tensor & normalize
        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor, mean=self.mean, std=self.std)

        # Convert mask to single-channel binary float tensor {0.0, 1.0}
        mask_array = np.array(mask, dtype=np.float32)
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
        # Binarize: any pixel > 0 is defect
        mask_binary = (mask_array > 127.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # (1, H, W)

        return img_tensor, mask_tensor

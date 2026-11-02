"""SurfaceGuard — Advanced Industrial Defect Segmentation System.

Pixel-level defect localization and segmentation for industrial quality assurance.
"""

__version__ = "0.1.0"
__author__ = "Habimana Moise"

from surfaceguard.models.unet import UNet
from surfaceguard.models.deeplabv3 import DeepLabV3Plus
from surfaceguard.models.losses import CombinedBCEDiceLoss, DiceLoss, FocalLoss
from surfaceguard.inference.segmentor import SurfaceSegmentor

__all__ = [
    "UNet",
    "DeepLabV3Plus",
    "DiceLoss",
    "FocalLoss",
    "CombinedBCEDiceLoss",
    "SurfaceSegmentor",
]

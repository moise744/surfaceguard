"""
DeepLabV3+ Architecture with Atrous Spatial Pyramid Pooling (ASPP).

Reference:
    Chen et al. "Encoder-Decoder with Atrous Separable Convolution for
    Semantic Image Segmentation", ECCV 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_channels: int, atrous_rates: tuple[int, int, int] = (6, 12, 18), out_channels: int = 128) -> None:
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ semantic segmentation model for defect masks.

    Args:
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output segmentation masks (1 for binary defect).
        base_channels: Channel width factor.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels

        # Lightweight custom encoder producing low-level and high-level features
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=3, stride=2, padding=1, bias=False),  # H/2
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b * 2, kernel_size=3, stride=2, padding=1, bias=False),        # H/4 (low-level)
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
        )

        self.middle = nn.Sequential(
            nn.Conv2d(b * 2, b * 4, kernel_size=3, stride=2, padding=1, bias=False),    # H/8
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 4, b * 8, kernel_size=3, stride=2, padding=1, bias=False),    # H/16 (high-level)
            nn.BatchNorm2d(b * 8),
            nn.ReLU(inplace=True),
        )

        # ASPP on high-level features
        self.aspp = ASPP(in_channels=b * 8, out_channels=b * 4)

        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(b * 2, b, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        # Decoder fusion
        self.decoder = nn.Sequential(
            nn.Conv2d(b * 4 + b, b * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(b * 2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[2:]

        low_level = self.entry(x)          # (B, b*2, H/4, W/4)
        high_level = self.middle(low_level) # (B, b*8, H/16, W/16)

        aspp_features = self.aspp(high_level)  # (B, b*4, H/16, W/16)
        aspp_up = F.interpolate(aspp_features, size=low_level.shape[2:], mode="bilinear", align_corners=False)

        low_proj = self.low_level_proj(low_level)  # (B, b, H/4, W/4)
        fused = torch.cat([aspp_up, low_proj], dim=1)

        dec_out = self.decoder(fused)
        logits = F.interpolate(dec_out, size=(H, W), mode="bilinear", align_corners=False)
        mask = torch.sigmoid(logits)

        return {"logits": logits, "mask": mask}

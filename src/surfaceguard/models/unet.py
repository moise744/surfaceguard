"""
Classic U-Net Architecture with Skip Connections for Industrial Defect Segmentation.

Reference:
    Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with Bilinear Interpolation or ConvTranspose then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle spatial padding if dims differ by 1 px due to odd dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Concatenate skip connection along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net Defect Segmentation Network.

    Args:
        in_channels: Input channels (3 for RGB).
        num_classes: Output mask channels (1 for binary defect segmentation).
        base_features: Number of channels in the first feature layer (default 32).
        bilinear: If True, uses bilinear upsampling instead of transposed conv.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_features: int = 32,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        f = base_features
        factor = 2 if bilinear else 1

        # Encoder path
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, (f * 16) // factor)

        # Decoder path with skip connections
        self.up1 = Up(f * 16, (f * 8) // factor, bilinear)
        self.up2 = Up(f * 8, (f * 4) // factor, bilinear)
        self.up3 = Up(f * 4, (f * 2) // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)

        # Output projection head (1x1 conv)
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns:
            Dictionary with:
            - 'logits': raw unnormalized scores (B, 1, H, W)
            - 'mask': sigmoid-activated probability map (B, 1, H, W)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        probs = torch.sigmoid(logits)

        return {"logits": logits, "mask": probs}

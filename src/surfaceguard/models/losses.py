"""
Loss functions for industrial defect segmentation.

Includes:
- DiceLoss: Directly maximizes pixel-level F1 / overlap on highly unbalanced defect pixels
- FocalLoss: Down-weights easy background pixels, focusing on hard defect boundaries
- CombinedBCEDiceLoss: Weighted sum of Binary Cross Entropy and Dice Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    L_dice = 1 - (2 * |P * G| + smooth) / (|P| + |G| + smooth)
    """

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: Predicted raw logits before sigmoid (B, 1, H, W).
            targets: Binary ground truth masks in {0, 1} (B, 1, H, W).
        """
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        denominator = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth + self.eps)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance between background and tiny defects."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)

        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        return (focal_weight * bce_loss).mean()


class CombinedBCEDiceLoss(nn.Module):
    """Weighted composite loss: w_bce * BCE + w_dice * Dice."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice

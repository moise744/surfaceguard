"""Unit tests for segmentation loss functions."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from surfaceguard.models.losses import CombinedBCEDiceLoss, DiceLoss, FocalLoss


class TestLosses:
    def test_dice_loss_perfect_prediction(self):
        loss_fn = DiceLoss()
        targets = torch.zeros(1, 1, 32, 32)
        targets[:, :, 10:20, 10:20] = 1.0

        # Highly positive logits where target=1, highly negative where target=0
        logits = torch.where(targets == 1.0, 10.0, -10.0)

        loss = loss_fn(logits, targets)
        assert loss.item() < 0.05  # near zero loss for perfect match

    def test_dice_loss_complete_mismatch(self):
        loss_fn = DiceLoss()
        targets = torch.zeros(1, 1, 32, 32)
        targets[:, :, 0:10, 0:10] = 1.0

        # Predict opposite
        logits = torch.where(targets == 1.0, -10.0, 10.0)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.8  # high loss for complete mismatch

    def test_focal_loss_computation(self):
        loss_fn = FocalLoss()
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_combined_loss_computation(self):
        loss_fn = CombinedBCEDiceLoss(bce_weight=0.6, dice_weight=0.4)
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

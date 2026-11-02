"""
Segmentation Model Trainer.

Handles training loops, loss calculation, validation metrics tracking,
best checkpoint persistence, and early stopping.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from surfaceguard.evaluation.metrics import compute_dice_coefficient, compute_iou_score
from surfaceguard.models.losses import CombinedBCEDiceLoss

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """Trains segmentation architectures on paired image-mask datasets.

    Args:
        model: PyTorch model outputting {'logits': ..., 'mask': ...}.
        device: Compute device.
        loss_fn: Loss module (defaults to CombinedBCEDiceLoss).
        learning_rate: Initial Adam learning rate.
        weight_decay: L2 regularization factor.
        output_dir: Directory where checkpoints and history are saved.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn or CombinedBCEDiceLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images)
            logits = out["logits"] if isinstance(out, dict) else out

            loss = self.loss_fn(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation loader and compute Dice & IoU."""
        self.model.eval()
        total_loss = 0.0
        all_dices: List[float] = []
        all_ious: List[float] = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            out = self.model(images)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = out["mask"] if isinstance(out, dict) else torch.sigmoid(out)

            loss = self.loss_fn(logits, masks)
            total_loss += float(loss.item())

            p_np = probs.cpu().numpy()
            m_np = masks.cpu().numpy()

            for i in range(len(p_np)):
                d = compute_dice_coefficient(p_np[i, 0], m_np[i, 0])
                iou = compute_iou_score(p_np[i, 0], m_np[i, 0])
                all_dices.append(d)
                all_ious.append(iou)

        n_batches = len(dataloader)
        return {
            "val_loss": round(total_loss / max(1, n_batches), 4),
            "mean_dice": round(float(np.mean(all_dices)) if all_dices else 0.0, 4),
            "mean_iou": round(float(np.mean(all_ious)) if all_ious else 0.0, 4),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        early_stopping_patience: int = 5,
    ) -> Dict[str, list]:
        """Train model across epochs and save best model checkpoint."""
        history = {"train_loss": [], "val_loss": [], "mean_dice": [], "mean_iou": []}
        best_dice = -1.0
        patience_counter = 0

        for ep in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            dur = time.time() - t0

            val_dice = val_metrics["mean_dice"]
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(val_metrics["val_loss"])
            history["mean_dice"].append(val_dice)
            history["mean_iou"].append(val_metrics["mean_iou"])

            logger.info(
                "Epoch %02d/%02d [%.1fs] — Loss: %.4f | Val Dice: %.4f | Val IoU: %.4f",
                ep, epochs, dur, train_loss, val_dice, val_metrics["mean_iou"]
            )

            # Checkpoint best model
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                ckpt_path = self.output_dir / "best_model.pth"
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": self.model.state_dict(),
                        "best_dice": best_dice,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d.", ep)
                    break

        # Save history log
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

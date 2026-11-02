"""
Pixel-Level Quantitative Metrics for Defect Segmentation.

Computes:
- Dice Similarity Coefficient (DSC / F1-score)
- Intersection over Union (IoU / Jaccard Index)
- Pixel-level Precision, Recall, and Accuracy
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def compute_pixel_confusion_matrix(
    pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
) -> Tuple[int, int, int, int]:
    """Compute (TP, FP, FN, TN) counts between predicted and ground-truth masks.

    Args:
        pred_mask: Predicted probabilities in [0.0, 1.0] or binary values.
        gt_mask: Ground-truth binary mask in {0, 1}.
        threshold: Binarization cutoff.

    Returns:
        Tuple of (TP, FP, FN, TN).
    """
    p_bin = (pred_mask >= threshold).astype(bool)
    g_bin = (gt_mask >= 0.5).astype(bool)

    tp = int(np.logical_and(p_bin, g_bin).sum())
    fp = int(np.logical_and(p_bin, ~g_bin).sum())
    fn = int(np.logical_and(~p_bin, g_bin).sum())
    tn = int(np.logical_and(~p_bin, ~g_bin).sum())

    return tp, fp, fn, tn


def compute_dice_coefficient(
    pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5, eps: float = 1e-7
) -> float:
    """Compute Dice Coefficient: 2*|P ∩ G| / (|P| + |G|)."""
    tp, fp, fn, _ = compute_pixel_confusion_matrix(pred_mask, gt_mask, threshold)
    denom = 2 * tp + fp + fn
    if denom == 0:
        # Both pred and gt are completely empty (normal surface) -> perfect match
        return 1.0
    return float((2.0 * tp) / (denom + eps))


def compute_iou_score(
    pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5, eps: float = 1e-7
) -> float:
    """Compute Intersection over Union (IoU / Jaccard Index): |P ∩ G| / |P ∪ G|."""
    tp, fp, fn, _ = compute_pixel_confusion_matrix(pred_mask, gt_mask, threshold)
    union = tp + fp + fn
    if union == 0:
        return 1.0
    return float(tp / (union + eps))


def compute_segmentation_metrics(
    pred_mask: np.ndarray | torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive segmentation evaluation report for an image or batch."""
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    tp, fp, fn, tn = compute_pixel_confusion_matrix(pred_mask, gt_mask, threshold)

    dice = compute_dice_coefficient(pred_mask, gt_mask, threshold)
    iou = compute_iou_score(pred_mask, gt_mask, threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if fn == 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if fp == 0 else 0.0
    total_pixels = tp + fp + fn + tn
    accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 1.0

    return {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "pixel_accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }

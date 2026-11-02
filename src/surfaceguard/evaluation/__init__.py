"""Segmentation evaluation metrics and failure case analysis."""
from surfaceguard.evaluation.metrics import (
    compute_dice_coefficient,
    compute_iou_score,
    compute_pixel_confusion_matrix,
    compute_segmentation_metrics,
)
from surfaceguard.evaluation.failure_analysis import FailureCaseAnalyzer

__all__ = [
    "compute_dice_coefficient",
    "compute_iou_score",
    "compute_pixel_confusion_matrix",
    "compute_segmentation_metrics",
    "FailureCaseAnalyzer",
]

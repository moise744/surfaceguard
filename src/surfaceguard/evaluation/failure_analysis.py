"""
Failure Case and Boundary Error Analyzer for Defect Segmentation.

Categorizes segmentation failure patterns:
- Under-segmentation / missed defects (False Negatives)
- Over-segmentation / false alarms (False Positives)
- Boundary erosion and boundary dilation errors
- Defect size sensitivity (small vs. large defect masks)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class FailureCaseAnalyzer:
    """Analyzes pixel-level errors and categorizes failure modes."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def analyze_sample(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        image_path: str = "",
    ) -> Dict[str, object]:
        """Examine a single sample for failure patterns.

        Args:
            pred_mask: Predicted probabilities (H, W).
            gt_mask: Ground-truth binary mask (H, W).
            image_path: Source image path.

        Returns:
            Dictionary with error diagnostics.
        """
        p_bin = (pred_mask >= self.threshold).astype(np.uint8)
        g_bin = (gt_mask >= 0.5).astype(np.uint8)

        tp_mask = np.logical_and(p_bin, g_bin).astype(np.uint8)
        fp_mask = np.logical_and(p_bin, ~g_bin).astype(np.uint8)
        fn_mask = np.logical_and(~p_bin, g_bin).astype(np.uint8)

        gt_area = int(g_bin.sum())
        pred_area = int(p_bin.sum())
        fp_area = int(fp_mask.sum())
        fn_area = int(fn_mask.sum())

        # Connected component analysis for defect fragmentation
        num_gt_blobs, _, _, _ = cv2.connectedComponentsWithStats(g_bin)
        num_pred_blobs, _, _, _ = cv2.connectedComponentsWithStats(p_bin)

        failure_mode = "accurate"
        if gt_area == 0 and pred_area > 0:
            failure_mode = "false_alarm"
        elif gt_area > 0 and pred_area == 0:
            failure_mode = "completely_missed"
        elif fn_area > fp_area and fn_area > gt_area * 0.4:
            failure_mode = "under_segmented"
        elif fp_area > fn_area and fp_area > pred_area * 0.4:
            failure_mode = "over_segmented"

        return {
            "image_path": image_path,
            "failure_mode": failure_mode,
            "gt_pixel_area": gt_area,
            "pred_pixel_area": pred_area,
            "false_positive_pixels": fp_area,
            "false_negative_pixels": fn_area,
            "num_ground_truth_defects": max(0, num_gt_blobs - 1),
            "num_predicted_defects": max(0, num_pred_blobs - 1),
        }

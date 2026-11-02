"""
Mask Visualization and Inspection Overlay Engine.

Generates:
- Alpha-blended semi-transparent defect masks
- Distinct contour outlines around defect boundaries
- 4-panel diagnostic panels:
  [Original Image] | [Ground Truth] | [Predicted Mask] | [Error Map: TP/FP/FN]
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class MaskVisualizer:
    """Renders visual segmentation diagnostics on industrial product images."""

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.45,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Blend a single binary defect mask onto an RGB/BGR image.

        Args:
            image: Base uint8 image (H, W, 3).
            mask: Probability or binary mask (H, W).
            color: Color tuple (B, G, R).
            alpha: Transparency weight.
            threshold: Cutoff for predicted mask.

        Returns:
            Blended uint8 image with defect regions highlighted.
        """
        canvas = image.copy()
        bin_mask = (mask >= threshold).astype(bool)

        if not np.any(bin_mask):
            return canvas

        colored_overlay = canvas.copy()
        colored_overlay[bin_mask] = color

        # Weighted blend
        blended = cv2.addWeighted(colored_overlay, alpha, canvas, 1.0 - alpha, 0)

        # Draw contour boundary lines around defect blobs
        uint_mask = (bin_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(uint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, color, 2, cv2.LINE_AA)

        return blended

    @staticmethod
    def create_comparison_panel(
        image: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Construct side-by-side diagnostic inspection canvas:

        [Original Image] | [Predicted Mask] | [Ground Truth] | [Error Heatmap]
        """
        h, w = image.shape[:2]

        # 1. Prediction Overlay
        p_overlay = MaskVisualizer.overlay_mask(
            image, pred_mask, color=(0, 0, 240), alpha=0.45, threshold=threshold
        )

        panels = [image, p_overlay]

        # 2. Ground Truth Overlay (Green)
        if gt_mask is not None:
            gt_overlay = MaskVisualizer.overlay_mask(
                image, gt_mask, color=(0, 220, 0), alpha=0.45, threshold=0.5
            )
            panels.append(gt_overlay)

            # 3. Error Map: Green=TP, Red=FP, Blue=FN
            p_bin = (pred_mask >= threshold).astype(bool)
            g_bin = (gt_mask >= 0.5).astype(bool)

            err_map = np.zeros((h, w, 3), dtype=np.uint8)
            err_map[np.logical_and(p_bin, g_bin)] = (0, 255, 0)    # TP: Green
            err_map[np.logical_and(p_bin, ~g_bin)] = (0, 0, 255)   # FP: Red (False Alarm)
            err_map[np.logical_and(~p_bin, g_bin)] = (255, 0, 0)   # FN: Blue (Missed Defect)

            panels.append(err_map)

        return np.hstack(panels)

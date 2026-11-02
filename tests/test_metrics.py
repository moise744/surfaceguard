"""Unit tests for segmentation metrics and failure case analyzer."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from surfaceguard.evaluation.failure_analysis import FailureCaseAnalyzer
from surfaceguard.evaluation.metrics import (
    compute_dice_coefficient,
    compute_iou_score,
    compute_segmentation_metrics,
)


class TestMetrics:
    def test_perfect_dice_and_iou(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:50, 20:50] = 1.0

        dice = compute_dice_coefficient(mask, mask)
        iou = compute_iou_score(mask, mask)

        assert pytest.approx(dice, abs=1e-3) == 1.0
        assert pytest.approx(iou, abs=1e-3) == 1.0

    def test_both_empty_surfaces(self):
        empty_pred = np.zeros((64, 64), dtype=np.float32)
        empty_gt = np.zeros((64, 64), dtype=np.float32)

        dice = compute_dice_coefficient(empty_pred, empty_gt)
        iou = compute_iou_score(empty_pred, empty_gt)

        assert dice == 1.0
        assert iou == 1.0

    def test_segmentation_metrics_dictionary(self):
        p = np.zeros((50, 50), dtype=np.float32)
        g = np.zeros((50, 50), dtype=np.float32)
        p[10:30, 10:30] = 1.0
        g[10:30, 10:30] = 1.0

        metrics = compute_segmentation_metrics(p, g)
        assert "dice" in metrics
        assert "iou" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert metrics["dice"] == 1.0
        assert metrics["precision"] == 1.0


class TestFailureCaseAnalyzer:
    def test_false_alarm_diagnosis(self):
        analyzer = FailureCaseAnalyzer()
        p = np.ones((50, 50), dtype=np.float32)
        g = np.zeros((50, 50), dtype=np.float32)

        res = analyzer.analyze_sample(p, g)
        assert res["failure_mode"] == "false_alarm"

    def test_missed_defect_diagnosis(self):
        analyzer = FailureCaseAnalyzer()
        p = np.zeros((50, 50), dtype=np.float32)
        g = np.ones((50, 50), dtype=np.float32)

        res = analyzer.analyze_sample(p, g)
        assert res["failure_mode"] == "completely_missed"

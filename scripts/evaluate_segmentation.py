"""
SurfaceGuard — Segmentation Evaluation Script.

Evaluates a trained U-Net or DeepLabV3+ checkpoint on the test set
and reports Dice score, IoU, precision, recall.

Usage:
    python scripts/evaluate_segmentation.py --checkpoint outputs/models/best_unet.pth \\
                                             --config configs/unet_mvtec.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from surfaceguard.data.dataset import SurfaceDefectDataset
from surfaceguard.data.transforms import PairedSegmentationTransforms
from surfaceguard.evaluation.metrics import (
    compute_segmentation_metrics,
    compute_dice_coefficient,
    compute_iou_score,
)
from surfaceguard.evaluation.failure_analysis import FailureCaseAnalyzer
from surfaceguard.models.deeplabv3 import DeepLabV3Plus
from surfaceguard.models.unet import UNet


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SurfaceGuard segmentation model")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/unet_mvtec.yaml"))
    p.add_argument("--output", type=Path, default=Path("outputs/segmentation_eval.json"))
    return p.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.is_file():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("[INFO]  Train first: python scripts/train_segmentation.py")
        sys.exit(1)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Data
    img_size = tuple(cfg["data"]["image_size"])
    val_tfm = PairedSegmentationTransforms(image_size=img_size, is_training=False)
    data_root = Path(cfg["data"]["root"])
    category = cfg["data"]["category"]

    if not data_root.exists():
        print(f"[WARN] Dataset not found at {data_root}. See data/README.md.")
        sys.exit(0)

    test_ds = SurfaceDefectDataset(root=data_root, category=category, split="test", transform=val_tfm)
    if len(test_ds) == 0:
        print("[WARN] No test samples found.")
        sys.exit(0)

    loader = DataLoader(test_ds, batch_size=cfg["data"].get("batch_size", 8), shuffle=False)
    print(f"[INFO] Test samples: {len(test_ds)}")

    # Model
    arch = cfg["model"].get("architecture", "unet")
    if arch == "unet":
        model = UNet(
            in_channels=cfg["model"].get("in_channels", 3),
            num_classes=cfg["model"].get("num_classes", 1),
        )
    else:
        model = DeepLabV3Plus(
            in_channels=cfg["model"].get("in_channels", 3),
            num_classes=cfg["model"].get("num_classes", 1),
        )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()
    print(f"[INFO] Loaded: {args.checkpoint}")

    # Inference
    all_dice, all_iou, all_prec, all_rec = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].float()

            out = model(images)
            pred_masks = (torch.sigmoid(out["logits"]) > 0.5).cpu().float()

            for i in range(images.size(0)):
                pred = pred_masks[i, 0].numpy()
                gt = masks[i, 0].numpy()
                metrics = compute_segmentation_metrics(pred, gt)
                all_dice.append(metrics["dice"])
                all_iou.append(metrics["iou"])
                all_prec.append(metrics["precision"])
                all_rec.append(metrics["recall"])

    print(f"\n{'='*55}")
    print(f"  SURFACEGUARD — SEGMENTATION EVALUATION")
    print(f"{'='*55}")
    print(f"  Mean Dice Score : {np.mean(all_dice):.4f}")
    print(f"  Mean IoU        : {np.mean(all_iou):.4f}")
    print(f"  Mean Precision  : {np.mean(all_prec):.4f}")
    print(f"  Mean Recall     : {np.mean(all_rec):.4f}")
    print(f"  Samples         : {len(all_dice)}")
    print(f"{'='*55}")

    result = {
        "mean_dice": round(float(np.mean(all_dice)), 4),
        "mean_iou": round(float(np.mean(all_iou)), 4),
        "mean_precision": round(float(np.mean(all_prec)), 4),
        "mean_recall": round(float(np.mean(all_rec)), 4),
        "num_samples": len(all_dice),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()

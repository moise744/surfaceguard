"""
SurfaceGuard — Training Script for Defect Segmentation.

Trains U-Net or DeepLabV3+ on paired image-mask datasets.

Usage:
    python scripts/train_segmentation.py --config configs/unet_mvtec.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml
from torch.utils.data import DataLoader

from surfaceguard.data.dataset import SurfaceDefectDataset
from surfaceguard.data.transforms import PairedSegmentationTransforms
from surfaceguard.models.deeplabv3 import DeepLabV3Plus
from surfaceguard.models.losses import CombinedBCEDiceLoss, DiceLoss, FocalLoss
from surfaceguard.models.unet import UNet
from surfaceguard.training.trainer import SegmentationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SurfaceGuard segmentation model")
    parser.add_argument("--config", type=Path, default=Path("configs/unet_mvtec.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Transforms
    img_size = tuple(cfg["data"]["image_size"])
    train_tfm = PairedSegmentationTransforms(image_size=img_size, is_training=True)
    val_tfm = PairedSegmentationTransforms(image_size=img_size, is_training=False)

    # Datasets
    data_root = Path(cfg["data"]["root"])
    cat = cfg["data"]["category"]
    batch_size = cfg["data"].get("batch_size", 8)

    train_ds = SurfaceDefectDataset(root=data_root, category=cat, split="test", transform=train_tfm)
    val_ds = SurfaceDefectDataset(root=data_root, category=cat, split="test", transform=val_tfm)

    if len(train_ds) == 0:
        print(f"[WARN] No samples found in {data_root / cat}. Ensure dataset is downloaded.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    arch = cfg["model"].get("architecture", "unet").lower()
    if arch == "deeplabv3plus":
        model = DeepLabV3Plus(in_channels=3, num_classes=1)
    else:
        model = UNet(in_channels=3, num_classes=1)

    # Loss
    loss_type = cfg.get("loss", {}).get("type", "combined_bce_dice")
    if loss_type == "focal":
        loss_fn = FocalLoss()
    elif loss_type == "dice":
        loss_fn = DiceLoss()
    else:
        loss_fn = CombinedBCEDiceLoss()

    output_dir = Path(cfg.get("output", {}).get("dir", "outputs/unet"))
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        loss_fn=loss_fn,
        learning_rate=cfg["training"]["learning_rate"],
        output_dir=output_dir,
    )

    print(f"[INFO] Starting training for {cfg['training']['epochs']} epochs...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 5),
    )
    print(f"[INFO] Finished! Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()

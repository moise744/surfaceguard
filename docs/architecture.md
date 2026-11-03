# SurfaceGuard — System Architecture

## Overview

SurfaceGuard is a pixel-level industrial defect segmentation system. Unlike object detection (which produces bounding boxes), segmentation produces per-pixel masks that precisely delineate defective regions — critical for estimating defect area, shape, and severity.

---

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Input Layer                                  │
│      Industrial surface image + optional ground-truth mask           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │   Preprocessing    │
                   │  Resize → Pad      │
                   │  Normalize         │
                   │  Paired augment    │  ← image and mask augmented
                   └─────────┬──────────┘     identically
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼────┐  ┌──────▼──────┐ ┌────▼───────────┐
    │   U-Net      │  │ DeepLabV3+  │ │ YOLO-Seg       │
    │  skip conns  │  │  ASPP       │ │ instance seg   │
    │  encoder-    │  │  atrous     │ │  head          │
    │  decoder     │  │  convs      │ │                │
    └─────────┬────┘  └──────┬──────┘ └────┬───────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │ predicted mask [H×W] ∈ {0,1}
                   ┌─────────▼──────────┐
                   │  Loss Computation  │
                   │  BCE + Dice Loss   │
                   │  (combined)        │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Metric Evaluation │
                   │  Dice / IoU /      │
                   │  Pixel Precision   │
                   │  Pixel Recall      │
                   └─────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼────┐  ┌──────▼──────┐ ┌────▼───────────┐
    │ Mask overlay │  │  FastAPI    │ │  Streamlit     │
    │ visualization│  │  REST API   │ │  Dashboard     │
    └──────────────┘  └─────────────┘ └────────────────┘
```

---

## Component Descriptions

| Component | Location | Responsibility |
|-----------|----------|---------------|
| `MVTecSegDataset` | `src/surfaceguard/data/dataset.py` | Loads MVTec AD image + mask pairs; handles categories with no defect masks (normal images); returns `{image, mask, label}` |
| `PairedSegmentationTransforms` | `src/surfaceguard/data/transforms.py` | Applies identical random spatial transforms to both image and mask; colour augmentation applied to image only |
| `validate_masks` | `src/surfaceguard/data/validation.py` | Checks that mask dimensions match image dimensions; verifies binary mask values; reports corrupt files |
| `UNet` | `src/surfaceguard/models/unet.py` | Standard U-Net with configurable encoder depth; skip connections concatenate encoder features with upsampled decoder features; output is a single-channel logit map |
| `DeepLabV3Plus` | `src/surfaceguard/models/deeplabv3plus.py` | Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale context; lightweight decoder with skip connection from low-level features |
| `YOLOSegWrapper` | `src/surfaceguard/models/yolo_seg.py` | Wraps Ultralytics YOLO segmentation model; converts instance mask outputs to binary defect mask |
| `BCEDiceLoss` | `src/surfaceguard/losses/combined.py` | Linear combination of BCE loss and Dice loss; configurable weight `alpha`; handles class imbalance better than BCE alone |
| `FocalLoss` | `src/surfaceguard/losses/focal.py` | Down-weights easy negatives; effective when defect pixels are a tiny fraction of the image |
| `compute_segmentation_metrics` | `src/surfaceguard/metrics/segmentation.py` | Computes Dice coefficient, IoU (Jaccard), pixel precision, pixel recall, and pixel F1 from predicted and ground-truth masks |
| `SurfaceSegmentor` | `src/surfaceguard/inference/segmentor.py` | Inference wrapper; loads checkpoint; runs forward pass; returns binary mask, confidence map, and per-metric scores |
| `api.py` | `app/api.py` | FastAPI endpoints: `POST /segment` (image → mask + metrics), `GET /health` |
| `streamlit_app.py` | `app/streamlit_app.py` | Upload image; display predicted mask overlay; compare with ground-truth mask if provided |

---

## Data Flow

```
MVTec AD image + ground-truth mask
          │
          ▼
PairedSegmentationTransforms
  (resize, flip, normalize — identical for image and mask)
          │
          ▼
Model forward pass (U-Net / DeepLabV3+)
  input:  [B, 3, H, W]
  output: [B, 1, H, W]  logits
          │
          ▼
sigmoid(logits) → probability map [0,1]
threshold (default 0.5) → binary mask {0,1}
          │
          ▼
BCEDiceLoss(predicted, ground_truth) → backprop
          │
          ▼
compute_segmentation_metrics(predicted_mask, gt_mask)
  → {dice, iou, pixel_precision, pixel_recall, pixel_f1}
          │
          ▼
Mask overlay visualization / FastAPI JSON / Streamlit display
```

---

## U-Net Architecture Detail

```
Encoder (contracting path)         Decoder (expanding path)
──────────────────────             ──────────────────────
[3, H,  W]  → Conv×2 → [64, H,  W]      ┐
             MaxPool  → [64, H/2,W/2]    │ skip connections
             Conv×2  → [128,H/2,W/2]    │ (concatenation)
             MaxPool  → [128,H/4,W/4]   │
             Conv×2  → [256,H/4,W/4]   ├───────────────────┐
             MaxPool  → [256,H/8,W/8]  │                   │
             Conv×2  → [512,H/8,W/8]  │                   │
             MaxPool  → [512,H/16,W/16]│                   │
Bottleneck:  Conv×2  → [1024,H/16,W/16]│                  │
                                        │                  │
             Upsample → [512, H/8,W/8] ◄── concat ────────┘
             Conv×2  → [512, H/8,W/8]
             Upsample → [256, H/4,W/4] ◄── concat
             Conv×2  → [256, H/4,W/4]
             Upsample → [128, H/2,W/2] ◄── concat
             Conv×2  → [128, H/2,W/2]
             Upsample → [64,  H,  W]   ◄── concat
             Conv×2  → [64,  H,  W]
             Conv 1×1 → [1,  H,  W]  (logit map)
```

---

## Key Design Decisions

### 1. BCE + Dice loss combination
Pure BCE loss treats every pixel equally, which is problematic when defects occupy < 1% of the image. The combined BCE+Dice loss provides stable gradients (BCE) while directly optimising the Dice coefficient (a measure of mask overlap).

### 2. Paired augmentation
Spatial augmentations (flip, rotation, crop) must be applied identically to both image and mask. Colour augmentations (brightness, contrast, hue) are applied to images only — masks are binary and must not be colour-augmented.

### 3. Multiple architectures
U-Net is optimised for precise boundary localisation. DeepLabV3+ uses ASPP to capture multi-scale context, improving detection of large irregular defects. The system supports both so users can compare on their own data.

### 4. Threshold tuning
The binary mask threshold (default 0.5) can be adjusted via the Streamlit slider or API parameter to trade precision for recall depending on inspection requirements.

---

## Performance Characteristics

| Model | Parameters | Typical inference (CPU, 256×256) |
|-------|-----------|----------------------------------|
| U-Net (small encoder) | ~7.8 M | ~80–150 ms |
| DeepLabV3+ (MobileNetV3 backbone) | ~5.8 M | ~100–180 ms |

> [!NOTE]
> GPU inference is typically 8–15× faster. Models run in float32 by default; consider float16 (GPU only) for production.

---

## Dataset

**MVTec AD** — the benchmark for industrial surface defect segmentation.
- 15 categories (hazelnut, bottle, carpet, metal_nut, etc.).
- Pixel-level ground-truth masks for all defect classes.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
- Download: https://www.mvtec.com/company/research/datasets/mvtec-ad

See `data/README.md` for download and preprocessing instructions.

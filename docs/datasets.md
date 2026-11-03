# SurfaceGuard — Dataset Guide

## Primary Dataset: MVTec AD

**MVTec AD** is the primary benchmark for pixel-level industrial defect segmentation.

| Property | Value |
|----------|-------|
| Source | MVTec Software GmbH |
| License | CC BY-NC-SA 4.0 (non-commercial) |
| Size | ~4.9 GB |
| Images | 5,354 training + 1,725 test |
| Masks | Pixel-level binary ground-truth masks (PNG) |
| Categories | 15 |

### Download and Setup

1. Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad

2. Extract to `data/mvtec_ad/`:
   ```
   data/
   └── mvtec_ad/
       ├── bottle/
       │   ├── train/good/          ← defect-free (no masks)
       │   ├── test/good/           ← defect-free test
       │   ├── test/broken_large/   ← defect images
       │   └── ground_truth/        ← PNG masks (white=defect)
       ├── carpet/
       ├── hazelnut/
       └── ...
   ```

3. Verify:
   ```bash
   python -c "from surfaceguard.data.dataset import MVTecSegDataset; d = MVTecSegDataset('data/mvtec_ad', 'bottle'); print(f'{len(d)} samples')"
   ```

### Mask Format

Ground-truth masks are PNG images where:
- **White pixels (255)** = defective region
- **Black pixels (0)** = background

The `MVTecSegDataset` loader converts masks to binary float tensors (0.0 / 1.0).

---

## Alternative Datasets

### VisA (Visual Anomaly and Segmentation)

- 12 categories, 10,821 images, pixel-level annotations.
- Paper: https://arxiv.org/abs/2205.09508
- Download: https://github.com/amazon-science/spot-diff

### DAGM 2007

- Simulated textured surface defects with weak labels.
- Download: https://resources.mpi-inf.mpg.de/conference/dagm/2007/

### Custom Data

For custom surfaces, create binary masks using:
- **CVAT** (polygon → binary mask export)
- **Label Studio** (semantic segmentation mode)
- **Labelme** (polygon → mask conversion)

---

## Expected Directory Structure

```
data/
├── mvtec_ad/
│   ├── bottle/
│   │   ├── train/good/
│   │   ├── test/good/
│   │   ├── test/broken_large/
│   │   └── ground_truth/broken_large/
│   └── ... (14 more categories)
├── custom/
│   ├── images/
│   └── masks/         ← binary PNG masks
└── README.md
```

Do not commit dataset images or masks to git.

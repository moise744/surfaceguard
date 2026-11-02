# SurfaceGuard — Segmentation Dataset Guidelines

## MVTec AD Segmentation Masks

SurfaceGuard trains on high-resolution industrial defect datasets with pixel-level ground truth masks.

### Directory Format
```
data/mvtec_ad/
└── <category>/
    ├── test/
    │   ├── good/                # Clean surface images
    │   └── <defect_type>/       # Defective surface images
    └── ground_truth/
        └── <defect_type>/       # Ground-truth binary segmentation masks
```

### Extraction
Place the MVTec AD dataset inside `data/mvtec_ad`. Masks should be single-channel PNG files where white pixels (255) indicate defective surface regions and black (0) represents flawless material.

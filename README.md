<div align="center">

# 🛡️ SurfaceGuard

### Advanced Industrial Defect Segmentation System

Pixel-level defect localization, irregular boundary isolation, and spatial failure diagnostics for manufacturing quality inspection.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Overview

While object detection draws coarse bounding boxes around entire parts, high-precision industrial inspection requires **semantic segmentation** — predicting exactly which pixels correspond to cracks, scratches, voids, contamination, or micro-dents.

**SurfaceGuard** provides deep learning segmentation models specifically engineered for fine defect boundaries:
- **Architectures**:
  - **U-Net**: Contracting and expansive paths with high-resolution skip connections preserving delicate scratch edges.
  - **DeepLabV3+**: Atrous Spatial Pyramid Pooling (ASPP) extracting multi-scale context without loss of spatial resolution.
- **Loss Formulations**: Specialized `CombinedBCEDiceLoss` and `FocalLoss` handling severe class imbalance (where defect pixels often account for $< 1\%$ of the total surface area).
- **Diagnostics**: Connected component blob extraction measuring pixel area, centroids, and contour bounding boxes.
- **Interactive UI**: Streamlit application with live threshold tuning and side-by-side comparison panels.

---

## Architectural Comparison: Detection vs. Segmentation

```
Coarse Object Detection                   Pixel-Level Semantic Segmentation
┌───────────────────────────────┐        ┌───────────────────────────────┐
│     ┌───────────────────┐     │        │          .                    │
│     │   Bounding Box    │     │   vs   │         / \  (Exact Defect   │
│     │   (Includes 80%   │     │        │        / _ \   Boundary)      │
│     │   flawless area)  │     │        │       / / \ \                 │
│     └───────────────────┘     │        │      '-'   '-'                │
└───────────────────────────────┘        └───────────────────────────────┘
```

---

## Project Structure

```
surfaceguard/
├── configs/
│   ├── unet_mvtec.yaml            # U-Net hyperparameters & training config
│   └── deeplab_mvtec.yaml         # DeepLabV3+ ASPP configuration
├── data/
│   └── README.md                  # MVTec segmentation format instructions
├── scripts/
│   ├── train_segmentation.py      # Multi-epoch segmentation training loop
│   └── predict_mask.py            # Single-image mask prediction CLI
├── src/surfaceguard/
│   ├── data/                      # Paired transforms & MVTec dataset loader
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/                    # U-Net, DeepLabV3+, Dice & Focal losses
│   │   ├── unet.py
│   │   ├── deeplabv3.py
│   │   └── losses.py
│   ├── training/                  # Segmentation trainer with early stopping
│   │   └── trainer.py
│   ├── evaluation/                # Dice, IoU, and failure case analyzer
│   │   ├── metrics.py
│   │   └── failure_analysis.py
│   ├── inference/                 # SurfaceSegmentor & blob extractor
│   │   └── segmentor.py
│   └── visualization/             # Alpha-blended masks & 4-panel comparison
│       └── overlay.py
├── app/
│   └── streamlit_app.py           # Interactive inspection & blob table UI
├── tests/                         # Full automated pytest suite
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   └── test_inference.py
├── requirements.txt
├── setup.py
├── .gitignore
└── LICENSE
```

---

## Quick Start

### 1. Installation

```powershell
git clone https://github.com/moise744/surfaceguard.git
cd surfaceguard
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Test Suite

```powershell
python -m pytest tests/ -v
```

### 3. Launch Interactive Segmentation Dashboard

```powershell
streamlit run app/streamlit_app.py
```

### 4. Train on MVTec Defect Category

```powershell
python scripts/train_segmentation.py --config configs/unet_mvtec.yaml
```

---

## Quantitative Metrics

SurfaceGuard evaluates predictions against pixel ground truth masks:
- **Dice Similarity Coefficient (DSC)**: $2 \times \frac{|P \cap G|}{|P| + |G|}$
- **Intersection over Union (IoU / Jaccard Index)**: $\frac{|P \cap G|}{|P \cup G|}$
- **Pixel-Level Precision, Recall, and Accuracy**
- **Connected Defect Region Count & Area (px)**

---

## Author

**Habimana Moise** — Computer Vision & Machine Learning Engineer

- GitHub: [@moise744](https://github.com/moise744)
- LinkedIn: [habimana-moise](https://www.linkedin.com/in/habimana-moise/)
- Email: habimanamoise77@gmail.com

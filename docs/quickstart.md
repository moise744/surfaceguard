# SurfaceGuard — Quick Start Guide

Get up and running in under 5 minutes.

## Prerequisites

- Python 3.9+
- pip

## 1. Clone and Install

```bash
git clone https://github.com/moise744/surfaceguard.git
cd surfaceguard
pip install -e ".[dev]"
```

## 2. Verify Installation

```bash
python -m pytest tests/ -q
# Expected: 17 passed
```

## 3. Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8503
```

Upload a surface image. Select the model (U-Net or DeepLabV3+). Adjust the threshold slider. See the defect mask overlay.

## 4. Run the FastAPI Server

```bash
pip install fastapi uvicorn[standard]
uvicorn app.api:app --host 0.0.0.0 --port 8002 --reload
# API docs at http://localhost:8002/docs
```

## 5. Predict a Mask on a Single Image

```bash
python scripts/predict_mask.py --image path/to/surface.jpg --model unet
```

## 6. Train on MVTec AD

```bash
# Download MVTec AD first (see data/README.md)
python scripts/train_segmentation.py \
  --config configs/unet_mvtec.yaml \
  --category bottle

# Evaluate segmentation metrics
python scripts/evaluate_segmentation.py \
  --checkpoint outputs/best_unet.pth \
  --category bottle
```

## Key Commands Reference

| Command | Purpose |
|---------|---------|
| `make install` | Install production deps |
| `make test` | Run all tests |
| `make run-app` | Launch Streamlit dashboard |
| `make run-api` | Launch FastAPI server |
| `make docker-build` | Build Docker image |

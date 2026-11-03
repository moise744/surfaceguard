# Contributing to SurfaceGuard

Thank you for your interest in contributing to SurfaceGuard — a pixel-level industrial defect segmentation system.

## Table of Contents
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Test Requirements](#test-requirements)
- [Submitting Changes](#submitting-changes)

---

## Development Setup

### Prerequisites
- Python 3.9+
- Git

### Steps

1. **Fork the repository** on GitHub.

2. **Clone your fork:**
   ```bash
   git clone https://github.com/<your-username>/surfaceguard.git
   cd surfaceguard
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   # or
   make install-dev
   ```

5. **Verify the test suite passes before making changes:**
   ```bash
   make test
   ```

---

## Project Structure

```
surfaceguard/
├── src/surfaceguard/
│   ├── data/           # Dataset loading, paired transforms, mask validation
│   ├── models/         # UNet, DeepLabV3+, YOLO segmentation wrapper
│   ├── losses/         # BCEDiceLoss, FocalLoss, combined losses
│   ├── training/       # Segmentation training loop
│   ├── metrics/        # Dice, IoU, pixel-level precision/recall
│   └── inference/      # Segmentor predictor
├── app/
│   ├── streamlit_app.py  # Pixel-level inspection dashboard
│   └── api.py            # FastAPI segmentation endpoints
├── scripts/            # Training, evaluation, segmentation scripts
├── tests/              # Unit tests (pytest)
├── notebooks/          # Dataset EDA and model comparison notebooks
└── docs/               # Architecture and API documentation
```

---

## Code Standards

- **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/). Maximum line length: **100 characters**.
- **Formatting:** Use `black` (`make format`) before committing.
- **Imports:** Sort with `isort` (`make format`).
- **Docstrings:** All public classes and functions **must** have Google-style docstrings.
- **Type hints:** All new functions should include type annotations.
- **No fabricated results:** Dice/IoU scores must come from real model inference — not hardcoded values.

---

## Test Requirements

- All new features **must** include corresponding unit tests.
- All tests must pass: `make test`
- Segmentation model tests use small synthetic tensors (no real dataset required).
- FastAPI tests use `pytest.importorskip("fastapi")`.

---

## Submitting Changes

1. Ensure all tests pass: `make test`
2. Run lint: `make lint`
3. Open a Pull Request with a clear description and example mask visualizations if possible.

---

## Reporting Issues

Open a GitHub Issue with:
- A minimal reproducible example.
- Input image + mask samples if the issue is dataset-related.
- Your Python version and OS.
- The full error traceback.

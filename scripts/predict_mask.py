"""
SurfaceGuard — Defect Mask Inference CLI.

Predicts pixel-level defect masks and outputs side-by-side diagnostic panels.

Usage:
    python scripts/predict_mask.py --image path/to/sample.png --output outputs/mask.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from surfaceguard.inference.segmentor import SurfaceSegmentor
from surfaceguard.visualization.overlay import MaskVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment defect pixels in an image")
    parser.add_argument("--image", type=Path, required=True, help="Input image file")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Trained weights .pth")
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.png"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Cutoff threshold")
    args = parser.parse_args()

    segmentor = SurfaceSegmentor(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )

    print(f"[INFO] Running segmentation on: {args.image}")
    result = segmentor.predict(args.image)

    print("\nRESULTS:")
    print(f"  Defect detected:       {result.has_defect}")
    print(f"  Defect blob count:     {len(result.defect_regions)}")
    print(f"  Defect pixel area:     {result.total_defect_area_pixels} px")
    print(f"  Inference Latency:     {result.latency_ms:.2f} ms")

    # Generate visualization
    raw_bgr = cv2.imread(str(args.image))
    overlay = MaskVisualizer.overlay_mask(
        raw_bgr, result.defect_probability_map, color=(0, 0, 240), threshold=args.threshold
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)
    print(f"[INFO] Annotated overlay saved to: {args.output}")


if __name__ == "__main__":
    main()

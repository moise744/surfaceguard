"""
SurfaceGuard — FastAPI REST API
Endpoints for pixel-level industrial defect segmentation.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Conditional import of SurfaceGuard modules
# ---------------------------------------------------------------------------
try:
    from surfaceguard.inference.segmentor import SurfaceSegmentor
    from surfaceguard.metrics.segmentation import compute_segmentation_metrics
    _SURFACEGUARD_AVAILABLE = True
except ImportError:
    _SURFACEGUARD_AVAILABLE = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
if _FASTAPI_AVAILABLE:
    app = FastAPI(
        title="SurfaceGuard API",
        description=(
            "Pixel-level industrial defect segmentation. "
            "Upload a surface image and receive a binary defect mask, "
            "a heatmap overlay, and segmentation metrics."
        ),
        version="1.0.0",
        contact={
            "name": "Habimana Moise",
            "email": "habimanamoise77@gmail.com",
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Global model state
    # -----------------------------------------------------------------------
    _segmentor: Optional[Any] = None

    def _get_segmentor() -> Any:
        global _segmentor
        if _segmentor is None:
            if not _SURFACEGUARD_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="SurfaceGuard modules not installed. Run `pip install -e .`",
                )
            _segmentor = SurfaceSegmentor(model_type="unet")
        return _segmentor

    def _decode_image(upload_bytes: bytes) -> np.ndarray:
        """Decode uploaded image bytes to a BGR numpy array."""
        if not _CV2_AVAILABLE:
            raise HTTPException(status_code=503, detail="OpenCV not available.")
        arr = np.frombuffer(upload_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        return img

    def _mask_to_base64_png(mask: np.ndarray) -> str:
        """Encode a binary mask (H×W uint8) as a base64 PNG string."""
        _, buf = cv2.imencode(".png", (mask * 255).astype(np.uint8))
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay a binary defect mask on the original image as a red channel."""
        overlay = image.copy()
        defect_pixels = mask.astype(bool)
        overlay[defect_pixels] = overlay[defect_pixels] * (1 - alpha) + np.array([0, 0, 255]) * alpha
        return overlay.astype(np.uint8)

    def _image_to_base64_jpeg(image: np.ndarray, quality: int = 80) -> str:
        """Encode a BGR image to base64 JPEG."""
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @app.get("/health", summary="Health check")
    async def health() -> Dict[str, str]:
        """Returns service health and dependency availability."""
        return {
            "status": "ok",
            "service": "surfaceguard",
            "fastapi": str(_FASTAPI_AVAILABLE),
            "opencv": str(_CV2_AVAILABLE),
            "torch": str(_TORCH_AVAILABLE),
            "surfaceguard_modules": str(_SURFACEGUARD_AVAILABLE),
        }

    @app.post("/segment", summary="Segment defective regions in a surface image")
    async def segment(
        file: UploadFile = File(..., description="Surface image (JPEG/PNG)"),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Mask binarization threshold"),
        return_overlay: bool = Query(True, description="Include base64-encoded overlay image in response"),
        return_mask: bool = Query(False, description="Include base64-encoded binary mask PNG in response"),
    ) -> JSONResponse:
        """
        Run pixel-level defect segmentation on the uploaded surface image.

        Returns:
        - **verdict**: "defective" if any defective pixels detected, else "normal"
        - **defect_pixel_fraction**: fraction of pixels classified as defective
        - **overlay_image_b64**: base64 JPEG of the image with defect mask overlay (if return_overlay=True)
        - **mask_b64**: base64 PNG of the binary mask (if return_mask=True)

        Note: segmentation metrics (Dice, IoU) require a ground-truth mask and
        are not computed by this endpoint.
        """
        segmentor = _get_segmentor()
        image_bytes = await file.read()
        image_bgr = _decode_image(image_bytes)

        # Run segmentation
        result = segmentor.segment(image_bgr, threshold=threshold)

        # Extract mask
        mask = result.get("mask", np.zeros(image_bgr.shape[:2], dtype=np.uint8))
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        mask = mask.squeeze().astype(np.uint8)

        defect_fraction = float(mask.mean())
        verdict = "defective" if defect_fraction > 0.001 else "normal"

        response: Dict[str, Any] = {
            "filename": file.filename,
            "image_shape": {"height": image_bgr.shape[0], "width": image_bgr.shape[1]},
            "threshold": threshold,
            "verdict": verdict,
            "defect_pixel_fraction": round(defect_fraction, 6),
            "defect_pixel_count": int(mask.sum()),
        }

        if return_overlay:
            overlay = _overlay_mask_on_image(image_bgr, mask)
            response["overlay_image_b64"] = _image_to_base64_jpeg(overlay)
            response["overlay_format"] = "base64/jpeg"

        if return_mask:
            response["mask_b64"] = _mask_to_base64_png(mask)
            response["mask_format"] = "base64/png"

        return JSONResponse(content=response)

    @app.post("/segment/compare", summary="Segment and compare against ground-truth mask")
    async def segment_compare(
        image_file: UploadFile = File(..., description="Surface image (JPEG/PNG)"),
        mask_file: UploadFile = File(..., description="Ground-truth binary mask (PNG, white=defect)"),
        threshold: float = Query(0.5, ge=0.0, le=1.0),
    ) -> JSONResponse:
        """
        Run segmentation and compute metrics against a provided ground-truth mask.

        Computes: Dice coefficient, IoU (Jaccard), pixel precision, pixel recall, pixel F1.
        """
        if not _SURFACEGUARD_AVAILABLE:
            raise HTTPException(status_code=503, detail="SurfaceGuard modules not available.")

        segmentor = _get_segmentor()

        image_bytes = await image_file.read()
        image_bgr = _decode_image(image_bytes)

        mask_bytes = await mask_file.read()
        gt_arr = np.frombuffer(mask_bytes, np.uint8)
        gt_mask = cv2.imdecode(gt_arr, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise HTTPException(status_code=400, detail="Could not decode ground-truth mask.")
        gt_binary = (gt_mask > 127).astype(np.uint8)

        result = segmentor.segment(image_bgr, threshold=threshold)
        pred_mask = result.get("mask", np.zeros_like(gt_binary))
        if hasattr(pred_mask, "numpy"):
            pred_mask = pred_mask.numpy()
        pred_mask = pred_mask.squeeze().astype(np.uint8)

        # Resize pred to match gt if needed
        if pred_mask.shape != gt_binary.shape:
            pred_mask = cv2.resize(pred_mask, (gt_binary.shape[1], gt_binary.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        metrics = compute_segmentation_metrics(pred_mask, gt_binary)

        return JSONResponse(content={
            "filename": image_file.filename,
            "threshold": threshold,
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
            "verdict": "defective" if float(pred_mask.mean()) > 0.001 else "normal",
        })

    @app.get("/models", summary="List available segmentation models")
    async def list_models() -> Dict[str, Any]:
        """Lists the available segmentation model architectures."""
        return {
            "models": [
                {
                    "name": "unet",
                    "description": "U-Net with skip connections. Best for precise boundary localisation.",
                    "parameters": "~7.8M",
                },
                {
                    "name": "deeplabv3plus",
                    "description": "DeepLabV3+ with ASPP. Best for multi-scale context.",
                    "parameters": "~5.8M",
                },
            ],
            "active_model": "unet",
        }

    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, str]:
        return {"message": "SurfaceGuard API. Visit /docs for the interactive API reference."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn[standard]")
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)

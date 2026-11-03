# SurfaceGuard — API Reference

Base URL: `http://localhost:8002`

Interactive docs: `http://localhost:8002/docs` (Swagger UI)

---

## Endpoints

### `GET /health`

Health check.

**Response:**
```json
{
  "status": "ok",
  "service": "surfaceguard",
  "fastapi": "True",
  "opencv": "True",
  "torch": "True",
  "surfaceguard_modules": "True"
}
```

---

### `POST /segment`

Run pixel-level defect segmentation on an uploaded image.

**Request:** `multipart/form-data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | — | Surface image (JPEG/PNG) |
| `threshold` | float | `0.5` | Mask binarization threshold (0–1) |
| `return_overlay` | bool | `true` | Include base64 overlay image |
| `return_mask` | bool | `false` | Include base64 binary mask PNG |

**Response:**
```json
{
  "filename": "surface.jpg",
  "image_shape": {"height": 900, "width": 900},
  "threshold": 0.5,
  "verdict": "defective",
  "defect_pixel_fraction": 0.023456,
  "defect_pixel_count": 18993,
  "overlay_image_b64": "<base64-jpeg>",
  "overlay_format": "base64/jpeg"
}
```

**Decoding the overlay image (Python):**
```python
import base64, cv2, numpy as np

b64 = response.json()["overlay_image_b64"]
img_bytes = base64.b64decode(b64)
arr = np.frombuffer(img_bytes, np.uint8)
overlay = cv2.imdecode(arr, cv2.IMREAD_COLOR)
cv2.imwrite("overlay.jpg", overlay)
```

---

### `POST /segment/compare`

Segment and compute metrics against a ground-truth mask.

**Request:** `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_file` | file | ✅ | Surface image (JPEG/PNG) |
| `mask_file` | file | ✅ | Ground-truth binary mask (PNG) |
| `threshold` | float | — | Default `0.5` |

**Response:**
```json
{
  "filename": "bottle_001.jpg",
  "threshold": 0.5,
  "metrics": {
    "dice": 0.823,
    "iou": 0.699,
    "pixel_precision": 0.791,
    "pixel_recall": 0.858,
    "pixel_f1": 0.823
  },
  "verdict": "defective"
}
```

---

### `GET /models`

List available segmentation models.

**Response:**
```json
{
  "models": [
    {"name": "unet", "description": "U-Net with skip connections", "parameters": "~7.8M"},
    {"name": "deeplabv3plus", "description": "DeepLabV3+ with ASPP", "parameters": "~5.8M"}
  ],
  "active_model": "unet"
}
```

---

## Error Responses

| HTTP Code | Meaning |
|-----------|---------|
| `400` | Unreadable image or mask file |
| `503` | SurfaceGuard modules or OpenCV not available |

---

## Python Client Example

```python
import requests, base64, cv2, numpy as np

with open("surface.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8002/segment",
        files={"file": f},
        params={"threshold": 0.4, "return_overlay": True},
    )

data = resp.json()
print("Verdict:", data["verdict"])
print("Defect pixels:", f"{data['defect_pixel_fraction']*100:.2f}%")

# Save overlay
b64 = base64.b64decode(data["overlay_image_b64"])
arr = np.frombuffer(b64, np.uint8)
overlay = cv2.imdecode(arr, cv2.IMREAD_COLOR)
cv2.imwrite("defect_overlay.jpg", overlay)
```

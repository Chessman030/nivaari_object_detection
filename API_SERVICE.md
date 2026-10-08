# Nivaari API Service Documentation

This document describes the HTTP API exposed by the civic issue detection service.

## Base URL

- Local: `http://127.0.0.1:8000`
- Public (optional): your ngrok URL after running `ngrok http 8000`

## Run the API

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The model is loaded on startup from:

- `weights/best.pt`

The server automatically uses:

- `cuda` if available
- otherwise `cpu`

## API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service info and status |
| GET | `/health` | Health and model metadata |
| POST | `/predict` | Detect civic issues in an uploaded image |

---

## 1) GET `/`

Returns basic service information.

### Example Request

```bash
curl http://127.0.0.1:8000/
```

### Example Response

```json
{
  "message": "Nivaari object detection API is running",
  "predict_endpoint": "/predict",
  "model_loaded": true,
  "device": "cuda"
}
```

---

## 2) GET `/health`

Returns model/server health details.

### Example Request

```bash
curl http://127.0.0.1:8000/health
```

### Example Response

```json
{
  "status": "ok",
  "model_path": ".../weights/best.pt",
  "device": "cuda",
  "classes": [
    "Damaged_Concrete_Structures",
    "Damaged_Electric_Poles",
    "Damaged_Road_Signs",
    "Dead_Animal_Pollution",
    "Fallen_Trees",
    "Garbage",
    "Graffiti",
    "pothole",
    "road_crack"
  ]
}
```

---

## 3) POST `/predict`

Runs object detection on one image and returns detections sorted by confidence (highest first).

### Query Parameters

| Name | Type | Required | Default | Range | Description |
|---|---|---|---|---|---|
| `confidence` | float | No | `0.25` | `0.0` to `1.0` | Confidence threshold used by model inference |

### Form Data

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | Yes | Input image (`image/*` content type) |

### Example Request (curl)

```bash
curl -X POST "http://127.0.0.1:8000/predict?confidence=0.3" \
  -F "image=@C:/path/to/image.jpg"
```

### Example Request (PowerShell)

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict?confidence=0.3" `
  -Method Post `
  -Form @{ image = Get-Item "C:\path\to\image.jpg" }
```

### Success Response

```json
{
  "category": "Garbage",
  "confidence": 0.9134,
  "detections": [
    {
      "category": "Garbage",
      "class_id": 5,
      "confidence": 0.9134,
      "bbox": [120, 85, 420, 360]
    }
  ],
  "total_detections": 1,
  "filename": "image.jpg"
}
```

### No Detection Case

If no object passes the threshold:

```json
{
  "category": "No issue detected",
  "confidence": 0.0,
  "detections": [],
  "total_detections": 0,
  "filename": "image.jpg"
}
```

---

## Error Responses

| Status | When it happens | Example `detail` |
|---|---|---|
| `400` | Uploaded file is not an image | `Please upload an image file` |
| `400` | Uploaded file is empty | `Uploaded image is empty` |
| `400` | Invalid or unreadable image bytes | `Invalid image file` |
| `422` | Validation error (missing file or invalid query param) | FastAPI validation payload |
| `503` | Model not loaded yet | `Model is not loaded yet` |

---

## Detection Object Schema

Each item in `detections` contains:

- `category` (string): class name
- `class_id` (integer): class index
- `confidence` (float): prediction confidence
- `bbox` (array of 4 integers): `[x1, y1, x2, y2]`

Coordinates are in image pixel space.

## Notes

- Class names are loaded from `data.yaml` (`names` field).
- If class names cannot be loaded, fallback class names are used in code.
- Detections are sorted descending by confidence.

#!/usr/bin/env python3
"""
HTTP API for civic issue detection.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Then expose it with:
    ngrok http 8000
"""

import io
import os
from typing import Any

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")
DATA_YAML_PATH = os.path.join(BASE_DIR, "data.yaml")
DEFAULT_CONFIDENCE = 0.25

app = FastAPI(title="Nivaari Civic Issue Detection API")

model: YOLO | None = None
class_names: list[str] = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_class_names() -> list[str]:
    fallback_names = [
        "Damaged_Concrete_Structures",
        "Damaged_Electric_Poles",
        "Damaged_Road_Signs",
        "Dead_Animal_Pollution",
        "Fallen_Trees",
        "Garbage",
        "Graffiti",
        "pothole",
        "road_crack",
    ]

    try:
        with open(DATA_YAML_PATH, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        names = data.get("names", fallback_names)
        return list(names)
    except Exception:
        return fallback_names


@app.on_event("startup")
def startup() -> None:
    global model, class_names

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    class_names = load_class_names()
    model = YOLO(MODEL_PATH)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Nivaari object detection API is running",
        "predict_endpoint": "/predict",
        "model_loaded": model is not None,
        "device": device,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_path": MODEL_PATH,
        "device": device,
        "classes": class_names,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    confidence: float = Query(DEFAULT_CONFIDENCE, ge=0.0, le=1.0),
) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    results = model(pil_image, conf=confidence, device=device, verbose=False)

    detections: list[dict[str, Any]] = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            score = float(box.conf[0].cpu().numpy())
            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"Class {class_id}"
            )

            detections.append(
                {
                    "category": class_name,
                    "class_id": class_id,
                    "confidence": score,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

    detections.sort(key=lambda detection: detection["confidence"], reverse=True)
    best_detection = detections[0] if detections else None

    return {
        "category": best_detection["category"] if best_detection else "No issue detected",
        "confidence": best_detection["confidence"] if best_detection else 0.0,
        "detections": detections,
        "total_detections": len(detections),
        "filename": image.filename,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)

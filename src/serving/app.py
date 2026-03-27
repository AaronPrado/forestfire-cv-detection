from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from ultralytics import YOLO

from src.utils.config import config

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = YOLO(config["training"]["best_model"])
    yield


app = FastAPI(title="Wildfire Fire & Smoke Detection API", lifespan=lifespan)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_SIZE = 10 * 1024 * 1024  # 10MB


@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {ext}")

    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="Archivo demasiado grande (máx 10MB)")

    np_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    results = model.predict(img, conf=config["serving"]["confidence_threshold"])

    detections = []
    for box in results[0].boxes:
        detections.append(
            {
                "class": int(box.cls[0]),
                "class_name": results[0].names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            }
        )

    return {
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model": config["training"]["best_model"]}

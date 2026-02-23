from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from src.utils.config import config
import numpy as np
import cv2

app = FastAPI(title="Wildfire Smoke Detection API")
model = YOLO(config["training"]["best_model"])

@app.post("/predict")
async def predict(file: UploadFile):
    contents =  await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    results = model.predict(img, conf=config["serving"]["confidence_threshold"])

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls[0]),
            "class_name": results[0].names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(), 
        })

    return {
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections,
    }
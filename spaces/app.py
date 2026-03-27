import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("best.pt")


def predict(image: Image.Image) -> tuple[np.ndarray, str]:
    results = model.predict(image, conf=0.25)
    detections = results[0].boxes
    text = f"{len(detections)} detecciones encontradas"
    return results[0].plot(), text


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy"), gr.Text()],
    title="Wildfire Fire & Smoke Detection",
    description="Sube una imagen para detectar fuego y humo de incendios forestales",
)

demo.launch()

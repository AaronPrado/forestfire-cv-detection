from fastapi.testclient import TestClient
from src.serving.app import app
from pathlib import Path

client = TestClient(app)


def test_predict_returns_200():
    test_images = list(Path("data/raw/test/images").glob("*.jpg"))
    assert len(test_images) > 0, "No hay imágenes de test"

    with open(test_images[0], "rb") as f:
        response = client.post("/predict", files={"file": f})

    assert response.status_code == 200


def test_predict_response_structure():
    test_images = list(Path("data/raw/test/images").glob("*.jpg"))

    with open(test_images[0], "rb") as f:
        response = client.post("/predict", files={"file": f})

    data = response.json()
    assert "filename" in data
    assert "detections_count" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_predict_detection_structure():
    test_images = list(Path("data/raw/test/images").glob("*.jpg"))

    with open(test_images[0], "rb") as f:
        response = client.post("/predict", files={"file": f})

    data = response.json()
    if data["detections_count"] > 0:
        detection = data["detections"][0]
        assert "class" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert "bbox" in detection
        assert len(detection["bbox"]) == 4

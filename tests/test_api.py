from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_predict_returns_200(client):
    test_images = list(Path("data/raw/test/images").glob("*.jpg"))
    assert len(test_images) > 0, "No hay imágenes de test"

    with open(test_images[0], "rb") as f:
        response = client.post("/predict", files={"file": f})

    assert response.status_code == 200


def test_predict_response_structure(client):
    test_images = list(Path("data/raw/test/images").glob("*.jpg"))

    with open(test_images[0], "rb") as f:
        response = client.post("/predict", files={"file": f})

    data = response.json()
    assert "filename" in data
    assert "detections_count" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_predict_detection_structure(client):
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


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_structure(client):
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model" in data


def test_predict_invalid_extension(client):
    response = client.post("/predict", files={"file": ("test.gif", b"data", "image/gif")})
    assert response.status_code == 400


def test_predict_file_too_large(client):
    big_file = b"x" * (11 * 1024 * 1024)  # 11MB
    response = client.post("/predict", files={"file": ("test.jpg", big_file, "image/jpeg")})
    assert response.status_code == 400

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.processing.resize import resize_dataset, resize_image


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Crea una estructura de dataset mínima con un split train/valid/test."""
    for split in ["train", "valid", "test"]:
        images_dir = tmp_path / split / "images"
        labels_dir = tmp_path / split / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "img.jpg"), img)
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    return tmp_path


def test_resize_image_dimensions(sample_image: Path, tmp_path: Path):
    output = tmp_path / "output.jpg"
    resize_image(sample_image, output, 640)
    result = cv2.imread(str(output))
    assert result.shape[:2] == (640, 640)


def test_resize_image_creates_output(sample_image: Path, tmp_path: Path):
    output = tmp_path / "output.jpg"
    resize_image(sample_image, output, 640)
    assert output.exists()


def test_resize_dataset_copies_labels(sample_dataset: Path, tmp_path: Path):
    processed = tmp_path / "processed"
    resize_dataset(sample_dataset, processed, 640)
    assert (processed / "train" / "labels" / "img.txt").exists()


def test_resize_dataset_creates_splits(sample_dataset: Path, tmp_path: Path):
    processed = tmp_path / "processed"
    resize_dataset(sample_dataset, processed, 640)
    for split in ["train", "valid", "test"]:
        assert (processed / split / "images" / "img.jpg").exists()

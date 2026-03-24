from pathlib import Path

import cv2
import numpy as np
import pytest

from src.processing.validate import validate_image


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_label(tmp_path: Path) -> Path:
    label_path = tmp_path / "test.txt"
    label_path.write_text("0 0.5 0.5 0.3 0.4\n")
    return label_path


def test_validate_image_valid(sample_image: Path, sample_label: Path):
    assert validate_image(sample_image, sample_label) is True


def test_validate_image_corrupt(tmp_path: Path, sample_label: Path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("esto no es una imagen")
    assert validate_image(corrupt, sample_label) is False


def test_validate_image_missing_label(sample_image: Path, tmp_path: Path):
    missing = tmp_path / "noexiste.txt"
    assert validate_image(sample_image, missing) is False


def test_validate_image_bad_label_format(sample_image: Path, tmp_path: Path):
    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5\n")  # Solo 3 valores, faltan 2
    assert validate_image(sample_image, label) is False


def test_validate_image_out_of_range_coords(sample_image: Path, tmp_path: Path):
    label = tmp_path / "outofrange.txt"
    label.write_text("0 1.5 0.5 0.3 0.4\n")  # Centro > 1.0
    assert validate_image(sample_image, label) is False


def test_validate_image_negative_class(sample_image: Path, tmp_path: Path):
    label = tmp_path / "negclass.txt"
    label.write_text("-1 0.5 0.5 0.3 0.4\n")
    assert validate_image(sample_image, label) is False


def test_validate_image_empty_label(sample_image: Path, tmp_path: Path):
    label = tmp_path / "empty.txt"
    label.write_text("")  # Label vacío = imagen negativa (clase null)
    assert validate_image(sample_image, label) is True

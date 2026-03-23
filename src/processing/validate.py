from pathlib import Path

import cv2
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_image(image_path, label_path):

    # Verificar que la imagen se puede abrir
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Imagen corrupta o no existe: %s", image_path)
        return False

    # Verificar que el label existe
    if not label_path.exists():
        logger.warning("Label no encontrado: %s", label_path)
        return False

    # Verificar el formato del label
    with open(label_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()

        if len(parts) != 5:
            logger.warning(
                "Label %s línea %d: esperaba 5 valores, tiene %d", label_path, i, len(parts)
            )
            return False

        try:
            cls = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            logger.warning("Label %s línea %d: valores no numéricos", label_path, i)
            return False

        if cls < 0:
            logger.warning("Label %s línea %d: clase negativa (%d)", label_path, i, cls)
            return False

        for j, coord in enumerate(coords):
            if coord < 0.0 or coord > 1.0:
                logger.warning(
                    "Label %s línea %d: coordenada %d fuera de rango (%f)", label_path, i, j, coord
                )
                return False

    return True


def validate_dataset(dataset_path):
    """Valida todo el dataset y devuelve el resumen"""
    dataset_path = Path(dataset_path)
    valid_count = 0
    invalid_count = 0

    for split in ["train", "valid", "test"]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        for img_file in images_dir.iterdir():
            if img_file.is_file():
                label_file = labels_dir / f"{img_file.stem}.txt"
                if validate_image(img_file, label_file):
                    valid_count += 1
                else:
                    invalid_count += 1

    logger.info(
        "Resultado de validación — Válidas: %d | Inválidas: %d | Total: %d",
        valid_count,
        invalid_count,
        valid_count + invalid_count,
    )

    return invalid_count == 0


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    validate_dataset(config["data"]["raw_dir"])

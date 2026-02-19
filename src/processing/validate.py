import cv2
from pathlib import Path
import yaml


def validate_image(image_path, label_path):

    # Verificar que la imagen se puede abrir
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Imagen corrupta o no existe: {image_path}")
        return False

    # Verificar que el label existe
    if not label_path.exists():
        print(f"Label no encontrado: {label_path}")
        return False

    # Verificar el formato del label
    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()

        if len(parts) != 5:
            print(f"Label {label_path} línea {i}: esperaba 5 valores, tiene {len(parts)}")
            return False

        try:
            cls = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            print(f"Label {label_path} línea {i}: valores no numéricos")
            return False

        if cls < 0:
            print(f"Label {label_path} línea {i}: clase negativa ({cls})")
            return False

        for j, coord in enumerate(coords):
            if coord < 0.0 or coord > 1.0:
                print(f"Label {label_path} línea {i}: coordenada {j} fuera de rango ({coord})")
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

    print(f"\nResultado de validación:")
    print(f"  Válidas: {valid_count}")
    print(f"  Inválidas: {invalid_count}")
    print(f"  Total: {valid_count + invalid_count}")

    return invalid_count == 0


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    validate_dataset(config["data"]["raw_dir"])

    

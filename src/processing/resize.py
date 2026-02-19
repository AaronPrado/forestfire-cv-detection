import cv2
import shutil
from pathlib import Path
import yaml


def resize_image(image_path, output_path, size):
    """Redimensiona una imagen y copia el label."""
    img = cv2.imread(str(image_path))
    img_resized = cv2.resize(img, (size, size))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), img_resized)


def resize_dataset(raw_dir, processed_dir, size):
    """Redimensiona todas las imágenes y copia los labels."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    img_count = 0

    for split in ["train", "valid", "test"]:
        images_dir = raw_path / split / "images"
        labels_dir = raw_path / split / "labels"

        for img_file in images_dir.iterdir():
            if img_file.is_file():
                # Ruta de destino de la imagen en data/processed/
                output_img = processed_path / split / "images" / img_file.name
                resize_image(img_file, output_img, size)

                # Copiar label al directorio procesado
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    output_label = processed_path / split / "labels" / label_file.name
                    output_label.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(label_file), str(output_label))

                img_count += 1

    print(f"Imágenes redimensionadas: {img_count}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    resize_dataset(config["data"]["raw_dir"], config["data"]["processed_dir"], config["data"]["image_size"])

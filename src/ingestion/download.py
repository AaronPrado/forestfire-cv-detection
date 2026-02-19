import pandas as pd
from pathlib import Path
from roboflow import Roboflow
from src.utils.config import config, ROBOFLOW_API_KEY
from src.utils.s3 import upload_to_s3, count_s3_objects


def download_dataset():
    """Descarga el dataset de Roboflow si no existe localmente."""
    raw_dir = Path(config["data"]["raw_dir"])

    if (raw_dir / "train" / "images").exists():
        print("Dataset ya existe localmente. Saltando descarga.")
        return

    print("Descargando dataset de Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(config["data"]["roboflow_workspace"]).project(config["data"]["roboflow_project"])
    version = project.version(config["data"]["roboflow_version"])
    version.download(model_format=config["data"]["format"], location=str(raw_dir))
    print("Dataset descargado.")


def upload_raw_to_s3():
    """Sube el dataset raw a S3 si no está subido."""
    bucket = config["aws"]["bucket"]
    prefix = config["aws"]["raw_prefix"]

    count = count_s3_objects(bucket, prefix)
    if count > 0:
        print(f"S3 ya tiene {count} archivos en {prefix}. Saltando subida.")
        return

    print("Subiendo dataset a S3...")
    total = upload_to_s3(config["data"]["raw_dir"], bucket, prefix)
    print(f"Archivos subidos: {total}")


def generate_metadata():
    """Genera un archivo Parquet con metadatos del dataset."""
    metadata_path = Path("data/metadata.parquet")

    if metadata_path.exists():
        print("Metadatos ya existen. Saltando generación.")
        return

    print("Generando metadatos...")
    local_path = Path(config["data"]["raw_dir"])
    bucket = config["aws"]["bucket"]
    prefix = config["aws"]["raw_prefix"]
    records = []

    for split in ["train", "valid", "test"]:
        images_dir = local_path / split / "images"
        if not images_dir.exists():
            continue

        for img_file in images_dir.iterdir():
            if img_file.is_file():
                label_file = local_path / split / "labels" / f"{img_file.stem}.txt"
                records.append({
                    "filename": img_file.name,
                    "split": split,
                    "s3_image_path": f"s3://{bucket}/{prefix}/{split}/images/{img_file.name}",
                    "s3_label_path": f"s3://{bucket}/{prefix}/{split}/labels/{img_file.stem}.txt",
                    "has_label": label_file.exists(),
                })

    df = pd.DataFrame(records)
    df.to_parquet(str(metadata_path), index=False)
    print(f"Metadatos generados: {len(df)} imágenes")
    print(f"\nResumen por split:")
    print(df["split"].value_counts())
    print(f"\nImágenes con label: {df['has_label'].sum()}/{len(df)}")


if __name__ == "__main__":
    download_dataset()
    upload_raw_to_s3()
    generate_metadata()

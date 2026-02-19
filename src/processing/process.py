import sys
from src.utils.config import config
from src.utils.s3 import upload_to_s3, count_s3_objects
from src.processing.validate import validate_dataset
from src.processing.resize import resize_dataset


def run_processing():
    """Orquesta el pipeline de procesamiento: validar, redimensionar, subir."""

    print("=== Paso 1: Validación ===")
    is_valid = validate_dataset(config["data"]["raw_dir"])

    if not is_valid:
        print("Dataset inválido. Corrige los errores antes de continuar.")
        sys.exit(1)

    print("\n=== Paso 2: Redimensionado ===")
    resize_dataset(
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["data"]["image_size"],
    )

    print("\n=== Paso 3: Subida a S3 ===")
    bucket = config["aws"]["bucket"]
    prefix = config["aws"]["processed_prefix"]

    count = count_s3_objects(bucket, prefix)
    if count > 0:
        print(f"S3 ya tiene {count} archivos en {prefix}. Saltando subida.")
        return

    total = upload_to_s3(config["data"]["processed_dir"], bucket, prefix)
    print(f"Archivos subidos: {total}")


if __name__ == "__main__":
    run_processing()

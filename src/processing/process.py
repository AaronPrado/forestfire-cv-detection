import sys

from src.processing.resize import resize_dataset
from src.processing.validate import validate_dataset
from src.utils.config import config
from src.utils.logging import get_logger
from src.utils.s3 import count_s3_objects, upload_to_s3

logger = get_logger(__name__)


def run_processing():
    """Orquesta el pipeline de procesamiento: validar, redimensionar, subir."""

    logger.info("=== Paso 1: Validación ===")
    is_valid = validate_dataset(config["data"]["raw_dir"])

    if not is_valid:
        logger.error("Dataset inválido. Corrige los errores antes de continuar.")
        sys.exit(1)

    logger.info("=== Paso 2: Redimensionado ===")
    resize_dataset(
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["data"]["image_size"],
    )

    logger.info("=== Paso 3: Subida a S3 ===")
    bucket = config["aws"]["bucket"]
    prefix = config["aws"]["processed_prefix"]

    count = count_s3_objects(bucket, prefix)
    if count > 0:
        logger.info("S3 ya tiene %d archivos en %s. Saltando subida.", count, prefix)
        return

    total = upload_to_s3(config["data"]["processed_dir"], bucket, prefix)
    logger.info("Archivos subidos: %d", total)


if __name__ == "__main__":
    run_processing()

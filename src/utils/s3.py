import boto3
from pathlib import Path
from src.utils.config import config, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

s3_client = boto3.client(
    "s3",
    region_name=config["aws"]["region"],
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def upload_to_s3(local_dir, bucket, s3_prefix):
    """Sube todos los archivos de un directorio local a S3."""
    local_path = Path(local_dir)
    uploaded = 0

    for file in local_path.rglob("*"):
        if file.is_file():
            s3_key = f"{s3_prefix}/{file.relative_to(local_path)}"
            s3_client.upload_file(str(file), bucket, s3_key)
            uploaded += 1

    return uploaded


def count_s3_objects(bucket, prefix):
    """Cuenta los archivos en un prefijo de S3."""
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        count += page.get("KeyCount", 0)
    return count

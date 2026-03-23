import click


@click.group()
@click.version_option(version="1.0.0")
def smoke():
    """Wildfire Smoke Detection — MLOps Pipeline CLI."""
    pass


@smoke.command()
def ingest():
    """Download dataset from Roboflow and upload to S3."""
    from src.ingestion.download import download_dataset, generate_metadata, upload_raw_to_s3

    download_dataset()
    upload_raw_to_s3()
    generate_metadata()


@smoke.command()
def process():
    """Validate, resize, and upload processed data to S3."""
    from src.processing.process import run_processing

    run_processing()


@smoke.command()
@click.option("--epochs", default=None, type=int, help="Override number of training epochs.")
@click.option("--batch-size", default=None, type=int, help="Override batch size.")
def train(epochs, batch_size):
    """Train YOLOv8 model with MLflow tracking."""
    from src.training.train import train as run_train

    run_train(epochs_override=epochs, batch_size_override=batch_size)


@smoke.command()
@click.option("--host", default="0.0.0.0", help="API host.")
@click.option("--port", default=8000, type=int, help="API port.")
def serve(host, port):
    """Start the FastAPI inference server."""
    import uvicorn

    uvicorn.run("src.serving.app:app", host=host, port=port)


@smoke.command()
@click.argument("image_path", type=click.Path(exists=True))
def predict(image_path):
    """Run inference on a single image."""
    from ultralytics import YOLO

    from src.utils.config import config

    model = YOLO(config["training"]["best_model"])
    results = model.predict(image_path, conf=config["serving"]["confidence_threshold"], save=True)
    for box in results[0].boxes:
        name = results[0].names[int(box.cls[0])]
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        click.echo(
            f"  {name}: {conf:.3f} at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
        )


@smoke.command()
def pipeline():
    """Run full pipeline: ingest -> process -> train."""
    from src.ingestion.download import download_dataset, generate_metadata, upload_raw_to_s3
    from src.processing.process import run_processing
    from src.training.train import train as run_train

    download_dataset()
    upload_raw_to_s3()
    generate_metadata()
    run_processing()
    run_train()


@smoke.command()
@click.argument("version", type=int)
@click.argument("stage", type=click.Choice(["Staging", "Production", "Archived"]))
def promote(version, stage):
    """Promote a model version to a stage."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.utils.config import config

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()
    client.transition_model_version_stage(
        name="wildfire-smoke-yolov8s",
        version=version,
        stage=stage,
    )
    click.echo(f"Modelo versión {version} → {stage}")


if __name__ == "__main__":
    smoke()

import mlflow
import torch
from ultralytics import YOLO

from src.utils.config import config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def train(epochs_override: int | None = None, batch_size_override: int | None = None):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    model = YOLO(config["training"]["model"])

    with mlflow.start_run(run_name=config["mlflow"]["model_name"]) as run:
        # Registrar parámetros
        mlflow.log_param("model", config["training"]["model"])
        mlflow.log_param("epochs", config["training"]["epochs"])
        mlflow.log_param("batch_size", config["training"]["batch_size"])
        mlflow.log_param("imgsz", config["training"]["imgsz"])

        # Entrenamiento
        results = model.train(
            data=config["data"]["data_yaml"],
            epochs=epochs_override or config["training"]["epochs"],
            batch=batch_size_override or config["training"]["batch_size"],
            imgsz=config["training"]["imgsz"],
            patience=config["training"]["patience"],
            device=0 if torch.cuda.is_available() else "cpu",
            project="runs/train",
            name="fire-smoke-v",
            workers=0,
        )

        # Registrar métricas
        metrics = results.results_dict
        mlflow.log_metric("mAP50", metrics["metrics/mAP50(B)"])
        mlflow.log_metric("mAP50-95", metrics["metrics/mAP50-95(B)"])
        mlflow.log_metric("precision", metrics["metrics/precision(B)"])
        mlflow.log_metric("recall", metrics["metrics/recall(B)"])

        # Guardar el modelo entrenado como artefacto
        mlflow.log_artifact(str(results.save_dir / "weights" / "best.pt"))

        # Model Registry
        artifact_uri = f"{run.info.artifact_uri}/best.pt"
        mv = mlflow.register_model(artifact_uri, config["mlflow"]["model_name"])
        logger.info("Modelo registrado: versión %s", mv.version)


if __name__ == "__main__":
    train()

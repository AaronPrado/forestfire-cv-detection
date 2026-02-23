from ultralytics import YOLO
import mlflow
from src.utils.config import config


def train():
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    model = YOLO(config["training"]["model"])

    with mlflow.start_run(run_name="wildfire-smoke-detection-initial"):
        # Registrar parámetros
        mlflow.log_param("model", config["training"]["model"])
        mlflow.log_param("epochs", config["training"]["epochs"])
        mlflow.log_param("batch_size", config["training"]["batch_size"])
        mlflow.log_param("imgsz", config["training"]["imgsz"])

        # Entrenamiento
        results = model.train(
            data="data/processed/data.yaml",       # YAML del dataset
            epochs=config["training"]["epochs"],
            batch=config["training"]["batch_size"],
            imgsz=config["training"]["imgsz"],
            patience=config["training"]["patience"],
            device=0,
            project="runs/train",
            name="smoke-v",
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


if __name__ == "__main__":
    train()
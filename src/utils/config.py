import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Credenciales desde variables de entorno
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


config = load_config()

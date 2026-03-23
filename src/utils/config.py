import os

import yaml
from dotenv import load_dotenv

load_dotenv()

# Credenciales desde variables de entorno
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Configuración del proyecto
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Wildfire Smoke Detection

[![CI](https://github.com/AaronPrado/forestfire-cv-detection/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AaronPrado/forestfire-cv-detection/actions/workflows/ci.yml)

Pipeline end-to-end de Computer Vision para detectar humo de incendios forestales en imágenes usando YOLOv8.

## Descripción

Este proyecto implementa un pipeline MLOps completo: ingesta de datos desde Roboflow, procesamiento de imágenes, entrenamiento del modelo con tracking de experimentos (MLflow), registro de modelos, inferencia en tiempo real mediante una API REST y una demo pública interactiva.

Proyecto complementario: [fire-risk-pipeline](https://github.com/AaronPrado/fire-risk-pipeline) — Pipeline de predicción de riesgo de incendios forestales.

Demo pública: [huggingface.co/spaces/AaronPrado/wildfire-smoke-detection](https://huggingface.co/spaces/AaronPrado/wildfire-smoke-detection)

## Estructura del proyecto

```
forestfire-cv-detection/
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD con GitHub Actions
├── configs/
│   └── config.yaml              # Configuración centralizada del proyecto
├── data/
│   └── .gitkeep
├── docker/
│   ├── Dockerfile               # Imagen Docker para la API
│   └── docker-compose.yml       # API + MLflow UI
├── spaces/
│   ├── app.py                   # Demo interactiva con Gradio (HF Spaces)
│   └── requirements.txt
├── src/
│   ├── cli.py                   # CLI unificado con Click
│   ├── ingestion/
│   │   └── download.py          # Descarga del dataset, subida a S3, generación de metadatos
│   ├── processing/
│   │   ├── validate.py          # Validación de imágenes y labels
│   │   ├── resize.py            # Redimensionado de imágenes
│   │   └── process.py           # Orquestador del pipeline de procesamiento
│   ├── training/
│   │   └── train.py             # Entrenamiento YOLOv8 con tracking y registro en MLflow
│   ├── serving/
│   │   └── app.py               # API REST con FastAPI
│   └── utils/
│       ├── config.py            # Carga centralizada de configuración y credenciales
│       ├── logging.py           # Logger estructurado
│       └── s3.py                # Funciones reutilizables de S3
├── tests/
│   ├── test_api.py              # Tests de los endpoints /predict y /health
│   ├── test_cli.py              # Tests del CLI
│   ├── test_config.py           # Tests de carga de configuración
│   ├── test_validate.py         # Tests de validación de imágenes y labels
│   └── test_resize.py           # Tests de redimensionado
├── .dockerignore
├── .env.example
├── .pre-commit-config.yaml      # Hooks de pre-commit (ruff, detect-private-key)
├── conftest.py
├── Makefile                     # Automatización de tareas
├── pyproject.toml               # Configuración del proyecto (ruff, pytest, dependencias)
├── requirements.txt
└── README.md
```

## Dataset

- **Fuente:** [Wildfire Smoke - Roboflow Universe](https://public.roboflow.com/object-detection/wildfire-smoke/1)
- **Imágenes:** 737 (train: 516, valid: 147, test: 74)
- **Clases:** 1 (smoke)
- **Formato:** YOLOv8 bounding boxes
- **Almacenamiento:** AWS S3

## Resultados del modelo

| Métrica | Valor |
|---------|-------|
| mAP50 | 0.972 |
| mAP50-95 | 0.550 |
| Precision | 0.927 |
| Recall | 0.946 |

Modelo: YOLOv8s entrenado durante 150 epochs sobre 516 imágenes.

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/AaronPrado/forestfire-cv-detection.git
cd forestfire-cv-detection
```

### 2. Crear entorno virtual

```bash
conda create -n wildfire python=3.11
conda activate wildfire
```

### 3. Instalar dependencias

```bash
make install
```

### 4. Configurar credenciales

Copia `.env.example` a `.env` y rellena tus credenciales:

```bash
cp .env.example .env
```

```
ROBOFLOW_API_KEY=tu_api_key_de_roboflow
AWS_ACCESS_KEY_ID=tu_access_key_de_aws
AWS_SECRET_ACCESS_KEY=tu_secret_key_de_aws
```

## Uso

### Ejecutar el pipeline completo

```bash
smoke ingest       # Descarga dataset de Roboflow y sube a S3
smoke process      # Valida, redimensiona y sube datos procesados a S3
smoke train        # Entrena YOLOv8s con tracking en MLflow

# O todo de una vez
smoke pipeline
```

### Opciones de entrenamiento

```bash
smoke train --epochs 50 --batch-size 8
```

### Exportar modelo a ONNX

```bash
smoke export
```

### Iniciar la API

```bash
smoke serve
```

### Hacer una predicción

```bash
# Desde CLI (sin servidor)
smoke predict imagen.jpg

# Desde la API REST
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@imagen.jpg"
```

Respuesta:

```json
{
  "filename": "imagen.jpg",
  "detections_count": 1,
  "detections": [
    {
      "class": 0,
      "class_name": "smoke",
      "confidence": 0.596,
      "bbox": [277.4, 261.1, 517.0, 317.7]
    }
  ]
}
```

### Gestión de modelos con MLflow

```bash
smoke promote 1 Production
```

### Docker

```bash
# Solo la API
make docker-build
make docker-run

# API + MLflow UI
docker compose -f docker/docker-compose.yml up
```

### Tests

```bash
make test
make test-cov
```

### Calidad de código

```bash
make lint
make format
make all        # format + lint + test
```

## Stack tecnológico

- **Modelo de detección:** YOLOv8s (Ultralytics)
- **Tracking y registro de modelos:** MLflow
- **Almacenamiento de datos:** AWS S3
- **Procesamiento de imágenes:** OpenCV
- **API:** FastAPI + Uvicorn
- **CLI:** Click
- **Demo pública:** Gradio + Hugging Face Spaces
- **Contenedorización:** Docker + Docker Compose
- **CI/CD:** GitHub Actions
- **Calidad de código:** Ruff, pre-commit
- **Testing:** pytest + pytest-cov
- **Configuración:** PyYAML, python-dotenv

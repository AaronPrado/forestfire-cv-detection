# Wildfire Smoke Detection

Pipeline end-to-end de Computer Vision para detectar humo de incendios forestales en imágenes usando YOLOv8.

## Descripción

Este proyecto implementa un pipeline MLOps completo: ingesta de datos desde Roboflow, procesamiento de imágenes, entrenamiento del modelo con tracking de experimentos (MLflow) e inferencia en tiempo real mediante una API REST.

## Estructura del proyecto

```
wildfire-smoke-detection/
├── configs/
│   └── config.yaml          # Configuración centralizada del proyecto
├── data/
│   └── .gitkeep
├── docker/
│   └── Dockerfile           # Imagen Docker para la API
├── src/
│   ├── ingestion/
│   │   └── download.py      # Descarga del dataset, subida a S3, generación de metadatos
│   ├── processing/
│   │   ├── validate.py      # Validación de imágenes y labels
│   │   ├── resize.py        # Redimensionado de imágenes
│   │   └── process.py       # Orquestador del pipeline de procesamiento
│   ├── training/
│   │   └── train.py         # Entrenamiento YOLOv8 con tracking en MLflow
│   ├── serving/
│   │   └── app.py           # API REST con FastAPI
│   └── utils/
│       ├── config.py        # Carga centralizada de configuración y credenciales
│       └── s3.py            # Funciones reutilizables de S3
├── tests/
│   └── test_api.py          # Tests del endpoint /predict
├── .env.example
├── conftest.py
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
pip install -r requirements.txt
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
# 1. Ingesta de datos
python -m src.ingestion.download

# 2. Procesamiento de datos
python -m src.processing.process

# 3. Entrenamiento
python -m src.training.train
```

### Iniciar la API

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### Hacer una predicción

```bash
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

### Docker

```bash
docker build -f docker/Dockerfile -t wildfire-api .
docker run --env-file .env -p 8000:8000 wildfire-api
```

### Tests

```bash
pytest tests/test_api.py -v
```

## Stack tecnológico

- **Modelo de detección:** YOLOv8s (Ultralytics)
- **Tracking de experimentos:** MLflow
- **Almacenamiento de datos:** AWS S3
- **Procesamiento de imágenes:** OpenCV
- **API:** FastAPI + Uvicorn
- **Contenedorización:** Docker
- **Testing:** pytest
- **Configuración:** PyYAML, python-dotenv

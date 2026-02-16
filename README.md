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
├── notebooks/
├── src/
│   ├── ingestion/
│   │   └── download.py      # Descarga del dataset, subida a S3, generación de metadatos
│   ├── processing/           # Validación, redimensionado y augmentation de imágenes
│   ├── training/             # Entrenamiento YOLOv8 con tracking en MLflow
│   └── serving/              # API de inferencia con FastAPI
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Dataset

- **Fuente:** [Wildfire Smoke - Roboflow Universe](https://public.roboflow.com/object-detection/wildfire-smoke/1)
- **Imágenes:** 737 (train: 516, valid: 147, test: 74)
- **Clases:** 1 (smoke)
- **Formato:** YOLOv8 bounding boxes
- **Almacenamiento:** AWS S3

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

### 5. Ejecutar ingesta de datos

```bash
python src/ingestion/download.py
```

Esto hace lo siguiente:
- Descarga el dataset desde Roboflow (si no existe localmente)
- Lo sube a S3 (si no está subido)
- Genera un archivo de metadatos en formato Parquet

## Stack tecnológico

- **Modelo de detección:** YOLOv8 (Ultralytics)
- **Tracking de experimentos:** MLflow
- **Almacenamiento de datos:** AWS S3
- **Procesamiento de imágenes:** OpenCV, Albumentations
- **API:** FastAPI + Uvicorn
- **Testing:** pytest

## Fases del pipeline

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1. Ingesta | Descarga del dataset, subida a S3, generación de metadatos | Hecho |
| 2. Procesamiento | Validación, redimensionado y augmentation de imágenes | Pendiente |
| 3. Entrenamiento | Entrenamiento de YOLOv8 con tracking en MLflow | Pendiente |
| 4. Serving | API REST para inferencia en tiempo real | Pendiente |
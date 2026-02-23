FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema que OpenCV necesita
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y configuración
COPY src/ src/
COPY configs/ configs/

# Copiar modelo entrenado
COPY models/best.pt models/best.pt

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

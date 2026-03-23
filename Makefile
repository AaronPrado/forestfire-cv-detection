.PHONY: install lint format test test-cov ingest process train serve pipeline docker-build docker-run clean all


install:
		pip install -e ".[dev]"

lint:
		ruff check src/ tests/

format:
		ruff format src/ tests/ && ruff check --fix src/ tests/

test:
		pytest tests/

test-cov:
		pytest --cov=src tests/

ingest:
		python -m src.ingestion.download

process:
		python -m src.processing.process

train:
		python -m src.training.train

serve:
		uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

pipeline: ingest process train

docker-build:
		docker build -f docker/Dockerfile -t wildfire-api .

docker-run:
		docker run --env-file .env -p 8000:8000 wildfire-api

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

all: format lint test

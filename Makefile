.PHONY: help venv install data prepare train api test lint dvc-repro mlflow up down drift retrain clean

help:
	@echo "Targets:"
	@echo "  venv        Create virtual environment"
	@echo "  install     Install python dependencies"
	@echo "  data        Download raw dataset"
	@echo "  prepare     Run DVC prepare stage"
	@echo "  train       Run DVC train stage"
	@echo "  dvc-repro   Reproduce full DVC pipeline"
	@echo "  mlflow      Start MLflow tracking server"
	@echo "  api         Run FastAPI locally"
	@echo "  drift       Run Evidently drift report"
	@echo "  retrain     Run Prefect retraining flow"
	@echo "  test        Run test suite"
	@echo "  up          docker compose up --build"
	@echo "  down        docker compose down -v"
	@echo "  clean       Remove caches and build artifacts"

venv:
	python -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

data:
	python -m src.data.download

prepare:
	dvc repro prepare

train:
	dvc repro train

dvc-repro:
	dvc repro

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

drift:
	python -m src.monitoring.drift

retrain:
	python -m flows.retrain_flow

test:
	pytest -q

up:
	docker compose up --build -d

down:
	docker compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov

# MLOps Customer Churn Prediction

A complete end-to-end MLOps pipeline for predicting customer churn using the Telco Customer Churn dataset. This project demonstrates production-ready ML practices including data versioning, experiment tracking, model registry, API serving, monitoring, and automated retraining.

## Architecture

```
Data → DVC → Train → MLflow Registry → FastAPI → Prometheus → Grafana
                          ↑
                    Prefect Retrain ← Evidently Drift
```

## Features

- **Data Versioning**: DVC for dataset and model versioning
- **Experiment Tracking**: MLflow for tracking experiments, metrics, and model registry
- **Model Training**: Scikit-learn RandomForest with preprocessing pipeline
- **Model Serving**: FastAPI prediction API with Prometheus metrics
- **Monitoring**: Evidently AI for data drift detection
- **Orchestration**: Prefect for automated retraining flows
- **Containerization**: Docker + Docker Compose for all services
- **CI/CD**: GitHub Actions for testing and validation

## Project Structure

```
MLOps-Customer-Churn-Prediction-/
├── .github/workflows/ci.yml          # GitHub Actions CI pipeline
├── api/
│   └── main.py                       # FastAPI prediction service
├── data/
│   ├── raw/                          # Raw data (DVC-tracked)
│   │   ├── Telco-Customer-Churn.csv
│   │   └── Telco-Customer-Churn.csv.dvc
│   └── processed/                    # Processed train/test splits
│       ├── train.csv
│       └── test.csv
├── flows/
│   └── retrain_flow.py               # Prefect retraining flow
├── monitoring/
│   └── drift.py                      # Evidently drift detection
├── src/
│   ├── data.py                       # Data download & preparation
│   ├── train.py                      # Model training with MLflow
│   └── promote.py                    # Promote model to Production
├── tests/
│   ├── test_api.py                   # API endpoint tests
│   └── test_data.py                  # Data preparation tests
├── docker/
│   ├── prometheus.yml                # Prometheus config
│   └── grafana/                      # Grafana dashboards & datasources
├── .dvc/                             # DVC configuration
├── .dvcignore                        # DVC ignore patterns
├── .env.example                      # Environment variable template
├── .gitignore                        # Git ignore patterns
├── Dockerfile                        # API container image
├── docker-compose.yml                # Multi-service orchestration
├── dvc.yaml                          # DVC pipeline stages
├── Makefile                          # Common commands
├── params.yaml                       # Model hyperparameters
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## File Descriptions

### Source Code (`src/`)

| File | Purpose |
|------|---------|
| `data.py` | Downloads Telco Churn dataset from GitHub, cleans data (drops `customerID`, handles `TotalCharges`), creates stratified train/test split (80/20), saves to `data/processed/` |
| `train.py` | Loads processed data, builds sklearn Pipeline (imputer + scaler + OneHotEncoder + RandomForest), trains with MLflow tracking, logs params/metrics/model, registers model as `churn_model`, saves `models/model.pkl` |
| `promote.py` | Uses MLflow Client to transition latest `churn_model` version from `None` stage to `Production`, archiving existing versions |

### API (`api/main.py`)

FastAPI service with:
- `GET /health` — Health check with model load status
- `POST /predict` — Accepts feature dict, returns prediction (0/1) and probability
- `GET /metrics` — Prometheus metrics endpoint (`predictions_total`, `prediction_latency_seconds`)
- Loads model from MLflow Model Registry at startup (`models:/churn_model/Production`)
- Logs predictions to `logs/predictions.csv`

### Configuration

| File | Purpose |
|------|---------|
| `params.yaml` | Model hyperparameters (`n_estimators`, `max_depth`) |
| `dvc.yaml` | DVC pipeline: `prepare` stage (data.py) → `train` stage (train.py) |
| `requirements.txt` | 17 pinned dependencies (pandas, sklearn, mlflow, fastapi, dvc, prefect, evidently, etc.) |
| `.env.example` | Template for MLflow URI, model name/stage, Prefect API URL |
| `docker-compose.yml` | Services: mlflow (port 5000), api (port 8000), prometheus (9090), grafana (3000) |

### Monitoring & Orchestration

| File | Purpose |
|------|---------|
| `monitoring/drift.py` | Evidently drift report comparing reference (train) vs current data |
| `flows/retrain_flow.py` | Prefect flow: checks drift → retrains if needed → promotes new model |

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### 1. Clone & Setup

```bash
git clone <repo-url>
cd MLOps-Customer-Churn-Prediction-

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize DVC

```bash
dvc init
dvc remote add -d localremote /tmp/dvcstore
```

### 3. Data Preparation

```bash
# Download data and create train/test splits
python -m src.data

# Track raw data with DVC
dvc add data/raw/Telco-Customer-Churn.csv
git add .dvc/.gitignore data/raw/Telco-Customer-Churn.csv.dvc
dvc push
```

### 4. Start MLflow (via Docker)

```bash
docker compose up -d mlflow
# Wait ~10 seconds for MLflow to start
```

### 5. Train Model

```bash
# Train with MLflow tracking
python -m src.train

# Promote latest model to Production
python -m src.promote
```

Open MLflow UI at **http://localhost:5000** to view experiments and registered model.

### 6. Run Prediction API

```bash
# Option A: Local
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Option B: Docker (includes MLflow, Prometheus, Grafana)
docker compose up -d
```

API available at **http://localhost:8000**
- Health: `GET http://localhost:8000/health`
- Predict: `POST http://localhost:8000/predict` with JSON body
- Metrics: `GET http://localhost:8000/metrics`

### 7. Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
  }'
```

### 8. Run Drift Detection

```bash
python -m monitoring.drift
# Generates reports/drift_report.html
```

### 9. Run Prefect Retraining Flow

```bash
# Start Prefect server (separate terminal)
prefect server start

# Run flow
python -m flows.retrain_flow
```

## Using Makefile

Common commands wrapped in `Makefile`:

```bash
make help          # Show all targets
make venv          # Create virtual environment
make install       # Install dependencies
make data          # Run data preparation
make prepare       # DVC repro prepare stage
make train         # DVC repro train stage
make dvc-repro     # Run full DVC pipeline
make mlflow        # Start MLflow via docker compose
make api           # Start API locally
make drift         # Run drift detection
make retrain       # Run Prefect retrain flow
make test          # Run pytest
make up            # Docker compose up
make down          # Docker compose down
make clean         # Clean artifacts
```

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server |
| `MLFLOW_EXPERIMENT_NAME` | `customer-churn` | MLflow experiment name |
| `MLFLOW_REGISTERED_MODEL_NAME` | `churn-randomforest` | Registered model name |
| `MODEL_URI` | `models:/churn-randomforest/Production` | Model URI for API |
| `MODEL_PATH` | `models/model.pkl` | Local model path fallback |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API port |
| `PREFECT_API_URL` | `http://localhost:4200/api` | Prefect API URL |

## DVC Pipeline

```bash
# Reproduce full pipeline
dvc repro

# Reproduce specific stage
dvc repro prepare
dvc repro train

# View pipeline
dvc dag
```

Stages:
1. **prepare** — `python -m src.data` → outputs `data/processed/train.csv`, `data/processed/test.csv`
2. **train** — `python -m src.train` → outputs `models/model.pkl`, uses params `model.n_estimators`, `model.max_depth`

## MLflow Model Registry

Models are registered as `churn_model` with stages:
- `None` — Newly registered versions
- `Staging` — Candidate for production
- `Production` — Currently serving model (promoted via `promote.py`)

API loads from `models:/churn_model/Production`.

## Monitoring

- **Prometheus**: Scrapes `/metrics` from API (port 9090)
- **Grafana**: Dashboards at port 3000 (admin/admin)
- **Evidently**: Drift report at `reports/drift_report.html`

## Testing

```bash
pytest -q
```

Tests cover:
- Data preparation output shapes
- API health and prediction endpoints

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push/PR:
- Lint & format checks
- Unit tests
- DVC pipeline validation
- Docker build verification

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `dvc add` fails with git-ignored | Ensure `.gitignore` has `!/data/raw/*.dvc` exceptions |
| MLflow connection refused | Wait for container to fully start; check `docker compose logs mlflow` |
| Model not found in registry | Run `python -m src.promote` after training |
| Import errors in venv | Reinstall: `pip install -r requirements.txt` |
| DVC pathspec error | `pip install --upgrade pathspec dvc` |

## License

MIT License — feel free to use and modify for your own projects.
import os
import csv
import time
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import mlflow.sklearn

app = FastAPI(title="Churn Prediction API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
LOG_FILE = Path("logs/predictions.csv")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

model = None

PREDICTIONS = Counter("predictions_total", "Total predictions", ["result"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model from {model_uri}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: dict):
    start = time.time()
    df = pd.DataFrame([features])
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])

    LATENCY.observe(time.time() - start)
    PREDICTIONS.labels(result=str(pred)).inc()

    row = features.copy()
    row["prediction"] = pred
    row["probability"] = proba

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    return {"prediction": pred, "probability": proba}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
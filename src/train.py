import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import yaml
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
params = yaml.safe_load(open(ROOT / "params.yaml"))
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn")

train = pd.read_csv(ROOT / "data/processed/train.csv")
test = pd.read_csv(ROOT / "data/processed/test.csv")

X_train = train.drop(columns=["Churn"])
y_train = train["Churn"].map({"No": 0, "Yes": 1})
X_test = test.drop(columns=["Churn"])
y_test = test["Churn"].map({"No": 0, "Yes": 1})

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=42
    ))
])

with mlflow.start_run() as run:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba)
    }

    mlflow.log_params(params["model"])
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="churn_model"
    )
    print(metrics)

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models/model.pkl")
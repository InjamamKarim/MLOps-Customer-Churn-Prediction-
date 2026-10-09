import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
model_name = "churn_model"

latest = client.get_latest_versions(model_name, stages=["None"])[0]
client.transition_model_version_stage(
    name=model_name,
    version=latest.version,
    stage="Production",
    archive_existing_versions=True
)
print(f"Promoted version {latest.version} to Production")
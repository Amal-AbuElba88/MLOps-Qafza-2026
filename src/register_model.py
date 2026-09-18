"""Register the notebook-trained pipeline and metadata in MLflow."""

from __future__ import annotations

import json
import logging
import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.config import load_settings
from src.logging_config import setup_logging


logger = logging.getLogger(__name__)


def _wait_for_mlflow(tracking_uri: str, attempts: int = 30) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    for attempt in range(1, attempts + 1):
        try:
            MlflowClient().search_experiments(max_results=1)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2)


def register() -> str:
    settings = load_settings()
    setup_logging(settings)
    _wait_for_mlflow(settings.mlflow_tracking_uri)

    with settings.metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    model = joblib.load(settings.model_path)
    feature_columns = list(joblib.load(settings.feature_columns_path))
    sample = pd.read_csv(settings.reference_features_path, nrows=5)[feature_columns]
    signature = infer_signature(sample, model.predict(sample))

    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="register-notebook-selected-model") as run:
        mlflow.log_param("model_name", metadata["model_name"])
        mlflow.log_param("decision_threshold", metadata["decision_threshold"])
        mlflow.log_param("training_rows", metadata["training_rows"])
        for metric_name, metric_value in metadata.get("test_metrics", {}).items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
        mlflow.log_artifact(str(settings.metadata_path), artifact_path="metadata")
        mlflow.log_artifact(
            str(settings.feature_columns_path), artifact_path="metadata"
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=sample.head(1),
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    registered = mlflow.register_model(model_uri, settings.model_name)
    client = MlflowClient()
    client.set_model_version_tag(
        settings.model_name,
        registered.version,
        "decision_threshold",
        str(metadata["decision_threshold"]),
    )
    client.set_model_version_tag(
        settings.model_name,
        registered.version,
        "feature_columns",
        ",".join(feature_columns),
    )
    client.set_registered_model_alias(
        settings.model_name, settings.model_alias, registered.version
    )
    logger.info(
        "model_registered name=%s version=%s alias=%s",
        settings.model_name,
        registered.version,
        settings.model_alias,
    )
    return str(registered.version)


if __name__ == "__main__":
    register()

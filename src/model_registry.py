"""Model loading from MLflow Registry or a local test artifact."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import Settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    feature_columns: List[str]
    version: str
    threshold: float
    source: str


def _load_local_bundle(settings: Settings) -> ModelBundle:
    """Load notebook artifacts for tests and model registration only."""

    with settings.metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return ModelBundle(
        model=joblib.load(settings.model_path),
        feature_columns=list(joblib.load(settings.feature_columns_path)),
        version=str(metadata.get("model_version", "local")),
        threshold=float(metadata.get("decision_threshold", settings.threshold)),
        source="local_artifact",
    )


def load_registered_model(settings: Settings) -> ModelBundle:
    """Resolve an alias and load its exact registered MLflow model version."""

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        settings.model_name, settings.model_alias
    )
    model_uri = f"models:/{settings.model_name}/{version.version}"
    logger.info("loading_registered_model uri=%s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    threshold = float(version.tags.get("decision_threshold", settings.threshold))
    feature_columns = version.tags.get("feature_columns")
    columns = (
        feature_columns.split(",") if feature_columns else settings.feature_columns
    )
    return ModelBundle(
        model=model,
        feature_columns=list(columns),
        version=str(version.version),
        threshold=threshold,
        source=model_uri,
    )


def load_model_bundle(settings: Settings) -> ModelBundle:
    """Load from the configured source without silently changing provenance."""

    if settings.model_source == "mlflow":
        return load_registered_model(settings)
    if settings.model_source == "local":
        return _load_local_bundle(settings)
    raise ValueError(f"Unsupported model source: {settings.model_source}")

"""Central configuration loader with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    project_name: str
    project_version: str
    model_name: str
    model_alias: str
    model_source: str
    model_path: Path
    feature_columns_path: Path
    metadata_path: Path
    threshold: float
    log_file: Path
    log_level: str
    expectations_file: Path
    reference_distribution: Path
    versioned_artifacts_dir: Path
    training_features_path: Path
    reference_features_path: Path
    validation_report_path: Path
    database_enabled: bool
    database_url: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    api_host: str
    api_port: int
    max_batch_size: int
    drift_minimum_samples: int
    drift_psi_threshold: float
    error_rate_alert_threshold: float
    latency_alert_seconds: float
    feature_columns: List[str]


def _env(name: str, default: Any) -> Any:
    return os.getenv(name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load YAML configuration and apply deployment environment overrides."""

    path = _absolute(config_path or "config/config.yaml")
    with path.open("r", encoding="utf-8") as config_file:
        config: Dict[str, Any] = yaml.safe_load(config_file)

    paths = config["paths"]
    model = config["model"]
    api = config["api"]
    database = config["database"]
    mlflow_config = config["mlflow"]
    logging_config = config["logging"]
    monitoring = config["monitoring"]

    return Settings(
        project_name=config["project"]["name"],
        project_version=config["project"]["version"],
        model_name=_env("MLOPS_MODEL_NAME", model["name"]),
        model_alias=_env("MLOPS_MODEL_ALIAS", model["alias"]),
        model_source=_env("MLOPS_MODEL_SOURCE", model["source"]).lower(),
        model_path=_absolute(_env("MLOPS_MODEL_PATH", model["local_model_path"])),
        feature_columns_path=_absolute(
            _env("MLOPS_FEATURE_COLUMNS_PATH", model["feature_columns_path"])
        ),
        metadata_path=_absolute(_env("MLOPS_METADATA_PATH", model["metadata_path"])),
        threshold=float(_env("MLOPS_MODEL_THRESHOLD", model["threshold"])),
        log_file=_absolute(_env("MLOPS_LOG_FILE", paths["log_file"])),
        log_level=str(_env("MLOPS_LOG_LEVEL", logging_config["level"])),
        expectations_file=_absolute(paths["expectations_file"]),
        reference_distribution=_absolute(paths["reference_distribution"]),
        versioned_artifacts_dir=_absolute(paths["versioned_artifacts_dir"]),
        training_features_path=_absolute(config["data"]["training_features_path"]),
        reference_features_path=_absolute(config["data"]["reference_features_path"]),
        validation_report_path=_absolute(config["data"]["validation_report_path"]),
        database_enabled=_as_bool(_env("MLOPS_DATABASE_ENABLED", database["enabled"])),
        database_url=str(_env("DATABASE_URL", database["url"])),
        mlflow_tracking_uri=str(
            _env("MLFLOW_TRACKING_URI", mlflow_config["tracking_uri"])
        ),
        mlflow_experiment_name=mlflow_config["experiment_name"],
        api_host=str(_env("MLOPS_API_HOST", api["host"])),
        api_port=int(_env("MLOPS_API_PORT", api["port"])),
        max_batch_size=int(_env("MLOPS_MAX_BATCH_SIZE", api["max_batch_size"])),
        drift_minimum_samples=int(monitoring["drift_minimum_samples"]),
        drift_psi_threshold=float(monitoring["drift_psi_threshold"]),
        error_rate_alert_threshold=float(monitoring["error_rate_alert_threshold"]),
        latency_alert_seconds=float(monitoring["latency_alert_seconds"]),
        feature_columns=list(model["feature_columns"]),
    )


def clear_settings_cache() -> None:
    """Clear cached settings for tests that change environment variables."""

    load_settings.cache_clear()

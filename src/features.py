"""Feature construction and serialized artifact utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd


logger = logging.getLogger(__name__)


def load_artifact(artifact_path: str) -> Any:
    """Load a fitted artifact without fitting it again."""

    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    logger.info("loading_artifact path=%s", path)
    return joblib.load(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create notebook-equivalent derived features for inference."""

    result = df.copy()

    if "order_purchase_timestamp" in result.columns:
        purchase_time = pd.to_datetime(
            result["order_purchase_timestamp"], errors="coerce", utc=True
        )
        result["purchase_year"] = result.get("purchase_year", purchase_time.dt.year)
        result["purchase_month"] = result.get("purchase_month", purchase_time.dt.month)
        result["purchase_dayofweek"] = result.get(
            "purchase_dayofweek", purchase_time.dt.dayofweek
        )
        result["purchase_hour"] = result.get("purchase_hour", purchase_time.dt.hour)

    if "purchase_dayofweek" in result.columns and "is_weekend" not in result.columns:
        result["is_weekend"] = (result["purchase_dayofweek"] >= 5).astype(int)

    if "freight_ratio" not in result.columns:
        if {"total_freight", "total_price"}.issubset(result.columns):
            denominator = result["total_price"].replace(0, pd.NA)
            result["freight_ratio"] = (result["total_freight"] / denominator).fillna(
                0.0
            )

    if "estimated_delivery_days" not in result.columns and {
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    }.issubset(result.columns):
        purchase = pd.to_datetime(
            result["order_purchase_timestamp"], errors="coerce", utc=True
        )
        estimated = pd.to_datetime(
            result["order_estimated_delivery_date"], errors="coerce", utc=True
        )
        result["estimated_delivery_days"] = (
            estimated - purchase
        ).dt.total_seconds() / 86400.0

    return result


def prepare_features(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Build, select, and order the exact features expected by the model."""

    prepared = build_features(df)
    missing_features = [
        column for column in feature_columns if column not in prepared.columns
    ]
    if missing_features:
        raise ValueError(f"Input is missing required features: {missing_features}")
    return prepared[list(feature_columns)]

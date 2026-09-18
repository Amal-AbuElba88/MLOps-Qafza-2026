"""Reproducible DVC data-quality and leakage report."""

from __future__ import annotations

import json

import pandas as pd

from src.config import load_settings


TARGET = "is_late"
LEAKAGE_COLUMNS = {
    "order_delivered_customer_date",
    "delivery_delay_days",
    "actual_delivery_days",
}


def validate_training_data() -> dict:
    settings = load_settings()
    data = pd.read_csv(settings.training_features_path)
    leakage_found = sorted(LEAKAGE_COLUMNS.intersection(data.columns))
    required = {TARGET, "customer_state", "total_price", "total_freight"}
    missing_required = sorted(required.difference(data.columns))
    target_values = (
        sorted(data[TARGET].dropna().unique().tolist()) if TARGET in data else []
    )
    report = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_required_columns": missing_required,
        "leakage_columns_found": leakage_found,
        "target_values": target_values,
        "target_null_rate": (
            float(data[TARGET].isna().mean()) if TARGET in data else 1.0
        ),
        "passed": not missing_required
        and not leakage_found
        and target_values == [0, 1]
        and not data[TARGET].isna().any(),
    }
    settings.validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.validation_report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not report["passed"]:
        raise ValueError(f"Training data validation failed: {report}")
    return report


if __name__ == "__main__":
    validate_training_data()

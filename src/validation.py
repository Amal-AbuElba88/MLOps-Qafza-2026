"""Great Expectations validation for inference records."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from great_expectations.dataset import PandasDataset

from src.config import load_settings


logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when incoming inference data violates the expectation suite."""

    def __init__(self, failures: Iterable[str]):
        self.failures = list(failures)
        super().__init__("; ".join(self.failures))


def _load_suite(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as suite_file:
        return json.load(suite_file)


def _failed_expectations(validation_result: Dict[str, Any]) -> List[str]:
    failures = []
    for result in validation_result.get("results", []):
        if result.get("success"):
            continue
        config = result.get("expectation_config", {})
        expectation = config.get("expectation_type", "unknown_expectation")
        column = config.get("kwargs", {}).get("column", "table")
        failures.append(f"{column}: {expectation} failed")
    return failures


def validate_prediction_input(
    df: pd.DataFrame, expectations_path: Optional[Path] = None
) -> bool:
    """Reject input that fails types, ranges, categories, or missing-value rules."""

    if df.empty:
        raise DataValidationError(["Input data cannot be empty"])

    settings = load_settings()
    suite_path = expectations_path or settings.expectations_file
    suite = _load_suite(suite_path)
    ge_data = PandasDataset(df.copy())
    result = ge_data.validate(
        expectation_suite=suite,
        result_format="SUMMARY",
        catch_exceptions=True,
    )
    if not result.get("success", False):
        failures = _failed_expectations(result)
        logger.warning("input_validation_failed failures=%s", failures)
        raise DataValidationError(failures)

    logger.info("input_validation_passed rows=%s", len(df))
    return True

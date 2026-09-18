import pandas as pd
import pytest

from src.config import load_settings
from src.features import prepare_features
from src.validation import DataValidationError, validate_prediction_input


def _prepared(sample_order):
    settings = load_settings()
    return prepare_features(pd.DataFrame([sample_order]), settings.feature_columns)


def test_great_expectations_accepts_valid_input(sample_order):
    assert validate_prediction_input(_prepared(sample_order)) is True


def test_great_expectations_rejects_bad_category(sample_order):
    frame = _prepared(sample_order)
    frame.loc[0, "main_payment_type"] = "cash"
    with pytest.raises(DataValidationError):
        validate_prediction_input(frame)


def test_great_expectations_rejects_null(sample_order):
    frame = _prepared(sample_order)
    frame.loc[0, "estimated_delivery_days"] = None
    with pytest.raises(DataValidationError):
        validate_prediction_input(frame)


def test_empty_data_is_rejected():
    with pytest.raises(DataValidationError):
        validate_prediction_input(pd.DataFrame())

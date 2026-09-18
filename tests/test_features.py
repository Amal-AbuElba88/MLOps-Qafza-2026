import pandas as pd

from src.config import load_settings
from src.features import build_features, prepare_features


def test_build_features_from_order_timestamps(sample_order):
    result = build_features(pd.DataFrame([sample_order]))
    assert result.loc[0, "purchase_year"] == 2018
    assert result.loc[0, "purchase_month"] == 5
    assert result.loc[0, "purchase_dayofweek"] == 2
    assert result.loc[0, "purchase_hour"] == 14
    assert result.loc[0, "is_weekend"] == 0
    assert result.loc[0, "estimated_delivery_days"] == 10
    assert result.loc[0, "freight_ratio"] == 20 / 150


def test_prepare_features_matches_model_order(sample_order):
    settings = load_settings()
    result = prepare_features(pd.DataFrame([sample_order]), settings.feature_columns)
    assert result.columns.tolist() == settings.feature_columns
    assert result.shape == (1, 15)

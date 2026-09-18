import pandas as pd

from src.config import load_settings
from src.validate_training_data import LEAKAGE_COLUMNS, TARGET


def test_training_data_schema_ranges_nulls_and_leakage():
    data = pd.read_csv(load_settings().training_features_path)
    assert TARGET in data.columns
    assert set(data[TARGET].dropna().unique()) == {0, 1}
    assert data[TARGET].isna().sum() == 0
    assert data["total_price"].ge(0).all()
    assert data["total_freight"].ge(0).all()
    assert not LEAKAGE_COLUMNS.intersection(data.columns)

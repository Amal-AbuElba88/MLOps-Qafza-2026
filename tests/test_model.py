import joblib
import numpy as np
import pandas as pd

from src.config import load_settings
from src.features import prepare_features
from src.model_registry import load_model_bundle
from src.predict import DeliveryPredictor


def test_model_loads_and_predicts_expected_shape(sample_order):
    settings = load_settings()
    predictor = DeliveryPredictor(load_model_bundle(settings))
    predictions, probabilities = predictor.predict(pd.DataFrame([sample_order]))
    assert predictions.shape == (1,)
    assert probabilities.shape == (1,)
    assert predictions[0] in {0, 1}


def test_production_pipeline_matches_notebook_artifact(sample_order):
    settings = load_settings()
    notebook_model = joblib.load(settings.model_path)
    predictor = DeliveryPredictor(load_model_bundle(settings))
    features = prepare_features(pd.DataFrame([sample_order]), settings.feature_columns)
    notebook_probability = notebook_model.predict_proba(features)[:, 1]
    _, production_probability = predictor.predict(pd.DataFrame([sample_order]))
    np.testing.assert_allclose(
        production_probability, notebook_probability, rtol=0, atol=0
    )


def test_known_held_out_rows_are_predictable():
    settings = load_settings()
    data = pd.read_csv(settings.reference_features_path, nrows=5)
    predictor = DeliveryPredictor(load_model_bundle(settings))
    predictions, probabilities = predictor.predict(data)
    assert len(predictions) == 5
    assert np.all((probabilities >= 0) & (probabilities <= 1))

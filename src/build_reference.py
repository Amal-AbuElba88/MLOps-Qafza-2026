"""Build the reference probability distribution used for PSI drift checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import load_settings
from src.model_registry import load_model_bundle
from src.predict import DeliveryPredictor


def build_reference() -> dict:
    settings = load_settings()
    predictor = DeliveryPredictor(load_model_bundle(settings))
    data = pd.read_csv(settings.reference_features_path)
    _, probabilities = predictor.predict(data)
    if probabilities is None:
        raise RuntimeError("Reference distribution requires predict_proba")
    bins = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    counts, _ = np.histogram(probabilities, bins=bins)
    proportions = (counts / counts.sum()).tolist()
    return {
        "description": "Baseline from Task 2 held-out test features",
        "bins": bins.tolist(),
        "proportions": proportions,
        "sample_count": int(len(probabilities)),
    }


if __name__ == "__main__":
    print(json.dumps(build_reference(), indent=2))

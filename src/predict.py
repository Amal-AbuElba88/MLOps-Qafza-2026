"""Inference facade that never fits or mutates model artifacts."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features import prepare_features
from src.model_registry import ModelBundle


logger = logging.getLogger(__name__)


class DeliveryPredictor:
    def __init__(self, bundle: ModelBundle):
        self.model = bundle.model
        self.feature_columns = bundle.feature_columns
        self.threshold = bundle.threshold
        self.model_version = bundle.version
        self.model_source = bundle.source

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return thresholded predictions and late-delivery probabilities."""

        features = prepare_features(df, self.feature_columns)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)
        else:
            predictions = np.asarray(self.model.predict(features)).astype(int)
            probabilities = None

        logger.info("predictions_generated count=%s", len(predictions))
        return predictions, probabilities

    def predict_records(
        self, records: Sequence[dict]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.predict(pd.DataFrame(list(records)))

"""Command-line access to the same validated inference pipeline as the API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_settings
from src.features import prepare_features
from src.model_registry import load_model_bundle
from src.predict import DeliveryPredictor
from src.validation import validate_prediction_input


def predict_file(input_path: Path) -> dict:
    settings = load_settings()
    predictor = DeliveryPredictor(load_model_bundle(settings))
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    frame = prepare_features(pd.DataFrame([payload]), predictor.feature_columns)
    validate_prediction_input(frame)
    predictions, probabilities = predictor.predict(frame)
    probability = float(probabilities[0]) if probabilities is not None else None
    prediction = int(predictions[0])
    return {
        "prediction": "late" if prediction else "on_time",
        "is_late": prediction,
        "probability": probability,
        "model_version": predictor.model_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Delivery-delay inference CLI")
    parser.add_argument("command", choices=["predict"])
    parser.add_argument("--input", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.command == "predict":
        print(json.dumps(predict_file(arguments.input), indent=2))


if __name__ == "__main__":
    main()

"""Prometheus service metrics and rolling prediction-drift monitoring."""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List

from prometheus_client import Counter, Gauge, Histogram


REQUEST_COUNT = Counter(
    "delivery_api_requests_total", "API requests", ["method", "route", "status"]
)
REQUEST_LATENCY = Histogram(
    "delivery_api_request_latency_seconds", "API request latency", ["route"]
)
PREDICTION_COUNT = Counter(
    "delivery_predictions_total", "Predictions by class", ["prediction"]
)
PREDICTION_PROBABILITY = Histogram(
    "delivery_prediction_probability", "Predicted late-delivery probability"
)
PREDICTION_DRIFT_PSI = Gauge(
    "delivery_prediction_drift_psi", "PSI of recent prediction probabilities"
)
PREDICTION_DRIFT_ALERT = Gauge(
    "delivery_prediction_drift_alert", "1 when probability PSI exceeds threshold"
)


class DriftMonitor:
    def __init__(
        self,
        reference_path: Path,
        minimum_samples: int,
        psi_threshold: float,
        window_size: int = 1000,
    ):
        with reference_path.open("r", encoding="utf-8") as reference_file:
            reference: Dict[str, List[float]] = json.load(reference_file)
        self.bins = reference["bins"]
        self.expected = reference["proportions"]
        self.minimum_samples = minimum_samples
        self.psi_threshold = psi_threshold
        self.values: Deque[float] = deque(maxlen=window_size)
        self.lock = threading.Lock()

    def observe(self, probabilities: Iterable[float]) -> float:
        with self.lock:
            self.values.extend(float(value) for value in probabilities)
            psi = self.current_psi()
        if not math.isnan(psi):
            PREDICTION_DRIFT_PSI.set(psi)
            PREDICTION_DRIFT_ALERT.set(float(psi >= self.psi_threshold))
        return psi

    def current_psi(self) -> float:
        if len(self.values) < self.minimum_samples:
            return float("nan")
        counts = [0] * (len(self.bins) - 1)
        for value in self.values:
            for index in range(len(self.bins) - 1):
                upper_inclusive = index == len(self.bins) - 2
                if self.bins[index] <= value < self.bins[index + 1] or (
                    upper_inclusive and value == self.bins[index + 1]
                ):
                    counts[index] += 1
                    break
        actual = [count / len(self.values) for count in counts]
        epsilon = 1e-6
        return sum(
            (observed - expected)
            * math.log((observed + epsilon) / (expected + epsilon))
            for observed, expected in zip(actual, self.expected)
        )

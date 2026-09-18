"""FastAPI inference service for Olist delivery-delay predictions."""

from __future__ import annotations

import logging
import time
import uuid
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    OrderInput,
    PredictionResponse,
)
from src.config import load_settings
from src.database import PredictionStore
from src.features import prepare_features
from src.logging_config import setup_logging
from src.model_registry import load_model_bundle
from src.monitoring import (
    PREDICTION_COUNT,
    PREDICTION_PROBABILITY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    DriftMonitor,
)
from src.predict import DeliveryPredictor
from src.validation import DataValidationError, validate_prediction_input


settings = load_settings()
setup_logging(settings)
logger = logging.getLogger(__name__)

model_bundle = load_model_bundle(settings)
predictor = DeliveryPredictor(model_bundle)
prediction_store = PredictionStore(settings.database_url, settings.database_enabled)
drift_monitor = DriftMonitor(
    settings.reference_distribution,
    settings.drift_minimum_samples,
    settings.drift_psi_threshold,
)

app = FastAPI(
    title="Olist Delivery Delay Prediction API",
    description="Predict whether a new order will be delivered late or on time.",
    version=settings.project_version,
)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    """Measure every request, including validation and server errors."""

    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency = time.perf_counter() - start
        route = request.url.path
        REQUEST_COUNT.labels(request.method, route, str(status_code)).inc()
        REQUEST_LATENCY.labels(route).observe(latency)


def _model_ready_frame(orders: List[OrderInput]) -> pd.DataFrame:
    records = [order.model_dump(mode="json", exclude_none=True) for order in orders]
    return prepare_features(pd.DataFrame(records), predictor.feature_columns)


def _save_prediction(
    request_id: str,
    payload: dict,
    prediction: int,
    probability: float,
    latency_ms: float,
) -> None:
    try:
        prediction_store.save(
            request_id=request_id,
            payload=payload,
            prediction=prediction,
            probability=probability,
            model_version=predictor.model_version,
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("prediction_log_storage_failed request_id=%s", request_id)


def _response(
    request_id: str, prediction: int, probability: float
) -> PredictionResponse:
    return PredictionResponse(
        request_id=request_id,
        prediction="late" if prediction == 1 else "on_time",
        is_late=prediction,
        probability=probability,
        model_version=predictor.model_version,
    )


@app.get("/", include_in_schema=False)
def root():
    return {"docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        service=settings.project_name,
        model_version=predictor.model_version,
        model_source=predictor.model_source,
    )


@app.get("/model-info")
def model_info():
    return {
        "model_name": settings.model_name,
        "model_version": predictor.model_version,
        "model_source": predictor.model_source,
        "model_alias": settings.model_alias,
        "decision_threshold": predictor.threshold,
        "number_of_features": len(predictor.feature_columns),
        "feature_columns": predictor.feature_columns,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_order(order: OrderInput):
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    payload = order.model_dump(mode="json", exclude_none=True)
    try:
        frame = _model_ready_frame([order])
        validate_prediction_input(frame)
        predictions, probabilities = predictor.predict(frame)
        prediction = int(predictions[0])
        probability = float(probabilities[0]) if probabilities is not None else 0.0
        latency_ms = (time.perf_counter() - started) * 1000

        PREDICTION_COUNT.labels("late" if prediction else "on_time").inc()
        PREDICTION_PROBABILITY.observe(probability)
        drift_monitor.observe([probability])
        _save_prediction(request_id, payload, prediction, probability, latency_ms)
        logger.info(
            "prediction_completed",
            extra={
                "request_id": request_id,
                "payload": payload,
                "prediction": prediction,
                "probability": probability,
                "model_version": predictor.model_version,
                "latency_ms": round(latency_ms, 3),
            },
        )
        return _response(request_id, prediction, probability)
    except DataValidationError as error:
        logger.warning(
            "prediction_rejected request_id=%s failures=%s",
            request_id,
            error.failures,
        )
        raise HTTPException(status_code=422, detail=error.failures) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("prediction_failed request_id=%s", request_id)
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_orders_batch(orders: List[OrderInput]):
    if not orders:
        raise HTTPException(status_code=400, detail="The orders list cannot be empty")
    if len(orders) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {settings.max_batch_size} orders",
        )

    started = time.perf_counter()
    try:
        frame = _model_ready_frame(orders)
        validate_prediction_input(frame)
        predictions, probabilities = predictor.predict(frame)
        responses = []
        observed_probabilities = []
        for index, order in enumerate(orders):
            request_id = str(uuid.uuid4())
            prediction = int(predictions[index])
            probability = (
                float(probabilities[index]) if probabilities is not None else 0.0
            )
            latency_ms = (time.perf_counter() - started) * 1000
            payload = order.model_dump(mode="json", exclude_none=True)
            PREDICTION_COUNT.labels("late" if prediction else "on_time").inc()
            PREDICTION_PROBABILITY.observe(probability)
            observed_probabilities.append(probability)
            _save_prediction(request_id, payload, prediction, probability, latency_ms)
            responses.append(_response(request_id, prediction, probability))

        drift_monitor.observe(observed_probabilities)
        logger.info(
            "batch_prediction_completed count=%s model_version=%s latency_ms=%.3f",
            len(responses),
            predictor.model_version,
            (time.perf_counter() - started) * 1000,
        )
        return BatchPredictionResponse(
            predictions=responses,
            count=len(responses),
        )
    except DataValidationError as error:
        raise HTTPException(status_code=422, detail=error.failures) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("batch_prediction_failed")
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.cli import predict_file
from src.database import PredictionLog, PredictionStore


def test_cli_prediction(tmp_path, sample_order):
    input_path = tmp_path / "order.json"
    input_path.write_text(json.dumps(sample_order), encoding="utf-8")
    result = predict_file(input_path)
    assert result["prediction"] in {"late", "on_time"}
    assert 0 <= result["probability"] <= 1


def test_prediction_store_persists_record(tmp_path, sample_order):
    database_path = tmp_path / "predictions.db"
    store = PredictionStore(f"sqlite:///{database_path}")
    store.save("request-1", sample_order, 0, 0.2, "v1", 4.5)
    with Session(store.engine) as session:
        count = session.scalar(select(func.count()).select_from(PredictionLog))
    assert count == 1

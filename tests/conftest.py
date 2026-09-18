import os

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("MLOPS_MODEL_SOURCE", "local")
os.environ.setdefault("MLOPS_DATABASE_ENABLED", "false")

from app.main import app  # noqa: E402


SAMPLE_ORDER = {
    "customer_state": "SP",
    "total_price": 150.0,
    "total_freight": 20.0,
    "num_items": 2,
    "num_unique_sellers": 1,
    "total_payment": 170.0,
    "max_installments": 3,
    "main_payment_type": "credit_card",
    "order_purchase_timestamp": "2018-05-09T14:00:00Z",
    "order_estimated_delivery_date": "2018-05-19T14:00:00Z",
}


@pytest.fixture()
def sample_order():
    return SAMPLE_ORDER.copy()


@pytest.fixture()
def client():
    return TestClient(app)

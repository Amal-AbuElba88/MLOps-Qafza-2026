def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_version"] == "v1.0"


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "delivery_delay_model"
    assert body["number_of_features"] == 15
    assert 0 <= body["decision_threshold"] <= 1


def test_single_prediction_from_raw_order(client, sample_order):
    response = client.post("/predict", json=sample_order)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"late", "on_time"}
    assert body["is_late"] in {0, 1}
    assert 0 <= body["probability"] <= 1
    assert body["model_version"] == "v1.0"
    assert body["request_id"]


def test_batch_prediction(client, sample_order):
    second = sample_order.copy()
    second.update(
        {
            "customer_state": "RJ",
            "total_price": 80.0,
            "total_freight": 35.0,
            "total_payment": 115.0,
            "main_payment_type": "boleto",
        }
    )
    response = client.post("/predict-batch", json=[sample_order, second])
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_invalid_payloads_are_rejected(client, sample_order):
    invalid_state = sample_order.copy()
    invalid_state["customer_state"] = "XX"
    assert client.post("/predict", json=invalid_state).status_code == 422

    missing_feature = sample_order.copy()
    missing_feature.pop("customer_state")
    assert client.post("/predict", json=missing_feature).status_code == 422

    assert client.post("/predict-batch", json=[]).status_code == 400


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "delivery_api_requests_total" in response.text
    assert "delivery_prediction_drift_psi" in response.text


def test_openapi_contains_required_routes(client):
    schema = client.get("/openapi.json").json()
    assert {"/health", "/model-info", "/predict", "/predict-batch"}.issubset(
        schema["paths"]
    )

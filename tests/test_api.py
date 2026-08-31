"""Tests for the FastAPI prediction service."""

import pytest
from fastapi.testclient import TestClient

from deploy import api


class DummyModel:
    def predict(self, features):
        assert list(features.columns) == api.FEATURE_COLUMNS
        return [600.0]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "model", DummyModel())
    return TestClient(api.app)


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_seconds_and_minutes(client):
    response = client.post(
        "/predict",
        json={
            "passenger_count": 1,
            "distance_km": 5.2,
            "pickup_hour": 18,
            "pickup_day": 15,
            "pickup_month": 1,
            "pickup_weekday": 3,
            "rush_hour": 1,
            "is_weekend": 0,
            "vendor_id": 1,
            "store_and_fwd_flag": "N",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "trip_duration_seconds": 600.0,
        "trip_duration_minutes": 10.0,
    }


def test_predict_rejects_invalid_vendor(client):
    response = client.post(
        "/predict",
        json={
            "passenger_count": 1,
            "distance_km": 5.2,
            "pickup_hour": 18,
            "pickup_day": 15,
            "pickup_month": 1,
            "pickup_weekday": 3,
            "rush_hour": 1,
            "is_weekend": 0,
            "vendor_id": 3,
            "store_and_fwd_flag": "N",
        },
    )

    assert response.status_code == 422

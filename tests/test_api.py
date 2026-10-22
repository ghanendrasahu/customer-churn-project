"""Smoke tests for the churn API: the probe contract and the scoring path.

The model artifact is a build output and is not committed, so the scoring
test skips when it is missing - the probes must work either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api import app  # noqa: E402

MODEL_PRESENT = (ROOT / "models" / "model_artifacts.pkl").is_file()


@pytest.fixture()
def client():
    with TestClient(app) as live:
        yield live


def test_health_is_always_available(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_the_model_state(client):
    response = client.get("/ready")

    # 200 with the model loaded, 503 with a reason when the artifact is absent.
    assert response.status_code in {200, 503}
    body = response.json()
    if response.status_code == 503:
        assert body["detail"]
    else:
        assert body["model_loaded"] is True


@pytest.mark.skipif(not MODEL_PRESENT, reason="model artifact is a build output")
def test_predict_scores_a_customer(client):
    payload = {
        "tenure": 42,
        "MonthlyCharges": 70.5,
        "Contract": "One year",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
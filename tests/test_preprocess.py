"""Regression tests for the train/serve feature contract.

The original runtime left demographic columns as strings, so get_dummies
created columns the artifact does not contain and the model always saw zeros
for gender / Partner / Dependents / PaperlessBilling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import predict_runtime as runtime  # noqa: E402


def _customer(**overrides) -> dict:
    payload = {
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 42,
        "MonthlyCharges": 70.5,
        "TotalCharges": 2961.0,
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
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def feature_contract(monkeypatch):
    """Pretend the artifact was loaded with a small, explicit feature list."""
    columns = [
        "gender",
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "num_services",
        "charge_per_service",
        "InternetService_Fiber optic",
    ]
    monkeypatch.setitem(runtime._state, "feature_cols", columns)
    monkeypatch.setitem(runtime._state, "scaler", None)
    monkeypatch.setitem(runtime._state, "scale_cols", None)
    return columns


def test_demographics_are_numeric(feature_contract):
    male = runtime.preprocess(_customer(gender="Male", Partner="Yes", Dependents="No"))
    female = runtime.preprocess(
        _customer(gender="Female", Partner="No", Dependents="Yes")
    )

    assert male.iloc[0]["gender"] == 1
    assert female.iloc[0]["gender"] == 0
    assert male.iloc[0]["Partner"] == 1
    assert female.iloc[0]["Partner"] == 0
    assert male.iloc[0]["Dependents"] == 0
    assert female.iloc[0]["Dependents"] == 1
    assert male.iloc[0]["PaperlessBilling"] == 1


def test_categorical_columns_are_one_hot_encoded(feature_contract):
    fiber = runtime.preprocess(_customer(InternetService="Fiber optic"))
    dsl = runtime.preprocess(_customer(InternetService="DSL"))

    assert fiber.iloc[0]["InternetService_Fiber optic"] == 1
    assert dsl.iloc[0]["InternetService_Fiber optic"] == 0


def test_service_counts_and_charge_ratio(feature_contract):
    row = runtime.preprocess(_customer()).iloc[0]

    # Five "Yes" services in the default payload.
    assert row["num_services"] == 5
    assert row["charge_per_service"] == pytest.approx(70.5 / 6, abs=0.01)


def test_missing_fields_raise_a_clear_error(feature_contract):
    payload = _customer()
    del payload["Contract"]

    with pytest.raises(ValueError, match="Contract"):
        runtime.preprocess(payload)

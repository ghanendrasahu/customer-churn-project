"""Shared churn scoring runtime.

Both the FastAPI surface (app/api.py) and the Streamlit dashboard
(app/app.py) score customers through this module so they agree on
artifact loading, feature engineering and the result shape.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", str(BASE_DIR / "models" / "model_artifacts.pkl")))

_SVC_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

_state: dict = {
    "loaded": False,
    "model": None,
    "feature_cols": [],
    "scaler": None,
    "scale_cols": None,
    "error": None,
}


def load_artifacts() -> bool:
    """Load the model artifacts once; remember any failure for /ready."""
    if _state["loaded"]:
        return True
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file at: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as handle:
            artifacts = pickle.load(handle)
        _state["model"] = artifacts["best_model"]
        _state["feature_cols"] = artifacts["feature_cols"]
        _state["scaler"] = artifacts.get("scaler")
        _state["scale_cols"] = artifacts.get("scale_cols")
        _state["error"] = None
        _state["loaded"] = True
        print(f"Pipeline ready. Loaded {len(_state['feature_cols'])} features.")
        return True
    except Exception as exc:  # noqa: BLE001 - surfaced through predict()/ready
        _state["error"] = str(exc)
        print(f"Model artifacts unavailable: {exc}")
        return False


def load_error() -> str | None:
    """Reason the artifacts could not be loaded, or None when all is well."""
    return _state["error"]


def feature_count() -> int:
    return len(_state["feature_cols"])


def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    for col in _SVC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    df["num_services"] = df[_SVC_COLS].sum(axis=1)
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)
    df["is_longterm"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
    df["unprotected_internet"] = (
        (df["InternetService"] != "No") & (df["OnlineSecurity"] == 0) & (df["TechSupport"] == 0)
    ).astype(int)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
        include_lowest=True,
    )

    df = pd.get_dummies(df)
    df = df.reindex(columns=_state["feature_cols"], fill_value=0)

    if _state["scaler"] is not None and _state["scale_cols"] is not None:
        df[_state["scale_cols"]] = _state["scaler"].transform(df[_state["scale_cols"]])

    return df


def predict(data: dict) -> dict:
    """Score one customer. Returns the API result shape, or {"error": ...}."""
    if not load_artifacts():
        return {"error": f"Model artifacts unavailable: {_state['error']}"}
    try:
        df_processed = preprocess(data)
        prob = _state["model"].predict_proba(df_processed)[0][1]
        prediction = int(prob > 0.5)

        return {
            "churn_probability": round(float(prob), 4),
            "prediction": prediction,
            "status": "Success",
        }
    except Exception as exc:  # noqa: BLE001 - callers render the message
        import traceback

        print(traceback.format_exc())
        return {"error": str(exc)}

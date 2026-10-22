import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Telco churn probability scoring with domain features.",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_artifacts.pkl"

# The artifacts are loaded lazily: the process must start (and answer
# /health) even when the model file is missing, so an orchestrator can tell
# "not ready" apart from "crashed".
_artifacts: dict | None = None
model = None
feature_cols: list = []
scaler = None
scale_cols = None
_load_error: str | None = None


def load_artifacts() -> bool:
    """Load the model artifacts once; remember any failure for /ready."""
    global _artifacts, model, feature_cols, scaler, scale_cols, _load_error

    if model is not None:
        return True
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file at: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as handle:
            artifacts = pickle.load(handle)
        _artifacts = artifacts
        model = artifacts["best_model"]
        feature_cols = artifacts["feature_cols"]
        scaler = artifacts.get("scaler")
        scale_cols = artifacts.get("scale_cols")
        _load_error = None
        print(f"Pipeline ready. Loaded {len(feature_cols)} features.")
        return True
    except Exception as exc:  # noqa: BLE001 - surfaced through /ready
        _load_error = str(exc)
        print(f"Model artifacts unavailable: {exc}")
        return False


load_artifacts()


@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}


@app.get("/health")
def health():
    """Liveness: the process is up. Never depends on the model artifact."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: the model artifact is loaded, so scoring can be served."""
    if not load_artifacts():
        raise HTTPException(
            status_code=503, detail=f"Model artifacts unavailable: {_load_error}"
        )
    return {"status": "ready", "model_loaded": True, "features": len(feature_cols)}


def preprocess(data: dict):
    df = pd.DataFrame([data])

    svc_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in svc_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    df["num_services"] = df[svc_cols].sum(axis=1)
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
    df = df.reindex(columns=feature_cols, fill_value=0)

    if scaler is not None and scale_cols is not None:
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df


@app.post("/predict")
def predict(data: dict):
    try:
        df_processed = preprocess(data)
        prob = model.predict_proba(df_processed)[0][1]
        prediction = int(prob > 0.5)

        return {
            "churn_probability": round(float(prob), 4),
            "prediction": prediction,
            "status": "Success",
        }
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        return {"error": str(e)}

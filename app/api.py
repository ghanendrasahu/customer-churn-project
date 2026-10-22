"""FastAPI surface for the churn scoring runtime.

All scoring, artifact loading and preprocessing live in
app/predict_runtime.py; this module only exposes it over HTTP.
"""

from fastapi import FastAPI, HTTPException

from app.predict_runtime import feature_count, load_artifacts, load_error, predict

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Telco churn probability scoring with domain features.",
)


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
            status_code=503, detail=f"Model artifacts unavailable: {load_error()}"
        )
    return {"status": "ready", "model_loaded": True, "features": feature_count()}


@app.post("/predict")
def predict_http(data: dict):
    return predict(data)

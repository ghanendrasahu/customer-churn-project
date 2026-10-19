"""FastAPI surface for the churn scoring runtime.

All scoring, artifact loading and preprocessing live in
app/predict_runtime.py; this module only exposes it over HTTP.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predict_runtime import feature_count, load_artifacts, load_error, predict

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Telco churn probability scoring with domain features.",
)


class ChurnCustomer(BaseModel):
    """One customer record. Mirrors the Streamlit form fields."""

    gender: str = Field(pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: str = Field(pattern="^(Yes|No)$")
    Dependents: str = Field(pattern="^(Yes|No)$")
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str = Field(pattern="^(Yes|No)$")
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


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
def predict_http(customer: ChurnCustomer):
    result = predict(customer.model_dump())
    if "error" in result:
        # 503 when the artifact is missing; 500 when scoring itself failed.
        status = 503 if not load_artifacts() else 500
        raise HTTPException(status_code=status, detail=result["error"])
    return result

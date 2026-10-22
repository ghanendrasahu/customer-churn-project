# Customer Churn Prediction

Telco churn scoring: a Random Forest trained on 7,043 customer records with
engineered domain features (tenure group, charge per service, contract flag,
unprotected internet), served behind a FastAPI endpoint that returns the
churn probability and the decision.

The service is **probe-friendly**: model artifacts load lazily, so the
process starts and reports "not ready" instead of crashing when the model
file is missing.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.train                       # train -> models/model_artifacts.pkl
uvicorn app.api:app --reload
```

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Service banner |
| `GET` | `/health` | Liveness - always 200 while the process is up |
| `GET` | `/ready` | Readiness - 200 when the model is loaded, 503 otherwise |
| `POST` | `/predict` | Score a customer |

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 42, "MonthlyCharges": 70.5, "Contract": "One year",
       "InternetService": "Fiber optic", "OnlineSecurity": "No",
       "TechSupport": "No", "PhoneService": "Yes", "MultipleLines": "Yes",
       "OnlineBackup": "Yes", "DeviceProtection": "No",
       "StreamingTV": "Yes", "StreamingMovies": "Yes"}'
```

```json
{"churn_probability": 0.383, "prediction": 0, "status": "Success"}
```

## How the model works

1. **Service columns** are binarised (`Yes`/`No` → 1/0) and the number of
   active services is derived.
2. **Domain features**: charge per service, long-contract flag, unprotected
   internet (no security *and* no support) and a tenure band.
3. The pipeline is scaled and fitted with a **Random Forest**; the
   decision threshold is tuned for recall on the churn class, because a
   missed churn costs more than a false alarm.
4. `models/model_artifacts.pkl` holds the fitted pipeline, the feature list
   and the scaler.

## Project structure

```text
app/api.py       FastAPI service (/health, /ready, /predict)
app/train.py     training and threshold tuning
models/          model_artifacts.pkl
data/            the telco dataset
notebooks/       exploration
```

## Notes

- `/ready` and `/health` are the contract an orchestrator needs: the
  container must start even without artifacts, and report "not ready"
  until the model can be loaded.
- Retrain with `python -m app.train` and restart the service.

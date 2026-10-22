import os
import sys
from pathlib import Path

import requests
import streamlit as st

# Allow `streamlit run app/app.py` from the repo root and from any cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.predict_runtime import predict_churn  # noqa: E402

# Set CHURN_API_BASE (e.g. https://churn-api.onrender.com) to score through
# the FastAPI service instead of the in-process model. Streamlit Cloud uses
# the in-process path, since it cannot reach a localhost sidecar.
API_BASE = os.environ.get("CHURN_API_BASE")

st.set_page_config(page_title="Telco Churn Analytics", layout="wide")

st.title("Customer Churn Prediction Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.subheader("Subscription Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

with col3:
    st.subheader("Services and Charges")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

with st.expander("Additional Services"):
    s1, s2, s3 = st.columns(3)
    phone = s1.selectbox("Phone Service", ["Yes", "No"])
    multiple = s1.selectbox("Multiple Lines", ["Yes", "No"])
    security = s2.selectbox("Online Security", ["Yes", "No"])
    backup = s2.selectbox("Online Backup", ["Yes", "No"])
    protection = s3.selectbox("Device Protection", ["Yes", "No"])
    support = s3.selectbox("Tech Support", ["Yes", "No"])
    tv = st.selectbox("Streaming TV", ["Yes", "No"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No"])

data = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "PaperlessBilling": paperless,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "InternetService": internet,
    "Contract": contract,
    "PaymentMethod": payment,
}

st.markdown("---")

def show_result(res: dict) -> None:
    if "error" in res:
        st.error(f"Logic Error: {res['error']}")
        return

    prob = res["churn_probability"]
    risk_val = prob * 100

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="Churn Probability", value=f"{risk_val:.1f}%")
        if prob > 0.7:
            st.error("HIGH RISK: This customer is likely to leave.")
        elif prob > 0.4:
            st.warning("MEDIUM RISK: Targeted retention recommended.")
        else:
            st.success("LOW RISK: This is likely a loyal customer.")

    with res_col2:
        st.subheader("Retention Strategy")
        if prob > 0.5:
            st.write("1. Offer upgrade to long-term contract.")
            st.write("2. Bundle tech support or online security.")
            st.write("3. Provide a loyalty discount on monthly charges.")
        else:
            st.write("Maintain current service level and consider selective upsell.")


if st.button("Analyze Customer Risk", use_container_width=True):
    if API_BASE:
        try:
            res = requests.post(f"{API_BASE}/predict", json=data, timeout=30).json()
        except Exception:
            st.error(
                f"Connection Error: FastAPI did not respond at {API_BASE}. "
                "Unset CHURN_API_BASE to use the built-in in-process model."
            )
            st.stop()
    else:
        res = predict_churn(data)
    show_result(res)

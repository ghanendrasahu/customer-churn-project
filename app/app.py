"""Customer churn assessment UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from predict_runtime import predict as predict_churn  # noqa: E402

import ui  # noqa: E402

API_BASE = os.environ.get("CHURN_API_BASE")

ui.setup(
    "Churn Risk", icon=ui.app_icon(_APP_DIR.parent / "assets"), sidebar_state="expanded"
)

st.markdown(
    """
    <style>
      .churn-bar { height: 10px; border-radius: 999px; background: #232B3D; overflow: hidden; }
      .churn-bar span { display: block; height: 100%; border-radius: 999px; }
      .churn-scale { display: flex; justify-content: space-between; margin-top: 6px;
                     color: #9AA4B8; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

ui.page_header(
    "Customer churn",
    "Estimate how likely a subscriber is to leave, then act on the factors that keep them.",
    eyebrow="CUSTOMER SUCCESS",
    badge="Model ready",
    icon_name="users",
)

left, right = st.columns([1.5, 1], gap="large")

with left:
    ui.section(
        "Customer profile",
        "Start from the current account details. Adjust any field to test a scenario.",
    )
    with st.form("churn_form"):
        st.markdown("**Demographics**")
        d1, d2 = st.columns(2)
        gender = d1.selectbox("Gender", ["Male", "Female"])
        senior = d2.selectbox("Senior citizen", ["No", "Yes"])
        partner = d1.selectbox("Partner", ["Yes", "No"])
        dependents = d2.selectbox("Dependents", ["Yes", "No"])

        st.markdown("**Subscription**")
        s1, s2 = st.columns(2)
        tenure = s1.slider("Tenure (months)", 0, 72, 12)
        contract = s2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = s1.selectbox("Paperless billing", ["Yes", "No"])
        payment = s2.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        st.markdown("**Services and charges**")
        c1, c2, c3 = st.columns(3)
        internet = c1.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        monthly = c2.number_input("Monthly charges (USD)", 0.0, 200.0, 70.0)
        total = c3.number_input("Total charges (USD)", 0.0, 10000.0, 840.0)

        with st.expander("Additional services"):
            a1, a2, a3 = st.columns(3)
            phone = a1.selectbox("Phone service", ["Yes", "No"])
            multiple = a1.selectbox("Multiple lines", ["Yes", "No"])
            security = a2.selectbox("Online security", ["Yes", "No"])
            backup = a2.selectbox("Online backup", ["Yes", "No"])
            protection = a3.selectbox("Device protection", ["Yes", "No"])
            support = a3.selectbox("Tech support", ["Yes", "No"])
            tv = a1.selectbox("Streaming TV", ["Yes", "No"])
            movies = a2.selectbox("Streaming movies", ["Yes", "No"])

        submitted = st.form_submit_button(
            "Assess risk", type="primary", width="stretch"
        )

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

RISK_BANDS = (
    (
        0.7,
        "High risk",
        "danger",
        "#F17C7C",
        "Likely to leave. Act before the next renewal.",
    ),
    (
        0.4,
        "Medium risk",
        "warning",
        "#F2B544",
        "Some warning signs. A targeted offer can help.",
    ),
    (
        0.0,
        "Low risk",
        "success",
        "#46D39A",
        "Likely to stay. Maintain service and upsell carefully.",
    ),
)


def show_result(res: dict) -> None:
    if "error" in res:
        ui.error_state(
            "The model could not score this profile. Check the account fields and try again.",
            title="Assessment failed",
            hint="Your customer data was not stored.",
        )
        return

    prob = float(res["churn_probability"])
    pct = prob * 100
    _, band, tone, color, message = next(
        (entry for entry in RISK_BANDS if prob > entry[0]), RISK_BANDS[-1]
    )

    ui.metric_grid(
        [
            {
                "label": "Churn probability",
                "value": f"{pct:.1f}%",
                "caption": "estimated for this profile",
            }
        ]
    )
    st.markdown(
        f'<div class="churn-bar"><span style="width:{pct:.1f}%; background:{color};"></span></div>'
        '<div class="churn-scale"><span>0%</span><span>100%</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        ui.badge(band, tone)
        + f"<p style='margin:.6rem 0 0; color:#9AA4B8'>{message}</p>",
        unsafe_allow_html=True,
    )

    ui.section("Retention playbook")
    if prob > 0.5:
        st.markdown(
            "- Offer an upgrade to a one or two year contract.\n"
            "- Bundle tech support or online security.\n"
            "- Add a loyalty discount to the monthly charge.\n"
            "- Schedule a check-in before the next billing date."
        )
    else:
        st.markdown(
            "- Keep the current service level.\n"
            "- Consider a selective upsell for streaming or backup.\n"
            "- Re-score after the next contract change."
        )


with right:
    ui.section("Assessment")
    if submitted:
        if API_BASE:
            try:
                response = requests.post(f"{API_BASE}/predict", json=data, timeout=30)
                response.raise_for_status()
                result = response.json()
            except Exception:
                ui.error_state(
                    "The scoring service did not respond. Try again in a moment.",
                    title="Service unavailable",
                    hint="The in-process model is used when no external service is configured.",
                )
                result = None
        else:
            result = predict_churn(data)

        if result is not None:
            show_result(result)
    else:
        ui.empty_state(
            "No assessment yet",
            "Adjust the customer profile and select Assess risk. The churn probability "
            "and a retention playbook will appear here.",
            icon_name="chart",
        )

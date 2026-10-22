import os
import pickle
import warnings

import matplotlib
import pandas as pd

matplotlib.use("Agg") # Required for running without a GUI
from pathlib import Path

import matplotlib.ticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Define Paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_pipeline():
    print("=" * 68)
    print("SECTION 1 — DATA LOADING & CLEANING")
    print("=" * 68)

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    
    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask_tc = df["TotalCharges"].isna()
    df.loc[mask_tc, "TotalCharges"] = df.loc[mask_tc, "tenure"] * df.loc[mask_tc, "MonthlyCharges"]
    
    df.drop(columns=["customerID"], inplace=True, errors='ignore')
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    print("\n" + "=" * 68)
    print("SECTION 3 — FEATURE ENGINEERING")
    print("=" * 68)

    df_fe = df.copy()
    
    # Tenure groups
    df_fe["tenure_group"] = pd.cut(
        df_fe["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
        include_lowest=True
    )

    # Feature Engineering logic
    svc_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    
    df_fe["num_services"] = df_fe[svc_cols].isin(["Yes"]).sum(axis=1)
    df_fe["charge_per_service"] = (df_fe["MonthlyCharges"] / (df_fe["num_services"] + 1)).round(2)
    df_fe["is_longterm"] = df_fe["Contract"].isin(["One year", "Two year"]).astype(int)
    
    df_fe["unprotected_internet"] = (
        (df_fe["InternetService"] != "No") &
        (df_fe["OnlineSecurity"] == "No") &
        (df_fe["TechSupport"] == "No")
    ).astype(int)

    # Encoding
    binary_map = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines",
                   "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    
    for col in yes_no_cols:
        df_fe[col] = df_fe[col].map(binary_map)
    
    df_fe["gender"] = (df_fe["gender"] == "Male").astype(int)
    
    ohe_cols = ["InternetService", "Contract", "PaymentMethod", "tenure_group"]
    df_fe = pd.get_dummies(df_fe, columns=ohe_cols, drop_first=False)

    feature_cols = [c for c in df_fe.columns if c != "Churn"]
    X = df_fe[feature_cols].astype(float)
    y = df_fe["Churn"].values

    SCALE_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "charge_per_service"]
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[SCALE_COLS] = scaler.fit_transform(X[SCALE_COLS])

    # Model Training (Simplifying for brevity, using Random Forest as Best for artifacts)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    best_model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
    best_model.fit(X_train, y_train)
    
    print("\n" + "=" * 68)
    print("SAVING ARTIFACTS")
    print("=" * 68)
    
    artifacts = {
        "best_model": best_model,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "scale_cols": SCALE_COLS
    }
    
    with open(MODEL_DIR / "model_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print(f"✅ Successfully saved model_artifacts.pkl to {MODEL_DIR}")

if __name__ == "__main__":
    run_pipeline()
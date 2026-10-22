#!/usr/bin/env bash
# Streamlit Cloud entry point.
#
# The dataset and model artifacts are build outputs (gitignored), so they are
# generated deterministically at boot. Every step is idempotent and skipped
# when its output already exists.
set -euo pipefail
cd "$(dirname "$0")"

pip install -r requirements.txt

if [ ! -f data/telco_churn.csv ]; then
  python src/generate_dataset.py
fi

if [ ! -f models/model_artifacts.pkl ]; then
  python src/churn_analysis.py
fi

exec streamlit run app/app.py

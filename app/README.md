# App

This folder contains the interactive Streamlit dashboard for the capstone project.

## Running the App

```bash
# From the repository root:
pip install streamlit plotly pandas numpy scikit-learn lightgbm xgboost joblib pyarrow

streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`.

## Features

- **📊 Overview** — Dataset summary stats and time-series visualisation
- **🔍 EDA Explorer** — Interactive charts: hourly patterns, seasonal boxplots, sub-meter breakdown
- **🤖 Forecasting** — Run any trained model on the test set and see predictions vs actuals
- **📈 Model Comparison** — Side-by-side performance metrics for all models
- **💡 Insights** — Actionable energy-saving recommendations based on model findings

## Notes

- The app loads `../data/hourly_clean.csv` and `../models/feature_cols.json` at startup.
- Trained model `.pkl` files are expected in `../models/` — run `02_Modelling.ipynb` to generate them.
- If model files are missing, the app falls back to a demo mode using a lightweight Random Forest trained on startup.

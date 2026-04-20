# 🏠 Forecasting Household Energy Consumption

**UMBC DATA 606 — Data Science Capstone | Spring 2026**
**Author:** Kushal Erramilli
**Instructor:** Dr. Chaojie (Jay) Wang

---

## 📌 Project Overview

This project builds a **machine learning pipeline to forecast household electricity consumption** using 4 years of minute-level sensor data from a single French household (UCI dataset). The best model — **LightGBM** — achieves a **41% reduction in RMSE** over a naïve baseline, with predictions accurate to within ±0.31 kW on average.

**Research Questions:**
1. Can historical usage patterns accurately predict near-future electricity consumption?
2. Which time features (hour, season, lag) are most predictive?
3. Do deep learning models (LSTM) outperform gradient boosting for this tabular time-series problem?

---

## 🗂️ Repository Structure

```
UMBC-DATA606-Capstone/
├── data/                          # Raw and processed datasets
│   ├── household_power_consumption.csv   # Raw UCI dataset (~127 MB)
│   ├── minute_clean.parquet              # Cleaned minute-level data
│   ├── hourly_clean.csv                  # Hourly aggregated (used for modelling)
│   └── daily_clean.csv                   # Daily aggregated
├── notebooks/
│   ├── 01_EDA.ipynb                      # Full EDA notebook
│   └── 02_Modelling.ipynb                # Feature engineering + ML models
├── app/
│   ├── streamlit_app.py                  # Interactive Streamlit dashboard
│   └── README.md
├── docs/
│   ├── report.md                         # Final project report
│   └── Resume.md                         # Author resume
├── models/
│   └── feature_cols.json                 # Feature column list for app
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** [UCI ML Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **Size:** ~127 MB, 2,075,259 rows × 9 columns
- **Period:** December 2006 – November 2010 (4 years)
- **Granularity:** 1-minute intervals

---

## 🤖 Models & Results

| Model | RMSE (kW) | R² |
|---|---|---|
| **LightGBM** ⭐ Best | **0.459** | **0.603** |
| XGBoost | 0.459 | 0.602 |
| Random Forest | 0.461 | 0.599 |
| Ridge Regression | 0.495 | 0.538 |
| LSTM | 0.588 | 0.349 |
| Naïve Baseline (lag-24h) | 0.778 | −0.140 |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/UMBC-DATA606-Capstone.git
cd UMBC-DATA606-Capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app/streamlit_app.py

# 4. Or open the notebooks
jupyter notebook notebooks/01_EDA.ipynb
```

---

## 🔗 Links

- 📄 [Final Report](docs/report.md)
- 📓 [EDA Notebook](notebooks/01_EDA.ipynb)
- 🤖 [Modelling Notebook](notebooks/02_Modelling.ipynb)
- 🎤 [YouTube Presentation](#) *(add link after recording)*
- 📊 [PowerPoint Slides](docs/) *(add link after upload)*
- 💼 [LinkedIn](https://www.linkedin.com/in/kushalerramilli/)

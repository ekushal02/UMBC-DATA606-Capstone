# Notebooks

Run in this order:

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Data loading, cleaning, calendar feature engineering, full EDA with Plotly visualisations. Outputs `../data/hourly_clean.csv`. |
| `02_Modelling.ipynb` | Feature engineering (47 features, no leakage), train/test split, model training (Ridge → RF → XGBoost → LightGBM → LSTM), walk-forward CV, residual analysis. Saves models to `../models/`. |

## Requirements

All notebooks run with the packages in `../requirements.txt`. Python 3.10+ recommended.

```bash
pip install -r ../requirements.txt
jupyter notebook
```

## Notes

- Both notebooks use **Plotly** for all visualisations — charts render interactively in Jupyter Lab/Notebook.
- `02_Modelling.ipynb` expects `../data/hourly_clean.csv` to exist — run `01_EDA.ipynb` first.
- LightGBM and XGBoost are optional; the notebook degrades gracefully if they are not installed.
- LSTM training takes ~10–15 minutes on CPU; reduce `epochs` if needed.

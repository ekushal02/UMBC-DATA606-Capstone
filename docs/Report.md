# Forecasting Household Energy Consumption Using Machine Learning

**Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang
**Author:** Kushal Erramilli
**Semester:** Spring 2026

---

🎥 **[YouTube Presentation](#)** *(add link after recording)*
📊 **[PowerPoint Slides](./slides/)** *(add link after upload)*
💻 **[GitHub Repository](https://github.com/<your-username>/UMBC-DATA606-Capstone)**
💼 **[LinkedIn](https://www.linkedin.com/in/kushalerramilli/)**

---

## 1. Title and Author

**Project Title:** Forecasting Household Energy Consumption Using Machine Learning

**Author:** Kushal Erramilli
**Program:** M.S. Data Science, University of Maryland Baltimore County
**Semester:** Spring 2026
**Capstone Instructor:** Dr. Chaojie (Jay) Wang

---

## 2. Background

### What is it about?

This project builds a machine learning forecasting system for residential electricity consumption. Using four years of 1-minute-interval sensor readings from a single French household, the project develops and evaluates a pipeline of models — from a simple persistence baseline through ridge regression and tree ensembles to a deep-learning LSTM — that predict hourly global active power consumption.

### Why does it matter?

Residential electricity consumption accounts for roughly **27% of total U.S. energy use** (EIA, 2023). Accurate near-term forecasting has concrete downstream value:

- **Homeowners** can shift discretionary loads (dishwasher, EV charging, laundry) to off-peak windows, reducing electricity bills without changing comfort.
- **Utilities and grid operators** use aggregate residential forecasts for demand response programs, load balancing, and renewable integration.
- **Smart home systems** can automate appliance scheduling based on predicted peak windows.
- **Data science practitioners** benefit from a worked example showing when deep learning (LSTM) loses to classical gradient boosting on tabular time-series.

The problem is technically non-trivial: energy consumption is driven by overlapping cycles (hourly, daily, weekly, seasonal), highly non-linear interactions (time-of-day × season × occupancy), and unobserved confounders (weather, occupancy) that create an irreducible noise floor.

### Research Questions

1. Can historical usage patterns accurately predict near-future (1-hour-ahead) electricity consumption?
2. Which temporal features — lag values, rolling statistics, or calendar attributes — contribute most to predictive accuracy?
3. Does a deep learning sequence model (LSTM) outperform gradient boosting on this tabular time-series forecasting task?
4. Where (which hours, which seasons) is the model least reliable, and what can homeowners do with that information?

---

## 3. Data

### Data Source

**UCI Machine Learning Repository — Individual Household Electric Power Consumption Dataset**
- URL: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
- Collected by: Georges Hébrail & Alice Bérard, EDF R&D, France

### Data Size and Shape

| File | Rows | Columns | Size |
|------|------|---------|------|
| `household_power_consumption.csv` (raw) | 2,075,259 | 9 | ~127 MB |
| `minute_clean.parquet` (cleaned) | 2,049,280 | 8 | ~30 MB |
| `hourly_clean.csv` (aggregated) | 34,589 | 8 | ~3.5 MB |
| `daily_clean.csv` (aggregated) | 1,442 | 8 | ~200 KB |

### Time Period

**December 16, 2006 → November 26, 2010** (approximately 4 years)
Granularity: 1-minute intervals in raw data; aggregated to hourly for modelling.

### What Does Each Row Represent?

Each row in the raw dataset represents **one minute** of electricity measurements from a single household in Sceaux, France (7 km from Paris).

### Data Dictionary

| Column | Type | Definition | Range |
|--------|------|-----------|-------|
| `Date` | string | Measurement date (DD/MM/YYYY) | Dec 2006 – Nov 2010 |
| `Time` | string | Measurement time (HH:MM:SS) | 00:00:00 – 23:59:00 |
| `Global_active_power` | float | Household total active power (kW) — **target** | 0.076 – 11.122 |
| `Global_reactive_power` | float | Household reactive power (kW) | 0.000 – 1.390 |
| `Voltage` | float | Average voltage (V) | 223.2 – 254.2 |
| `Global_intensity` | float | Average current intensity (A) | 0.200 – 48.400 |
| `Sub_metering_1` | float | Kitchen energy (Wh): microwave, dishwasher, oven | 0 – 88 |
| `Sub_metering_2` | float | Laundry energy (Wh): washer, dryer, fridge, light | 0 – 80 |
| `Sub_metering_3` | float | Water heater + air conditioner energy (Wh) | 0 – 31 |

### Target Variable

`Global_active_power` — the total active power consumed by the household in kilowatts (kW). At hourly granularity, this represents the mean active power for each hour.

**Summary statistics (hourly):**

| Statistic | Value (kW) |
|-----------|-----------|
| Mean | 1.091 |
| Median | 0.600 |
| Std Dev | 1.054 |
| Min | 0.076 |
| Max | 8.891 |
| Skewness | ~1.80 (right-skewed) |

### Missing Values

Exactly **25,979 minute-level rows (~1.25%)** are missing across all sensor columns simultaneously — indicating block-level sensor dropouts rather than random individual-sensor failures. Strategy: **forward-fill** (propagates last valid reading into each gap), with a back-fill guard for leading NaNs.

---

## 4. Exploratory Data Analysis

### 4.1 Data Cleaning and Preparation

The raw dataset required four cleaning steps:

1. **Separator handling:** The raw CSV uses `;` as delimiter, not the standard comma. Loaded with `sep=";"`.
2. **Missing value encoding:** Missing entries are encoded as `?` strings. Loaded with `na_values=["?"]` and coerced to `float64`.
3. **Datetime index:** `Date` and `Time` columns combined into a `DatetimeIndex` (`dayfirst=True` for European format).
4. **Missing value imputation:** Forward-fill followed by back-fill. Forward-fill is appropriate because power consumption doesn't jump discontinuously — the last valid reading is a better estimate than interpolation for block-dropout gaps.

After cleaning: **2,049,280 rows, 0 missing values.** Downsampled to **34,589 hourly rows** via `.resample("H").mean()` for modelling.

### 4.2 Target Variable Distribution

`Global_active_power` is strongly right-skewed (skewness ≈ 1.80):
- Most minutes the household draws < 1 kW (standby loads, lighting)
- Mean (1.09 kW) is nearly double the median (0.60 kW)
- Occasional spikes to 11 kW represent water heater + oven + HVAC running simultaneously

**Decision:** Tree-based models handle right-skewed targets without log-transformation; RMSE and MAE report errors in the original kW units for interpretability.

### 4.3 Temporal Patterns

**Hourly cycle** — the clearest pattern in the data:
- Night trough (2–5 AM): 0.3–0.4 kW (standby only)
- Morning ramp (6–9 AM): rises to ~1.5 kW (breakfast, shower)
- Evening peak (7–9 PM): highest load (~2.0 kW) — dinner, TV, evening appliances
- Weekends follow the same shape but shifted ~1–2 hours later with a higher midday bump

**Weekly cycle** — weekends are ~6% higher than weekdays (people at home all day). The day-of-week effect is real but small relative to seasonal variation.

**Annual cycle** — a clean U-shaped pattern repeats across all four years:
- Winter peaks (December–February): 1.5–2.0 kW average
- Summer troughs (June–August): 0.6–0.8 kW average
- Winter median is approximately **90% higher** than summer median
- The pattern is stable year-over-year with a subtle downward trend from 2007 to 2010

### 4.4 Sub-Meter Contribution Analysis

| Sub-meter | Appliances | Energy Share |
|-----------|-----------|-------------|
| Sub_metering_1 | Kitchen (microwave, dishwasher, oven) | 0.10% |
| Sub_metering_2 | Laundry (washer, dryer, fridge) | 0.12% |
| Sub_metering_3 | Water heater + air conditioner | 0.59% |
| Unmetered / Other | Lighting, TV, computers, EV, etc. | **99.2%** |

The three sub-meters together explain only **0.8% of total consumption**. The dominant driver is unmetered loads — making the sub-meter readings weak direct predictors of global power. However, their temporal patterns (especially HVAC cycling) provide useful usage-rhythm signals and are retained in the feature set.

### 4.5 Correlation Analysis

- `Global_active_power` ↔ `Global_intensity`: very high correlation (r ≈ 0.99) — expected from Ohm's law; both encode the same electrical state
- `Global_active_power` ↔ `Sub_metering_3`: moderate positive (r ≈ 0.38) — HVAC is the largest metered sub-load
- `Voltage`: near-zero correlation with power — voltage is regulated and stable (~240 V), so it carries no predictive signal

---

## 5. Model Training

### 5.1 Feature Engineering (47 features, no leakage)

All features represent information available *strictly before* the target hour `t`. The leakage prevention policy:

| Feature type | Correct (no leakage) |
|---|---|
| Lag features | `GAP[t-1]` via `.shift(1)` |
| Rolling statistics | `.shift(1).rolling(w).mean()` (shift first, then roll) |
| Differencing | `.shift(1).diff(k)` (shift before differencing) |

**Feature categories:**

| Category | Features | Count |
|----------|---------|-------|
| Calendar | hour, dayofweek, month, quarter, is_weekend | 5 |
| Cyclical encoding | sin/cos of hour, dayofweek, month | 6 |
| Lag | 1, 2, 3, 6, 12, 24, 48, 72, 168, 336 h | 10 |
| Rolling stats | mean, std, min, max × windows 3, 6, 12, 24, 48, 168 h | 24 |
| Differencing | 1 h and 24 h | 2 |
| **Total** | | **47** |

Leakage sanity check: max |r| between any feature and target = 0.79 (lag_1h) — well below the leakage threshold of 0.98. ✅

### 5.2 Train / Test Split

**Temporal (not random) split — 80/20:**
- Train: Dec 2006 → Feb 2010 (~27,639 hours)
- Test: Feb 2010 → Nov 2010 (~6,950 hours, covers all four seasons)

Random splitting was explicitly rejected as it would mix future data into training, constituting a form of data leakage.

### 5.3 Models Evaluated

| Model | Key Configuration |
|-------|-------------------|
| Naïve Baseline | Lag-24h persistence (predict same hour yesterday) |
| Ridge Regression | L2 penalty α=0.5, StandardScaler pipeline |
| Random Forest | 300 trees, max_depth=20, max_features=0.6 |
| XGBoost | 1000 estimators, LR=0.03, early stopping (30 rounds) |
| LightGBM | 1000 estimators, LR=0.03, num_leaves=63, early stopping |
| LSTM | 3-layer (128→64→32), 48-hour lookback window, 50 epochs |

### 5.4 Evaluation Metrics

| Metric | Why |
|--------|-----|
| RMSE (kW) | Penalises large errors; critical for peak events |
| MAE (kW) | Average absolute error in interpretable units |
| MAPE (%) | Percentage error for non-technical communication |
| R² | Fraction of variance explained; 0 = no better than mean |

### 5.5 Development Environment

- Python 3.11, Jupyter Notebook
- Packages: pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, plotly
- Hardware: CPU-only (LSTM training: ~12 minutes)

---

## 6. Results

### 6.1 Model Performance Comparison

| Model | RMSE | MAE | MAPE (%) | R² |
|---|---|---|---|---|
| **LightGBM** ⭐ Best | **0.459** | **0.314** | **42.3** | **0.602** |
| XGBoost | 0.460 | 0.316 | 43.3 | 0.601 |
| Random Forest | 0.461 | 0.316 | 43.4 | 0.599 |
| Ridge Regression | 0.495 | 0.349 | 48.9 | 0.538 |
| LSTM | 0.580 | 0.417 | 56.9 | 0.365 |
| Naïve Baseline | 0.778 | 0.520 | 67.6 | −0.140 |

**LightGBM achieves a 41% RMSE reduction vs the naïve baseline.** 80% of predictions fall within ±0.5 kW.

### 6.2 Why LightGBM Wins

LightGBM's leaf-wise growth captures nested conditional interactions (time-of-day × season × recent-lag) more efficiently than other approaches. Its histogram-based split finding makes it computationally fast on this 27k-row training set. The fact that XGBoost and Random Forest are within 0.5% RMSE of LightGBM indicates the dataset's noise floor has been reached.

### 6.3 Why LSTM Underperforms

The LSTM (R²=0.35) is the **worst ML model** despite being the most complex. Root causes:
1. The 47 engineered features already encode all temporal dependencies via lag and rolling features — the LSTM's sequence memory adds nothing that isn't already represented
2. Only ~27k training sequences — deep sequence models typically need orders of magnitude more data
3. No exogenous signals (weather, occupancy) — LSTMs excel when external time-varying drivers modulate the target; here there are none

This is an important negative finding: deep learning is not always the right choice for time-series forecasting.

### 6.4 Feature Importance

Across all three tree models, the **top 5 features** are consistently:
1. `lag_1h` — previous hour's usage (dominant signal)
2. `lag_24h` — same hour yesterday
3. `roll_mean_3h` — 3-hour rolling average
4. `lag_2h` — two hours ago
5. `roll_mean_6h` — 6-hour rolling average

Calendar features (hour, month) rank in the top 15 but below lag features — confirming that recent history is more informative than time-of-day alone.

### 6.5 Walk-Forward Cross-Validation

5-fold walk-forward CV (Random Forest, 100 trees):

| Fold | RMSE |
|------|------|
| 1 | 0.636 |
| 2 | 0.520 |
| 3 | 0.516 |
| 4 | 0.521 |
| 5 | 0.452 |
| **Mean ± Std** | **0.529 ± 0.060** |

The low std (0.060) confirms the model is stable across time. The improving fold-by-fold trend shows performance increases monotonically with more training data.

### 6.6 Error Analysis by Time and Season

**By hour of day (MAE):**
- Night (2–5 AM): 0.13–0.17 kW — easiest; stable standby loads
- Evening peak (21–22 hours): 0.47–0.49 kW — hardest; variable dinner/TV combinations

**By season (MAE):**
- Summer: 0.27 kW — easiest
- Winter: 0.36 kW — hardest; heating behaviour is more variable day-to-day
- Winter median is approximately **36% higher** than summer (vs ~90% in consumption)

---

## 7. Application: Streamlit Dashboard

An interactive dashboard (`app/streamlit_app.py`) provides:
- **Overview** — dataset summary and 4-year time-series
- **EDA Explorer** — hourly patterns, seasonal boxplots, sub-meter breakdown, distribution
- **Forecasting** — live forecast vs actual with residual analysis
- **Model Comparison** — side-by-side metrics, error-by-hour, walk-forward CV results
- **Insights & Recommendations** — actionable guidance for homeowners and future research directions

Run with: `streamlit run app/streamlit_app.py`

---

## 8. Conclusion

### Summary

This project successfully demonstrates that **gradient boosting with carefully engineered temporal features** produces reliable 1-hour-ahead electricity forecasts, achieving a **41% RMSE reduction** over a persistence baseline. LightGBM (RMSE=0.459 kW, R²=0.603) is the recommended deployment model.

The work also produces a clear negative result: the LSTM underperforms all tree-based models, confirming that for tabular time-series tasks with rich feature engineering, deep learning does not automatically outperform classical ML.

### Limitations

- **No weather data**: Temperature and humidity are the primary unobserved drivers of HVAC usage. Their absence is the largest contributor to the residual 40% unexplained variance.
- **Single household**: Findings may not generalise to other homes with different occupancy patterns, climate zones, or appliance inventories.
- **No occupancy signal**: Whether people are home is the dominant unobserved behavioural variable.
- **Static model**: The trained model does not retrain on new data — in production, a sliding-window retraining schedule would be needed.
- **French household**: Consumption patterns (240V grid, climate zone) differ from US households.

### Lessons Learned

- Feature engineering quality matters more than model complexity for tabular time-series
- Leakage prevention (shifting before rolling) is critical and easy to get wrong
- Walk-forward CV is the correct evaluation protocol for time-series — random splits produce overly optimistic results
- LSTM requires much larger datasets and exogenous signals to justify its complexity

### Future Research Directions

1. **Weather integration**: Adding hourly temperature would likely improve R² by 15–20%
2. **Multi-step forecasting**: Extend predictions from 1-hour-ahead to 24-hour-ahead
3. **Anomaly detection**: Flag unusual consumption patterns for homeowner alerts
4. **Probabilistic forecasting**: Output prediction intervals rather than point estimates (e.g., quantile regression forests)
5. **Multi-household generalisation**: Train on multiple households and test transfer learning across homes

---

## 9. References

1. Hébrail, G., & Bérard, A. (2012). *Individual household electric power consumption*. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

5. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

6. U.S. Energy Information Administration (2023). *Residential energy consumption survey (RECS)*. https://www.eia.gov/consumption/residential/

7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

8. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 51–56.
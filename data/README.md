# Data

This folder contains all datasets used in the capstone project.

| File | Description | Size | Rows |
|------|-------------|------|------|
| `household_power_consumption.csv` | Raw UCI dataset — 1-minute intervals, Dec 2006 – Nov 2010 | ~127 MB | 2,075,259 |
| `minute_clean.parquet` | Cleaned minute-level data (missing values filled, datetime index set) | ~30 MB | 2,049,280 |
| `hourly_clean.csv` | Hourly aggregated dataset — primary input for modelling | ~3.5 MB | 34,589 |
| `daily_clean.csv` | Daily aggregated dataset — used for high-level trend visualisation | ~200 KB | 1,442 |

## Raw Data Schema

| Column | Type | Description |
|--------|------|-------------|
| Date | string | Format: DD/MM/YYYY |
| Time | string | Format: HH:MM:SS |
| Global_active_power | float | Household active power (kW) — **target variable** |
| Global_reactive_power | float | Household reactive power (kW) |
| Voltage | float | Average voltage (V) |
| Global_intensity | float | Average current intensity (A) |
| Sub_metering_1 | float | Kitchen energy (Wh) |
| Sub_metering_2 | float | Laundry room energy (Wh) |
| Sub_metering_3 | float | Water heater + AC energy (Wh) |

## Notes

- Missing values (~1.25% of records) are encoded as `?` in the raw file and handled with forward-fill in the cleaning step.
- The raw CSV uses `;` as separator — **not** the standard comma.
- Do not regenerate `hourly_clean.csv` without running `01_EDA.ipynb` end-to-end first.

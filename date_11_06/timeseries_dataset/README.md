# date_11_06 PPG EDA Pipeline

This folder contains a production-style pipeline for the PPG time-series CSV.

## What it generates

- Time continuity check and missing-timestamp report
- Missing value profile and fill strategy
- Time-series plots
- Trend, seasonality, and outlier analysis
- Weekly and monthly boxplots
- Stationarity checks
- ACF and PACF plots
- Feature engineering dataset and feature catalog

## Input

Place the CSV in `data/raw/` or pass a custom path to `main.py`.

## Run

```powershell
python main.py
```

Optional:

```powershell
python main.py --input "C:\\Users\\Administrator\\Downloads\\sakshi_ppg_20260611T074737_len148s.csv"
```

## Output

Artifacts are written under `artifacts/`, with a human-readable summary in `reports/`.

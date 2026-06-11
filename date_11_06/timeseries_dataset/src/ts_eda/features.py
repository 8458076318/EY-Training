from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def infer_start_timestamp_from_filename(path: Path) -> pd.Timestamp | None:
    match = re.search(r"(\d{8}T\d{6})", path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%S")


def add_time_columns(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    out = df.copy()
    start = infer_start_timestamp_from_filename(source_path)
    if start is None:
        start = pd.Timestamp("2026-01-01 00:00:00")
    base_ms = float(out["timestamp_ms"].iloc[0])
    elapsed_ms = out["timestamp_ms"].astype(float) - base_ms
    out["elapsed_ms"] = elapsed_ms
    out["elapsed_seconds"] = elapsed_ms / 1000.0
    out["observed_at"] = start + pd.to_timedelta(elapsed_ms, unit="ms")
    out["calendar_date"] = out["observed_at"].dt.date.astype(str)
    out["calendar_week"] = out["observed_at"].dt.to_period("W").astype(str)
    out["calendar_month"] = out["observed_at"].dt.to_period("M").astype(str)
    out["sample_index"] = np.arange(len(out))
    out["delta_ms"] = out["timestamp_ms"].diff().fillna(0.0)
    return out


def add_signal_features(df: pd.DataFrame, signal_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    windows = [25, 50, 125]
    lags = [1, 5, 25]

    for col in signal_columns:
        s = out[col].astype(float)
        out[f"{col}_diff_1"] = s.diff()
        out[f"{col}_pct_change_1"] = s.pct_change().replace([np.inf, -np.inf], np.nan)
        out[f"{col}_ema_10"] = s.ewm(span=10, adjust=False).mean()
        out[f"{col}_zscore_25"] = (s - s.rolling(25, min_periods=5).mean()) / s.rolling(25, min_periods=5).std(ddof=0)

        for lag in lags:
            out[f"{col}_lag_{lag}"] = s.shift(lag)

        for window in windows:
            roll = s.rolling(window, min_periods=max(3, window // 4))
            out[f"{col}_roll_mean_{window}"] = roll.mean()
            out[f"{col}_roll_std_{window}"] = roll.std(ddof=0)
            out[f"{col}_roll_min_{window}"] = roll.min()
            out[f"{col}_roll_max_{window}"] = roll.max()
            out[f"{col}_roll_median_{window}"] = roll.median()

    return out


def build_feature_catalog(signal_columns: list[str]) -> list[str]:
    catalog = [
        "seq",
        "timestamp_ms",
        "elapsed_ms",
        "elapsed_seconds",
        "observed_at",
        "calendar_date",
        "calendar_week",
        "calendar_month",
        "sample_index",
        "delta_ms",
    ]
    for col in signal_columns:
        catalog.extend(
            [
                col,
                f"{col}_diff_1",
                f"{col}_pct_change_1",
                f"{col}_ema_10",
                f"{col}_zscore_25",
                f"{col}_lag_1",
                f"{col}_lag_5",
                f"{col}_lag_25",
                f"{col}_roll_mean_25",
                f"{col}_roll_std_25",
                f"{col}_roll_min_25",
                f"{col}_roll_max_25",
                f"{col}_roll_median_25",
                f"{col}_roll_mean_50",
                f"{col}_roll_std_50",
                f"{col}_roll_min_50",
                f"{col}_roll_max_50",
                f"{col}_roll_median_50",
                f"{col}_roll_mean_125",
                f"{col}_roll_std_125",
                f"{col}_roll_min_125",
                f"{col}_roll_max_125",
                f"{col}_roll_median_125",
            ]
        )
    return catalog

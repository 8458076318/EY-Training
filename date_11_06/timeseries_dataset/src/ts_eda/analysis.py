from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats


@dataclass(frozen=True)
class StationarityResult:
    test_name: str
    test_statistic: float
    p_value: float
    is_stationary: bool
    n_lags: int
    critical_value_5pct: float


def infer_sampling_interval_ms(df: pd.DataFrame) -> float:
    diffs = df["timestamp_ms"].astype(float).diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1.0
    return float(diffs.round(6).mode().iloc[0])


def find_missing_timestamps(df: pd.DataFrame, interval_ms: float) -> pd.DataFrame:
    diffs = df["timestamp_ms"].astype(float).diff()
    gap_mask = diffs > interval_ms * 1.5
    rows = []
    for idx in np.flatnonzero(gap_mask.to_numpy()):
        prev_ts = float(df.loc[idx - 1, "timestamp_ms"])
        current_ts = float(df.loc[idx, "timestamp_ms"])
        missing_count = max(int(round((current_ts - prev_ts) / interval_ms)) - 1, 0)
        rows.append(
            {
                "row_index": int(idx),
                "previous_timestamp_ms": prev_ts,
                "current_timestamp_ms": current_ts,
                "gap_ms": current_ts - prev_ts,
                "estimated_missing_samples": missing_count,
            }
        )
    return pd.DataFrame(rows)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing,
                "missing_percent": round((missing / total) * 100.0, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def fill_missing_values(df: pd.DataFrame, signal_columns: list[str]) -> tuple[pd.DataFrame, str]:
    filled = df.copy()
    for col in signal_columns:
        if filled[col].isna().any():
            filled[col] = filled[col].interpolate(method="linear", limit_direction="both")
    for col in ["seq", "timestamp_ms"]:
        if filled[col].isna().any():
            filled[col] = filled[col].interpolate(method="linear", limit_direction="both").round().astype(int)

    strategy = (
        "No missing values were detected in the supplied dataset. "
        "If future rows contain gaps, use linear interpolation for dense sensor signals "
        "(`red`, `ir`, `red_corrected`, `ir_corrected`) and preserve `seq` / `timestamp_ms` "
        "after reindexing on the expected sampling interval."
    )
    return filled, strategy


def detect_outliers_iqr(df: pd.DataFrame, signal_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    records = []
    for col in signal_columns:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag_col = f"{col}_outlier_iqr"
        out[flag_col] = (out[col] < lower) | (out[col] > upper)
        records.append(
            {
                "column": col,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(out[flag_col].sum()),
            }
        )
    return out, pd.DataFrame(records)


def estimate_dominant_period_samples(series: pd.Series, sampling_hz: float) -> int:
    values = series.dropna().astype(float).to_numpy()
    if len(values) < 16:
        return max(8, int(round(sampling_hz)))
    centered = values - values.mean()
    freqs, power = signal.periodogram(centered, fs=sampling_hz)
    valid = freqs > 0
    if not valid.any():
        return max(8, int(round(sampling_hz)))
    peak_freq = freqs[valid][np.argmax(power[valid])]
    if peak_freq <= 0:
        return max(8, int(round(sampling_hz)))
    return max(8, int(round(sampling_hz / peak_freq)))


def moving_average_decomposition(series: pd.Series, period: int) -> pd.DataFrame:
    s = series.astype(float)
    trend = s.rolling(window=period, center=True, min_periods=max(3, period // 3)).mean()
    detrended = s - trend
    seasonal_pattern = detrended.groupby(np.arange(len(detrended)) % period).transform("mean")
    seasonal_pattern = seasonal_pattern - seasonal_pattern.mean()
    residual = s - trend - seasonal_pattern
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal_pattern, "residual": residual})


def adf_like_stationarity(series: pd.Series, max_lags: int = 5) -> StationarityResult:
    values = series.dropna().astype(float).to_numpy()
    if len(values) < max_lags + 10:
        return StationarityResult("ADF-like", float("nan"), float("nan"), False, max_lags, -2.86)

    dy = np.diff(values)
    y_lag = values[:-1]

    rows = len(dy) - max_lags
    y_target = dy[max_lags:]
    columns = [np.ones(rows), y_lag[max_lags:]]
    for lag in range(1, max_lags + 1):
        columns.append(dy[max_lags - lag : len(dy) - lag])
    x = np.column_stack(columns)

    beta, *_ = np.linalg.lstsq(x, y_target, rcond=None)
    residuals = y_target - x @ beta
    n_obs, n_params = x.shape
    dof = max(n_obs - n_params, 1)
    mse = float((residuals @ residuals) / dof)
    cov = mse * np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.diag(cov))

    gamma = beta[1]
    gamma_se = stderr[1] if stderr[1] > 0 else np.nan
    t_stat = float(gamma / gamma_se) if gamma_se and not np.isnan(gamma_se) else float("nan")
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof))) if not np.isnan(t_stat) else float("nan")
    critical_value = -2.86
    stationary = bool(not np.isnan(t_stat) and t_stat < critical_value)

    return StationarityResult("ADF-like", t_stat, p_value, stationary, max_lags, critical_value)


def acf_values(series: pd.Series, nlags: int = 40) -> pd.DataFrame:
    values = series.dropna().astype(float).to_numpy()
    values = values - values.mean()
    denom = float(np.dot(values, values))
    rows = []
    for lag in range(nlags + 1):
        if lag == 0:
            corr = 1.0
        elif lag >= len(values):
            corr = np.nan
        else:
            corr = float(np.dot(values[:-lag], values[lag:]) / denom) if denom else np.nan
        rows.append({"lag": lag, "acf": corr})
    return pd.DataFrame(rows)


def pacf_values(series: pd.Series, nlags: int = 40) -> pd.DataFrame:
    values = series.dropna().astype(float).to_numpy()
    rows = [{"lag": 0, "pacf": 1.0}]
    for lag in range(1, nlags + 1):
        if lag >= len(values):
            rows.append({"lag": lag, "pacf": np.nan})
            continue
        y = values[lag:]
        x_cols = [np.ones(len(y))]
        for i in range(1, lag + 1):
            x_cols.append(values[lag - i : len(values) - i])
        x = np.column_stack(x_cols)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        rows.append({"lag": lag, "pacf": float(beta[-1])})
    return pd.DataFrame(rows)

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks


sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_missing_summary(missing_df: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=missing_df, x="column", y="missing_percent", ax=ax, color="#3B82F6")
    ax.set_title("Missing Value Percentage by Column")
    ax.set_ylabel("Missing %")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, path)


def plot_series(df: pd.DataFrame, columns: list[str], path: Path) -> str:
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    axes = np.atleast_1d(axes)
    x = df["observed_at"] if "observed_at" in df.columns else df["timestamp_ms"]
    for ax, col in zip(axes, columns):
        ax.plot(x, df[col], lw=0.8, color="#0F766E")
        ax.set_title(col)
        ax.set_ylabel("value")
    axes[-1].set_xlabel("time")
    return _save(fig, path)


def plot_superimposed_signals(df: pd.DataFrame, signal_a: str, signal_b: str, path: Path, title: str = "Superimposed Signals") -> str:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = df["observed_at"] if "observed_at" in df.columns else df["timestamp_ms"]
    ax.plot(x, df[signal_a], lw=0.8, color="#EF4444", label=signal_a)
    ax.plot(x, df[signal_b], lw=0.8, color="#2563EB", label=signal_b, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("signal value")
    ax.legend()
    return _save(fig, path)


def plot_normalized_signals_with_peaks(
    df: pd.DataFrame,
    signal_a: str,
    signal_b: str,
    path: Path,
    title: str = "Normalized Signals with Peaks",
    prominence: float = 0.5,
    distance: int = 18,
) -> str:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = df["observed_at"] if "observed_at" in df.columns else df["timestamp_ms"]

    series_a = df[signal_a].astype(float)
    series_b = df[signal_b].astype(float)
    mean_a = float(series_a.mean())
    std_a = float(series_a.std(ddof=0))
    mean_b = float(series_b.mean())
    std_b = float(series_b.std(ddof=0))

    norm_a = (series_a - mean_a) / std_a
    norm_b = (series_b - mean_b) / std_b

    peaks_a, _ = find_peaks(norm_a.to_numpy(), prominence=prominence, distance=distance)
    peaks_b, _ = find_peaks(norm_b.to_numpy(), prominence=prominence, distance=distance)

    ax.plot(x, norm_a, lw=0.9, color="#EF4444", label=f"{signal_a} (z-score)")
    ax.plot(x, norm_b, lw=0.9, color="#2563EB", label=f"{signal_b} (z-score)", alpha=0.85)

    ax.axhline(0.0, color="#111827", lw=1.0, ls="--", alpha=0.75, label="mean (z-score = 0)")
    ax.axhline(1.0, color="#6B7280", lw=0.9, ls=":", alpha=0.75, label="+1 std")
    ax.axhline(-1.0, color="#6B7280", lw=0.9, ls=":", alpha=0.75, label="-1 std")

    if len(peaks_a):
        ax.scatter(x.iloc[peaks_a], norm_a.iloc[peaks_a], color="#B91C1C", s=28, marker="o", label=f"{signal_a} peaks")
    if len(peaks_b):
        ax.scatter(x.iloc[peaks_b], norm_b.iloc[peaks_b], color="#1D4ED8", s=28, marker="^", label=f"{signal_b} peaks")

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("normalized signal value")
    stats_text = (
        f"{signal_a}: mean={mean_a:.2f}, std={std_a:.2f}\n"
        f"{signal_b}: mean={mean_b:.2f}, std={std_b:.2f}"
    )
    ax.text(
        0.01,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#D1D5DB"),
    )
    ax.legend(ncols=2, loc="upper right")
    return _save(fig, path)


def plot_decomposition(decomp_df: pd.DataFrame, path: Path, title: str) -> str:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    x = np.arange(len(decomp_df))
    axes[0].plot(x, decomp_df["observed"], color="#111827", lw=0.8)
    axes[0].set_title(f"{title} - observed")
    axes[1].plot(x, decomp_df["trend"], color="#2563EB", lw=1.0)
    axes[1].set_title("trend")
    axes[2].plot(x, decomp_df["seasonal"], color="#DC2626", lw=0.8)
    axes[2].set_title("seasonal")
    axes[3].plot(x, decomp_df["residual"], color="#6B7280", lw=0.8)
    axes[3].set_title("residual")
    axes[3].set_xlabel("sample index")
    return _save(fig, path)


def plot_boxplots(df: pd.DataFrame, group_col: str, value_cols: list[str], path: Path, title: str) -> str:
    long_df = df[[group_col] + value_cols].melt(id_vars=group_col, var_name="signal", value_name="value")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=long_df, x=group_col, y="value", hue="signal", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, path)


def plot_stationarity(df: pd.DataFrame, series_col: str, path: Path, window: int = 250) -> str:
    s = df[series_col].astype(float)
    roll_mean = s.rolling(window, min_periods=max(10, window // 4)).mean()
    roll_std = s.rolling(window, min_periods=max(10, window // 4)).std(ddof=0)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(s.index, s, color="#111827", lw=0.6)
    axes[0].set_title(series_col)
    axes[1].plot(roll_mean.index, roll_mean, color="#2563EB", lw=1.0)
    axes[1].set_title("rolling mean")
    axes[2].plot(roll_std.index, roll_std, color="#DC2626", lw=1.0)
    axes[2].set_title("rolling std")
    axes[2].set_xlabel("sample index")
    return _save(fig, path)


def plot_acf_pacf(acf_df: pd.DataFrame, pacf_df: pd.DataFrame, path: Path, title: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].stem(acf_df["lag"], acf_df["acf"], basefmt=" ")
    axes[0].set_title(f"{title} - ACF")
    axes[1].stem(pacf_df["lag"], pacf_df["pacf"], basefmt=" ")
    axes[1].set_title(f"{title} - PACF")
    axes[1].set_xlabel("lag")
    return _save(fig, path)


def plot_outliers(df: pd.DataFrame, signal_col: str, outlier_flag_col: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = df["observed_at"] if "observed_at" in df.columns else df.index
    ax.plot(x, df[signal_col], color="#0F766E", lw=0.7, label=signal_col)
    flagged = df[df[outlier_flag_col]]
    if not flagged.empty:
        ax.scatter(flagged["observed_at"], flagged[signal_col], color="#DC2626", s=16, label="outlier")
    ax.set_title(f"Outlier check - {signal_col}")
    ax.legend()
    return _save(fig, path)

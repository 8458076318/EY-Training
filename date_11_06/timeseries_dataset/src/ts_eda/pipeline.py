from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from date_11_06.timeseries_dataset.src.ts_eda.analysis import (
    acf_values,
    adf_like_stationarity,
    detect_outliers_iqr,
    estimate_dominant_period_samples,
    fill_missing_values,
    find_missing_timestamps,
    infer_sampling_interval_ms,
    missing_value_summary,
    moving_average_decomposition,
    pacf_values,
)
from date_11_06.timeseries_dataset.src.ts_eda.config import build_paths
from date_11_06.timeseries_dataset.src.ts_eda.features import add_signal_features, add_time_columns, build_feature_catalog
from date_11_06.timeseries_dataset.src.ts_eda.io import ensure_output_dirs, load_ppg_csv
from date_11_06.timeseries_dataset.src.ts_eda.plots import (
    plot_acf_pacf,
    plot_boxplots,
    plot_decomposition,
    plot_missing_summary,
    plot_outliers,
    plot_normalized_signals_with_peaks,
    plot_series,
    plot_superimposed_signals,
    plot_stationarity,
)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write_df(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def run_pipeline(input_csv: Path | None = None, output_dir: Path | None = None) -> dict[str, str]:
    paths = build_paths(input_csv=input_csv, output_dir=output_dir)
    ensure_output_dirs(paths.artifacts_dir, paths.reports_dir, paths.logs_dir, paths.processed_dir)

    raw = load_ppg_csv(paths.input_csv)
    enriched = add_time_columns(raw, paths.input_csv)

    interval_ms = infer_sampling_interval_ms(enriched)
    sampling_hz = 1000.0 / interval_ms if interval_ms else 1.0
    missing_timestamps = find_missing_timestamps(enriched, interval_ms)
    missing_summary = missing_value_summary(enriched)
    filled, strategy_text = fill_missing_values(enriched, ["red", "ir", "red_corrected", "ir_corrected"])

    signal_columns = ["red", "ir", "red_corrected", "ir_corrected"]
    feature_frame = add_signal_features(filled, signal_columns)
    feature_catalog = build_feature_catalog(signal_columns)

    feature_columns_path = paths.artifacts_dir / "01_profile" / "feature_engineering_columns.md"
    missing_strategy_path = paths.artifacts_dir / "02_missing" / "missing_strategy.md"
    profile_json_path = paths.artifacts_dir / "01_profile" / "data_profile.json"
    missing_summary_csv = paths.artifacts_dir / "02_missing" / "missing_summary.csv"
    missing_gaps_csv = paths.artifacts_dir / "02_missing" / "missing_timestamps.csv"
    feature_data_csv = paths.processed_dir / "ppg_engineered_dataset.csv"

    _write_text(
        feature_columns_path,
        "# Feature Engineering Columns\n\n" + "\n".join(f"- {col}" for col in feature_catalog),
    )
    _write_text(missing_strategy_path, "# Missing Value Strategy\n\n" + strategy_text + "\n")
    _write_df(missing_summary, missing_summary_csv)
    _write_df(missing_timestamps, missing_gaps_csv)
    _write_df(feature_frame, feature_data_csv)

    dominant_period = estimate_dominant_period_samples(feature_frame["ir_corrected"], sampling_hz)
    decomp = moving_average_decomposition(feature_frame["ir_corrected"], dominant_period)
    outlier_frame, outlier_summary = detect_outliers_iqr(feature_frame, signal_columns)

    decomp_df = decomp.copy()
    decomp_df["observed_at"] = feature_frame["observed_at"].values

    stationarity = adf_like_stationarity(feature_frame["ir_corrected"])
    acf_df = acf_values(feature_frame["ir_corrected"], nlags=min(60, len(feature_frame) // 2 - 1))
    pacf_df = pacf_values(feature_frame["ir_corrected"], nlags=min(60, len(feature_frame) // 2 - 1))

    artifact_paths = {
        "time_series_plot": plot_series(
            feature_frame,
            ["red", "ir", "red_corrected", "ir_corrected"],
            paths.artifacts_dir / "03_series_plots" / "time_series.png",
        ),
        "superimposed_red_ir": plot_superimposed_signals(
            feature_frame,
            "red",
            "ir",
            paths.artifacts_dir / "03_series_plots" / "red_ir_superimposed.png",
            "Red and IR Superimposed",
        ),
        "normalized_red_ir_peaks": plot_normalized_signals_with_peaks(
            feature_frame,
            "red",
            "ir",
            paths.artifacts_dir / "03_series_plots" / "red_ir_normalized_peaks.png",
            "Normalized Red and IR with Peak Positions",
        ),
        "decomposition_plot": plot_decomposition(
            decomp_df,
            paths.artifacts_dir / "04_decomposition" / "ir_corrected_decomposition.png",
            "ir_corrected",
        ),
        "weekly_boxplot": plot_boxplots(
            feature_frame,
            "calendar_week",
            ["red_corrected", "ir_corrected"],
            paths.artifacts_dir / "05_boxplots" / "weekly_boxplot.png",
            "Weekly Boxplots",
        ),
        "monthly_boxplot": plot_boxplots(
            feature_frame,
            "calendar_month",
            ["red_corrected", "ir_corrected"],
            paths.artifacts_dir / "05_boxplots" / "monthly_boxplot.png",
            "Monthly Boxplots",
        ),
        "stationarity_plot": plot_stationarity(
            feature_frame,
            "ir_corrected",
            paths.artifacts_dir / "06_stationarity" / "rolling_stationarity.png",
        ),
        "acf_pacf_plot": plot_acf_pacf(
            acf_df,
            pacf_df,
            paths.artifacts_dir / "07_acf_pacf" / "acf_pacf.png",
            "ir_corrected",
        ),
        "outlier_red_plot": plot_outliers(
            outlier_frame,
            "red_corrected",
            "red_corrected_outlier_iqr",
            paths.artifacts_dir / "04_decomposition" / "red_corrected_outliers.png",
        ),
        "outlier_ir_plot": plot_outliers(
            outlier_frame,
            "ir_corrected",
            "ir_corrected_outlier_iqr",
            paths.artifacts_dir / "04_decomposition" / "ir_corrected_outliers.png",
        ),
    }

    _write_df(outlier_summary, paths.artifacts_dir / "04_decomposition" / "outlier_summary.csv")
    _write_df(acf_df, paths.artifacts_dir / "07_acf_pacf" / "acf_values.csv")
    _write_df(pacf_df, paths.artifacts_dir / "07_acf_pacf" / "pacf_values.csv")

    profile = {
        "input_csv": str(paths.input_csv),
        "rows": int(len(feature_frame)),
        "columns": list(feature_frame.columns),
        "sampling_interval_ms": interval_ms,
        "sampling_frequency_hz": sampling_hz,
        "dominant_period_samples": int(dominant_period),
        "missing_timestamp_groups": int(len(missing_timestamps)),
        "missing_rows_total": int(missing_summary["missing_count"].sum()),
        "stationarity_test": {
            "name": stationarity.test_name,
            "statistic": stationarity.test_statistic,
            "p_value": stationarity.p_value,
            "is_stationary": stationarity.is_stationary,
            "critical_value_5pct": stationarity.critical_value_5pct,
            "lags": stationarity.n_lags,
        },
        "outlier_counts": outlier_summary.set_index("column")["outlier_count"].to_dict(),
    }

    profile_json_path.parent.mkdir(parents=True, exist_ok=True)
    profile_json_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    report = f"""# PPG EDA Run Report

## Input

- CSV: `{paths.input_csv}`
- Rows: `{len(feature_frame)}`
- Columns: `{len(feature_frame.columns)}`
- Sampling interval: `{interval_ms:.3f} ms`
- Sampling frequency: `{sampling_hz:.3f} Hz`
- Dominant period: `{dominant_period}` samples

## Time Continuity

- Missing timestamp groups: `{len(missing_timestamps)}`
- Estimated missing rows from gaps: `{int(missing_timestamps['estimated_missing_samples'].sum()) if not missing_timestamps.empty else 0}`

## Missing Values

- Total missing cells: `{int(missing_summary['missing_count'].sum())}`
- Strategy: linear interpolation for future gaps in dense sensor channels, after reindexing to the expected cadence

## Stationarity

- Test: `{stationarity.test_name}`
- Statistic: `{stationarity.test_statistic:.6f}`
- P-value: `{stationarity.p_value:.6f}`
- Stationary: `{stationarity.is_stationary}`

## Outliers

- `red`: `{int(outlier_summary.loc[outlier_summary['column'] == 'red', 'outlier_count'].iloc[0])}`
- `ir`: `{int(outlier_summary.loc[outlier_summary['column'] == 'ir', 'outlier_count'].iloc[0])}`
- `red_corrected`: `{int(outlier_summary.loc[outlier_summary['column'] == 'red_corrected', 'outlier_count'].iloc[0])}`
- `ir_corrected`: `{int(outlier_summary.loc[outlier_summary['column'] == 'ir_corrected', 'outlier_count'].iloc[0])}`

## Artifacts

{chr(10).join(f"- {name}: `{path}`" for name, path in artifact_paths.items())}
"""
    report_path = paths.reports_dir / "ppg_eda_report.md"
    _write_text(report_path, report)

    manifest = {
        "input_csv": str(paths.input_csv),
        "output_root": str(paths.root),
        "artifacts": artifact_paths,
        "report": str(report_path),
        "feature_dataset": str(feature_data_csv),
    }
    _write_text(paths.artifacts_dir / "run_manifest.json", json.dumps(manifest, indent=2))

    return {
        "input_csv": str(paths.input_csv),
        "artifacts_dir": str(paths.artifacts_dir),
        "report_path": str(report_path),
    }

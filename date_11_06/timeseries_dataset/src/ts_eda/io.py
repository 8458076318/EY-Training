from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["seq", "timestamp_ms", "red", "ir", "red_corrected", "ir_corrected"]


def load_ppg_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df.loc[:, REQUIRED_COLUMNS].copy()


def ensure_output_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

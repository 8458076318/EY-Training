from __future__ import annotations

import argparse
from pathlib import Path

from date_11_06.timeseries_dataset.src.ts_eda.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate EDA artifacts for the PPG time series.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the input CSV. Defaults to data/raw/sakshi_ppg_20260611T074737_len148s.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom output directory for artifacts and reports.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_pipeline(input_csv=Path(args.input) if args.input else None, output_dir=Path(args.output_dir) if args.output_dir else None)

    print("Pipeline completed.")
    print(f"Input: {result['input_csv']}")
    print(f"Artifacts: {result['artifacts_dir']}")
    print(f"Report: {result['report_path']}")
    return 0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    input_csv: Path
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    logs_dir: Path


def build_paths(input_csv: Path | None = None, output_dir: Path | None = None) -> PipelinePaths:
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"

    if input_csv is None:
        candidates = sorted(raw_dir.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV found in {raw_dir}. Place the input file there or pass --input."
            )
        input_csv = candidates[0]
    else:
        input_csv = input_csv.resolve()

    base_output = output_dir.resolve() if output_dir else root
    artifacts_dir = base_output / "artifacts"
    reports_dir = base_output / "reports"
    logs_dir = base_output / "logs"

    return PipelinePaths(
        root=root,
        input_csv=input_csv,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )

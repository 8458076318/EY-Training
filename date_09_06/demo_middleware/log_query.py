from __future__ import annotations

import json
from pathlib import Path

from .settings import LOG_FILE


def query_log_file(
    correlation_id: str,
    min_latency_ms: float = 200,
    limit: int = 50,
    log_file: Path = LOG_FILE,
) -> list[dict]:
    matches: list[dict] = []
    if not log_file.exists():
        return matches

    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("correlation_id") != correlation_id:
                continue

            latency_ms = event.get("latency_ms")
            if latency_ms is None or float(latency_ms) <= float(min_latency_ms):
                continue

            matches.append(event)
            if len(matches) >= limit:
                break

    return matches


from __future__ import annotations

from typing import Any

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
except Exception:  # pragma: no cover - optional dependency fallback
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoOpGauge:
        def __init__(self, *_: Any, **__: Any) -> None:
            self._value = 0.0

        def set(self, value: float) -> None:
            self._value = float(value)

    def generate_latest() -> bytes:  # type: ignore[override]
        return b"# prometheus_client not installed\n"

    def Gauge(*args: Any, **kwargs: Any) -> _NoOpGauge:  # type: ignore[misc]
        return _NoOpGauge(*args, **kwargs)


DLQ_DEPTH_GAUGE = Gauge("payments_dlq_depth", "Current depth of the payments DLQ")

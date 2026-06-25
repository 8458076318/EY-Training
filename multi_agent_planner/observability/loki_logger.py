"""
Loki log handler - ships structured JSON logs to Grafana Loki.
"""
import json
import logging
import time
import threading
from collections import deque
import httpx
from config.settings import get_settings

settings = get_settings()


class LokiHandler(logging.Handler):
    def __init__(self, url: str, labels=None, batch_interval: float = 2.0):
        super().__init__()
        self.url = url.rstrip("/") + "/loki/api/v1/push"
        self.labels = labels or {"app": "multi-agent-planner", "env": settings.ENV}
        self.queue = deque(maxlen=1000)
        self._stop = threading.Event()
        self._interval = batch_interval
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": str(int(time.time_ns())),
                "line": json.dumps({
                    "level": record.levelname, "logger": record.name,
                    "message": self.format(record), "module": record.module,
                }),
            }
            self.queue.append(entry)
        except Exception:
            self.handleError(record)

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            self._send_batch()

    def _send_batch(self) -> None:
        if not self.queue:
            return
        entries = []
        while self.queue:
            entries.append(self.queue.popleft())
        payload = {"streams": [{"stream": self.labels, "values": [[e["ts"], e["line"]] for e in entries]}]}
        try:
            httpx.post(self.url, json=payload, timeout=5)
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        self._send_batch()
        super().close()


def setup_loki(loki_url=None) -> None:
    url = loki_url or getattr(settings, "LOKI_URL", "http://localhost:3100")
    handler = LokiHandler(url=url)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)

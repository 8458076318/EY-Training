from __future__ import annotations

import os


MAIN_QUEUE = "payments"
DLQ_QUEUE = "payments.dlq"

QUEUE_TTL_MS = int(os.getenv("PAYMENTS_QUEUE_TTL_MS", "86400000"))
MAX_DELIVERIES = int(os.getenv("PAYMENTS_MAX_DELIVERIES", "3"))
WORKER_POLL_INTERVAL = float(os.getenv("PAYMENTS_WORKER_POLL_INTERVAL", "0.25"))
BACKGROUND_WORKER_ENABLED = os.getenv("PAYMENTS_BACKGROUND_WORKER", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BROKER_BACKEND = os.getenv("PAYMENTS_BROKER", "memory").strip().lower()
DLQ_ALERT_THRESHOLD = int(os.getenv("PAYMENTS_DLQ_ALERT_THRESHOLD", "10"))
DLQ_ALERT_WINDOW_SECONDS = int(os.getenv("PAYMENTS_DLQ_ALERT_WINDOW_SECONDS", "30"))
PROCESSOR_FAIL_MODE = os.getenv("PAYMENTS_PROCESSOR_FAIL_MODE", "random").strip().lower()

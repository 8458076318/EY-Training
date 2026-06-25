from __future__ import annotations

from pathlib import Path

APP_TITLE = "EY Payment API"
APP_VERSION = "2.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = Path(__file__).resolve().with_name("middleware.ndjson")

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 100

FRAUD_RETRY_ATTEMPTS = 5
FRAUD_RETRY_MIN_SECONDS = 0.1
FRAUD_RETRY_MAX_SECONDS = 2
FRAUD_RETRY_MULTIPLIER = 1

FRAUD_BREAKER_FAIL_MAX = 3
FRAUD_BREAKER_RESET_TIMEOUT_SECONDS = 30

REQUEST_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)


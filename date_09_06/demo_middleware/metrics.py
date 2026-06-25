from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .settings import REQUEST_LATENCY_BUCKETS

REQUEST_COUNT = Counter(
    "demo_http_requests_total",
    "Total HTTP requests handled by the demo app",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "demo_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path", "status"],
    buckets=REQUEST_LATENCY_BUCKETS,
)
PAYMENT_AMOUNT = Histogram(
    "payment_amount_gbp",
    "Payment amount in GBP",
    buckets=[10, 50, 100, 500, 1000, 5000, 10000],
)
ERROR_COUNT = Counter(
    "payment_errors_total",
    "Total payment processing errors",
    ["error_type"],
)
RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total",
    "Total requests rejected by the rate limiter",
    ["client_ip"],
)
CIRCUIT_BREAKER_TRIPS = Counter(
    "circuit_breaker_trips_total",
    "Total times the circuit breaker moved into the OPEN state",
    ["breaker"],
)


def record_request(method: str, path: str, status: int, elapsed_seconds: float) -> None:
    labels = {
        "method": method,
        "path": path,
        "status": str(status),
    }
    REQUEST_COUNT.labels(**labels).inc()
    REQUEST_LATENCY.labels(**labels).observe(elapsed_seconds)


def render_metrics() -> bytes:
    return generate_latest()


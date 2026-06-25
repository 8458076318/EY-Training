from __future__ import annotations

TASK_LINES = (
    (
        "Task A",
        "Rate-limiting middleware",
        "Sliding-window in-memory limiter, 429 response, Retry-After header, and rate_limit_hits_total counter.",
    ),
    (
        "Task B",
        "Correlation ID propagation",
        "ContextVar storage plus httpx request hook so downstream calls receive X-Correlation-Id.",
    ),
    (
        "Task C",
        "Prometheus + Grafana dashboard",
        "Metrics endpoint plus Grafana dashboard JSON; Grafana Cloud remote-write still needs your stack credentials.",
    ),
    (
        "Task D",
        "Structured log aggregation",
        "structlog writes newline-delimited JSON to middleware.ndjson, with a local search helper for latency/correlation queries.",
    ),
    (
        "Bonus",
        "Retry + circuit breaker",
        "tenacity retry for flaky calls and pybreaker with a breaker-open counter for dashboarding.",
    ),
)


def print_task_summary() -> None:
    for label, title, detail in TASK_LINES:
        print(f"[{label}] {title}: {detail}")


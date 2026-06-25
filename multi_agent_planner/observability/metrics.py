"""
Prometheus metrics. Import these anywhere to record telemetry.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from config.settings import get_settings

settings = get_settings()

agent_call_counter = Counter(
    "agent_calls_total",
    "Number of agent invocations",
    ["agent"],
)

agent_latency = Histogram(
    "agent_latency_seconds",
    "Agent response latency",
    ["agent"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

api_request_counter = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

reminders_sent = Counter(
    "reminders_sent_total",
    "SMS reminders dispatched",
    ["provider"],
)

active_users = Gauge("active_users", "Currently active users")


def start_metrics_server() -> None:
    start_http_server(settings.PROMETHEUS_PORT)

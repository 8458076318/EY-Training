"""Prometheus custom metrics."""
from prometheus_client import Counter, Histogram, Gauge

AGENT_LATENCY = Histogram(
    "agent_run_seconds", "Time taken by each agent",
    ["agent"], buckets=[0.5, 1, 2, 5, 10, 30, 60]
)
AGENT_ERRORS = Counter(
    "agent_errors_total", "Total agent errors", ["agent"]
)
LLM_CALLS = Counter(
    "llm_api_calls_total", "LLM API calls", ["provider", "status"]
)
PLAN_GENERATED = Counter("plans_generated_total", "Weekly plans generated")
PUSH_SENT = Counter(
    "push_notifications_sent_total", "FCM push notifications sent", ["status"]
)
NOTIFICATIONS_READ = Counter(
    "notifications_read_total", "In-app notifications marked as read"
)
ACTIVE_USERS = Gauge("active_users_current", "Currently active users")

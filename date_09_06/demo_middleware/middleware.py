from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .correlation import get_or_create_correlation_id
from .metrics import ERROR_COUNT, RATE_LIMIT_HITS, record_request
from .settings import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

_rate_limit_state = defaultdict(deque)
_rate_limit_lock = Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, log):
        super().__init__(app)
        self.log = log

    async def dispatch(self, request: Request, call_next):
        client_ip = _client_ip(request)
        correlation_id = get_or_create_correlation_id(request)

        now = time.monotonic()
        with _rate_limit_lock:
            request_times = _rate_limit_state[client_ip]
            cutoff = now - RATE_LIMIT_WINDOW_SECONDS
            while request_times and request_times[0] < cutoff:
                request_times.popleft()

            if len(request_times) >= RATE_LIMIT_MAX_REQUESTS:
                retry_after = max(
                    1,
                    int(RATE_LIMIT_WINDOW_SECONDS - (now - request_times[0])),
                )
                RATE_LIMIT_HITS.labels(client_ip=client_ip).inc()
                self.log.warning(
                    "request.rate_limited",
                    path=request.url.path,
                    method=request.method,
                    client_ip=client_ip,
                    retry_after=retry_after,
                    correlation_id=correlation_id,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after_seconds": retry_after,
                        "correlation_id": correlation_id,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-Correlation-Id": correlation_id,
                    },
                )

            request_times.append(now)

        started = time.perf_counter()
        self.log.info(
            "request.started",
            path=request.url.path,
            method=request.method,
            correlation_id=correlation_id,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            ERROR_COUNT.labels(error_type=type(exc).__name__).inc()
            self.log.exception(
                "request.failed",
                path=request.url.path,
                method=request.method,
                correlation_id=correlation_id,
            )
            raise

        elapsed_seconds = time.perf_counter() - started
        record_request(request.method, request.url.path, response.status_code, elapsed_seconds)
        self.log.info(
            "request.completed",
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(elapsed_seconds * 1000, 2),
            correlation_id=correlation_id,
        )
        response.headers["X-Correlation-Id"] = correlation_id
        return response


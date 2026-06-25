import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from observability.metrics import api_request_counter

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000, 2)
        api_request_counter.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        ).inc()
        logger.info("%s %s %s %.2fms", request.method, request.url.path, response.status_code, elapsed)
        return response

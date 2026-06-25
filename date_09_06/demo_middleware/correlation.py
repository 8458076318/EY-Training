from __future__ import annotations

import uuid
from contextvars import ContextVar

import httpx
from fastapi import Request

CORRELATION_ID: ContextVar[str] = ContextVar("correlation_id", default="")


def get_or_create_correlation_id(request: Request) -> str:
    correlation_id = request.headers.get("X-Correlation-Id", "").strip()
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    CORRELATION_ID.set(correlation_id)
    return correlation_id


async def inject_correlation_id(request: httpx.Request) -> None:
    correlation_id = CORRELATION_ID.get()
    if correlation_id:
        request.headers["X-Correlation-Id"] = correlation_id


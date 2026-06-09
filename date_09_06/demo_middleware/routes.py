from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import Response

from .correlation import CORRELATION_ID
from .log_query import query_log_file
from .metrics import CONTENT_TYPE_LATEST, PAYMENT_AMOUNT, render_metrics
from .services import assess_fraud, call_mock_downstream_echo

router = APIRouter()


@router.get("/health/live")
async def liveness():
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness():
    return {"status": "ready", "db": "ok", "mq": "ok"}


@router.post("/payments")
async def create_payment(request: Request):
    body = await request.json()
    amount = float(body.get("amount", 0))
    currency = body.get("currency", "GBP")
    PAYMENT_AMOUNT.observe(amount)

    downstream_result = await call_mock_downstream_echo(
        app=request.app,
        amount=amount,
        currency=currency,
    )
    fraud_result = await assess_fraud(body)

    return {
        "payment_id": str(uuid.uuid4()),
        "status": "accepted",
        "downstream": downstream_result,
        "fraud": fraud_result,
        **body,
    }


@router.get("/mock-downstream/echo")
async def mock_downstream_echo(request: Request):
    return {
        "received_correlation_id": request.headers.get("X-Correlation-Id"),
        "received_path": request.url.path,
    }


@router.post("/fraud/check")
async def fraud_check(request: Request):
    body = await request.json()
    return await assess_fraud(body)


@router.get("/metrics")
async def metrics():
    return Response(render_metrics(), media_type=CONTENT_TYPE_LATEST)


@router.get("/admin/logs/search")
async def search_logs(
    correlation_id: str,
    min_latency_ms: float = 200,
    limit: int = 50,
):
    return {
        "correlation_id": correlation_id,
        "min_latency_ms": min_latency_ms,
        "matches": query_log_file(correlation_id, min_latency_ms=min_latency_ms, limit=limit),
    }


@router.get("/admin/correlation-id")
async def current_correlation_id():
    return {"correlation_id": CORRELATION_ID.get()}


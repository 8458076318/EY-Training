from __future__ import annotations

import asyncio
import contextlib

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

from .common import log
from .container import broker
from .metrics import CONTENT_TYPE_LATEST, generate_latest
from .models import PaymentRequest
from .settings import BACKGROUND_WORKER_ENABLED, DLQ_QUEUE, MAIN_QUEUE
from .worker import dlq_alert_loop, worker_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app.startup", backend=broker.name)
    await broker.connect()

    app.state.worker_task = None
    app.state.dlq_alert_task = None
    app.state.worker_pause = False

    if BACKGROUND_WORKER_ENABLED:
        app.state.worker_task = asyncio.create_task(worker_loop(app))
        app.state.dlq_alert_task = asyncio.create_task(dlq_alert_loop())

    try:
        yield
    finally:
        for task_name in ("worker_task", "dlq_alert_task"):
            task = getattr(app.state, task_name, None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await broker.close()
        log.info("app.shutdown", backend=broker.name)


app = FastAPI(title="EY Payment Queue API", version="3.0.0", lifespan=lifespan)


@app.post("/payments", status_code=202)
async def enqueue_payment(payment: PaymentRequest):
    msg_id = await broker.publish(payment.model_dump(), priority=payment.priority)
    return {
        "message_id": msg_id,
        "status": "queued",
        "backend": broker.name,
        "priority": payment.priority,
        "queue_depth": await broker.depth(MAIN_QUEUE),
    }


@app.get("/health/live")
async def liveness():
    return {"status": "alive", "backend": broker.name}


@app.get("/health/ready")
async def readiness():
    checks = {"db": "ok"}
    mq = await broker.health_check()
    checks["mq"] = mq["status"]
    all_ok = all(value == "ok" for value in checks.values())
    return JSONResponse(
        content={
            "status": "ready" if all_ok else "not_ready",
            "backend": broker.name,
            **checks,
            "queue_depth": await broker.depth(MAIN_QUEUE),
            "dlq_depth": await broker.depth(DLQ_QUEUE),
        },
        status_code=200 if all_ok else 503,
    )


@app.get("/admin/dlq")
async def inspect_dlq(limit: int = Query(default=10, le=50)):
    items = await broker.inspect_dlq(limit=limit)
    return {"backend": broker.name, "dlq_depth": await broker.depth(DLQ_QUEUE), "messages": items}


@app.post("/admin/dlq/retry")
async def replay_dlq(limit: int = Query(default=5, le=20)):
    replayed = await broker.replay_dlq(limit=limit)
    return {
        "backend": broker.name,
        "replayed": len(replayed),
        "message_ids": replayed,
        "payments_depth": await broker.depth(MAIN_QUEUE),
        "dlq_depth": await broker.depth(DLQ_QUEUE),
    }


@app.get("/admin/stats")
async def queue_stats():
    stats = getattr(broker, "stats", {})
    return {
        "backend": broker.name,
        **stats,
        "payments_depth": await broker.depth(MAIN_QUEUE),
        "dlq_depth": await broker.depth(DLQ_QUEUE),
        "background_worker": BACKGROUND_WORKER_ENABLED,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

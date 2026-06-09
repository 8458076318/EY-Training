from __future__ import annotations

import asyncio
import random
import uuid
from typing import Any, Dict, List, Optional

from .common import log
from .brokers import InMemoryBroker
from .container import broker
from .metrics import DLQ_DEPTH_GAUGE
from .models import QueueMessage
from .settings import (
    BACKGROUND_WORKER_ENABLED,
    DLQ_ALERT_THRESHOLD,
    DLQ_ALERT_WINDOW_SECONDS,
    DLQ_QUEUE,
    MAX_DELIVERIES,
    PROCESSOR_FAIL_MODE,
    WORKER_POLL_INTERVAL,
)


async def process_payment(message: QueueMessage) -> Dict[str, Any]:
    await asyncio.sleep(0.05)

    if message.body.get("force_fail") or PROCESSOR_FAIL_MODE == "always":
        raise ValueError(f"processor forced to fail for {message.body.get('account_id')}")

    if PROCESSOR_FAIL_MODE == "random" and random.random() < 0.4:
        raise ValueError(f"fraud check timeout for {message.body.get('account_id')}")

    return {
        "processed_id": str(uuid.uuid4()),
        "status": "settled",
        "account_id": message.body.get("account_id"),
    }


async def worker_tick() -> Optional[Dict[str, Any]]:
    message = await broker.get_next()
    if message is None:
        return None

    message.delivery_count += 1
    log.info(
        "worker.processing",
        backend=broker.name,
        message_id=message.id,
        priority=message.priority,
        delivery_count=message.delivery_count,
        account=message.body.get("account_id"),
    )

    try:
        result = await process_payment(message)
        await broker.ack(message)
        return {"acked": message.id, "priority": message.priority, "result": result}
    except Exception as exc:
        if message.delivery_count >= MAX_DELIVERIES:
            await broker.dead_letter(message, reason=str(exc))
            return {"dead_lettered": message.id, "priority": message.priority, "reason": str(exc)}

        await asyncio.sleep(0.5 * message.delivery_count)
        await broker.retry(message, reason=str(exc))
        return {"nacked": message.id, "priority": message.priority, "attempt": message.delivery_count}


async def worker_loop(app: Any) -> None:
    while True:
        if getattr(app.state, "worker_pause", False):
            await asyncio.sleep(0.5)
            continue

        try:
            result = await worker_tick()
            if result is None:
                await asyncio.sleep(WORKER_POLL_INTERVAL)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            log.exception("worker.loop_error", backend=broker.name, error=str(exc))
            await asyncio.sleep(1.0)


async def dlq_alert_loop() -> None:
    dlq_over_threshold_since: Optional[float] = None
    alert_fired = False

    while True:
        try:
            depth = await broker.depth(DLQ_QUEUE)
            DLQ_DEPTH_GAUGE.set(depth)

            now = time.monotonic()
            if depth > DLQ_ALERT_THRESHOLD:
                if dlq_over_threshold_since is None:
                    dlq_over_threshold_since = now
                elif not alert_fired and now - dlq_over_threshold_since >= DLQ_ALERT_WINDOW_SECONDS:
                    log.critical(
                        "payments.dlq_alert",
                        backend=broker.name,
                        depth=depth,
                        threshold=DLQ_ALERT_THRESHOLD,
                        window_seconds=DLQ_ALERT_WINDOW_SECONDS,
                    )
                    alert_fired = True
            else:
                dlq_over_threshold_since = None
                alert_fired = False

            await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            log.exception("dlq.alert_loop_error", error=str(exc))
            await asyncio.sleep(5)


import time


async def priority_ordering_snapshot() -> List[str]:
    test_broker = InMemoryBroker()
    await test_broker.publish({"account_id": "LOW", "amount": 10}, priority=2)
    await test_broker.publish({"account_id": "HIGH", "amount": 20000}, priority=8)
    await test_broker.publish({"account_id": "MID", "amount": 100}, priority=5)

    order: List[str] = []
    while True:
        message = await test_broker.get_next(timeout=0.01)
        if message is None:
            break
        order.append(message.body["account_id"])
    return order

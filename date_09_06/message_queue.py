"""Message queue demo with pluggable broker backends.

This module keeps the original FastAPI payment demo but makes the broker
switchable:

* ``memory`` - in-process queue with priority ordering
* ``rabbitmq`` - CloudAMQP / RabbitMQ via ``aio_pika``
* ``servicebus`` - Azure Service Bus via the Azure SDK

It also adds:

* priority-based dispatch for high-value payments
* a background worker that continuously drains the queue
* a DLQ depth Prometheus gauge plus a critical alert when the DLQ stays high
  for too long
* admin endpoints to inspect, replay, and summarize the DLQ

The module is intentionally import-safe. Nothing starts until FastAPI's lifespan
hooks run.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional

import nest_asyncio
import structlog
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator
from contextlib import asynccontextmanager

nest_asyncio.apply()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


try:
    from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
except Exception:  # pragma: no cover - fallback for minimal environments
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoOpGauge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._value = 0.0

        def set(self, value: float) -> None:
            self._value = float(value)

        def inc(self, amount: float = 1.0) -> None:
            self._value += amount

        def dec(self, amount: float = 1.0) -> None:
            self._value -= amount

    def generate_latest() -> bytes:  # type: ignore[override]
        return b"# prometheus_client not installed\n"

    def Gauge(*args: Any, **kwargs: Any) -> _NoOpGauge:  # type: ignore[misc]
        return _NoOpGauge(*args, **kwargs)


structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


MAIN_QUEUE = "payments"
DLQ_QUEUE = "payments.dlq"
QUEUE_TTL_MS = int(os.getenv("PAYMENTS_QUEUE_TTL_MS", "86400000"))
MAX_DELIVERIES = int(os.getenv("PAYMENTS_MAX_DELIVERIES", "3"))
WORKER_POLL_INTERVAL = float(os.getenv("PAYMENTS_WORKER_POLL_INTERVAL", "0.25"))
BACKGROUND_WORKER_ENABLED = os.getenv("PAYMENTS_BACKGROUND_WORKER", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BROKER_BACKEND = os.getenv("PAYMENTS_BROKER", "memory").strip().lower()
DLQ_ALERT_THRESHOLD = int(os.getenv("PAYMENTS_DLQ_ALERT_THRESHOLD", "10"))
DLQ_ALERT_WINDOW_SECONDS = int(os.getenv("PAYMENTS_DLQ_ALERT_WINDOW_SECONDS", "30"))
PROCESSOR_FAIL_MODE = os.getenv("PAYMENTS_PROCESSOR_FAIL_MODE", "random").strip().lower()

DLQ_DEPTH_GAUGE = Gauge(
    "payments_dlq_depth",
    "Current depth of the payments DLQ",
)


@dataclass
class QueueMessage:
    id: str
    body: Dict[str, Any]
    priority: int = 5
    attempts: int = 0
    delivery_count: int = 0
    created_at: str = field(default_factory=_utc_now)
    dead_lettered_at: Optional[str] = None
    dead_letter_reason: Optional[str] = None
    transport: Any = field(default=None, repr=False, compare=False)


class PaymentRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Amount in GBP")
    currency: str = Field(default="GBP")
    account_id: str
    reference: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def _assign_priority(self) -> "PaymentRequest":
        if self.amount > 10000:
            self.priority = max(self.priority, 8)
        return self


class BaseBroker(ABC):
    name = "base"

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    async def publish(self, payload: Dict[str, Any], priority: int, attempts: int = 0) -> str:
        raise NotImplementedError

    @abstractmethod
    async def get_next(self, timeout: float = WORKER_POLL_INTERVAL) -> Optional[QueueMessage]:
        raise NotImplementedError

    @abstractmethod
    async def ack(self, message: QueueMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    async def retry(self, message: QueueMessage, reason: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def dead_letter(self, message: QueueMessage, reason: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def inspect_dlq(self, limit: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def replay_dlq(self, limit: int = 5) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    async def depth(self, queue_name: str) -> int:
        raise NotImplementedError

    async def health_check(self) -> Dict[str, str]:
        return {"status": "ok", "backend": self.name}


class InMemoryBroker(BaseBroker):
    name = "memory"

    def __init__(self) -> None:
        self._main: asyncio.PriorityQueue[tuple[int, int, QueueMessage]] = asyncio.PriorityQueue()
        self._dlq: asyncio.Queue[QueueMessage] = asyncio.Queue()
        self._sequence = count()
        self.stats = {"published": 0, "consumed": 0, "dead_lettered": 0, "retried": 0}

    async def publish(self, payload: Dict[str, Any], priority: int, attempts: int = 0) -> str:
        message = QueueMessage(
            id=str(uuid.uuid4()),
            body=payload,
            priority=priority,
            attempts=attempts,
            delivery_count=attempts,
        )
        await self._main.put((-priority, next(self._sequence), message))
        self.stats["published"] += 1
        log.info("mq.published", backend=self.name, queue=MAIN_QUEUE, message_id=message.id, priority=priority)
        return message.id

    async def get_next(self, timeout: float = WORKER_POLL_INTERVAL) -> Optional[QueueMessage]:
        try:
            _, _, message = await asyncio.wait_for(self._main.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return message

    async def ack(self, message: QueueMessage) -> None:
        self.stats["consumed"] += 1
        log.info("worker.acked", backend=self.name, message_id=message.id, priority=message.priority)

    async def retry(self, message: QueueMessage, reason: str) -> None:
        message.attempts += 1
        message.delivery_count = message.attempts
        self.stats["retried"] += 1
        await self._main.put((-message.priority, next(self._sequence), message))
        log.warning(
            "worker.nacked",
            backend=self.name,
            message_id=message.id,
            priority=message.priority,
            attempt=message.delivery_count,
            error=reason,
        )

    async def dead_letter(self, message: QueueMessage, reason: str) -> None:
        message.dead_lettered_at = _utc_now()
        message.dead_letter_reason = reason
        await self._dlq.put(message)
        self.stats["dead_lettered"] += 1
        log.warning("mq.dead_lettered", backend=self.name, message_id=message.id, reason=reason)

    async def inspect_dlq(self, limit: int = 10) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        temp: List[QueueMessage] = []
        while len(items) < limit:
            try:
                message = self._dlq.get_nowait()
            except asyncio.QueueEmpty:
                break
            items.append(
                {
                    "id": message.id,
                    "body": message.body,
                    "priority": message.priority,
                    "attempts": message.attempts,
                    "delivery_count": message.delivery_count,
                    "dead_lettered_at": message.dead_lettered_at,
                    "reason": message.dead_letter_reason,
                }
            )
            temp.append(message)

        for message in temp:
            await self._dlq.put(message)
        return items

    async def replay_dlq(self, limit: int = 5) -> List[str]:
        replayed: List[str] = []
        for _ in range(limit):
            try:
                message = self._dlq.get_nowait()
            except asyncio.QueueEmpty:
                break
            message.dead_lettered_at = None
            message.dead_letter_reason = None
            message.delivery_count = 0
            await self._main.put((-message.priority, next(self._sequence), message))
            replayed.append(message.id)
            log.info("dlq.replayed", backend=self.name, message_id=message.id)
        return replayed

    async def depth(self, queue_name: str) -> int:
        if queue_name == DLQ_QUEUE:
            return self._dlq.qsize()
        return self._main.qsize()


class RabbitMQBroker(BaseBroker):
    name = "rabbitmq"

    def __init__(self, url: str) -> None:
        self.url = url
        self._connection: Any = None
        self._channel: Any = None
        self._exchange: Any = None
        self._main_queue: Any = None
        self._dlq_queue: Any = None
        self._initialized = False

    async def connect(self) -> None:
        if self._initialized:
            return

        try:
            import aio_pika
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("aio_pika is required for the rabbitmq backend") from exc

        self._connection = await aio_pika.connect_robust(self.url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=1)

        self._exchange = await self._channel.declare_exchange(
            "payments.dlx",
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        self._main_queue = await self._channel.declare_queue(
            MAIN_QUEUE,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "payments.dlx",
                "x-dead-letter-routing-key": DLQ_QUEUE,
                "x-message-ttl": QUEUE_TTL_MS,
                "x-max-priority": 10,
            },
        )
        self._dlq_queue = await self._channel.declare_queue(DLQ_QUEUE, durable=True)
        await self._dlq_queue.bind(self._exchange, routing_key=DLQ_QUEUE)
        self._initialized = True
        log.info("broker.connected", backend=self.name)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
        self._initialized = False

    async def publish(self, payload: Dict[str, Any], priority: int, attempts: int = 0) -> str:
        await self.connect()
        import aio_pika

        message_id = str(uuid.uuid4())
        message = aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=message_id,
            priority=priority,
            headers={"attempts": attempts, "priority": priority},
        )
        await self._channel.default_exchange.publish(message, routing_key=MAIN_QUEUE)
        log.info("mq.published", backend=self.name, queue=MAIN_QUEUE, message_id=message_id, priority=priority)
        return message_id

    async def get_next(self, timeout: float = WORKER_POLL_INTERVAL) -> Optional[QueueMessage]:
        await self.connect()
        try:
            raw = await asyncio.wait_for(self._main_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        body = json.loads(raw.body.decode("utf-8"))
        attempts = int(raw.headers.get("attempts", 0) if raw.headers else 0)
        priority = int(raw.priority or body.get("priority", 5))
        return QueueMessage(
            id=raw.message_id or str(uuid.uuid4()),
            body=body,
            priority=priority,
            attempts=attempts,
            delivery_count=attempts,
            transport=raw,
        )

    async def ack(self, message: QueueMessage) -> None:
        await message.transport.ack()
        log.info("worker.acked", backend=self.name, message_id=message.id, priority=message.priority)

    async def retry(self, message: QueueMessage, reason: str) -> None:
        await self.publish(message.body, priority=message.priority, attempts=message.attempts)
        await message.transport.ack()
        log.warning(
            "worker.nacked",
            backend=self.name,
            message_id=message.id,
            priority=message.priority,
            attempt=message.delivery_count,
            error=reason,
        )

    async def dead_letter(self, message: QueueMessage, reason: str) -> None:
        await message.transport.reject(requeue=False)
        message.dead_lettered_at = _utc_now()
        message.dead_letter_reason = reason
        log.warning("mq.dead_lettered", backend=self.name, message_id=message.id, reason=reason)

    async def inspect_dlq(self, limit: int = 10) -> List[Dict[str, Any]]:
        await self.connect()
        import aio_pika

        items: List[Dict[str, Any]] = []
        temp: List[Any] = []
        try:
            while len(items) < limit:
                raw = await asyncio.wait_for(self._dlq_queue.get(), timeout=0.2)
                body = json.loads(raw.body.decode("utf-8"))
                attempts = int(raw.headers.get("attempts", 0) if raw.headers else 0)
                priority = int(raw.priority or body.get("priority", 5))
                items.append(
                    {
                        "id": raw.message_id or str(uuid.uuid4()),
                        "body": body,
                        "priority": priority,
                        "attempts": attempts,
                        "delivery_count": attempts,
                        "dead_lettered_at": _utc_now(),
                        "reason": (raw.headers or {}).get("dead_letter_reason"),
                    }
                )
                temp.append(raw)
        except asyncio.TimeoutError:
            pass

        # Put the sample back so the DLQ remains intact.
        for raw in temp:
            clone = aio_pika.Message(
                body=raw.body,
                content_type=raw.content_type,
                delivery_mode=raw.delivery_mode,
                message_id=raw.message_id,
                priority=raw.priority,
                headers=dict(raw.headers or {}),
            )
            await self._channel.default_exchange.publish(clone, routing_key=DLQ_QUEUE)
        return items

    async def replay_dlq(self, limit: int = 5) -> List[str]:
        await self.connect()
        replayed: List[str] = []
        try:
            for _ in range(limit):
                raw = await asyncio.wait_for(self._dlq_queue.get(), timeout=0.2)
                body = json.loads(raw.body.decode("utf-8"))
                priority = int(raw.priority or body.get("priority", 5))
                attempts = int(raw.headers.get("attempts", 0) if raw.headers else 0)
                await self.publish(body, priority=priority, attempts=attempts)
                await raw.ack()
                replayed.append(raw.message_id or str(uuid.uuid4()))
                log.info("dlq.replayed", backend=self.name, message_id=replayed[-1])
        except asyncio.TimeoutError:
            pass
        return replayed

    async def depth(self, queue_name: str) -> int:
        await self.connect()
        queue = self._dlq_queue if queue_name == DLQ_QUEUE else self._main_queue
        result = await queue.declare(passive=True)
        return int(result.message_count)

    async def health_check(self) -> Dict[str, str]:
        try:
            await self.connect()
            await self.depth(MAIN_QUEUE)
            return {"status": "ok", "backend": self.name}
        except Exception as exc:
            return {"status": f"error: {exc}", "backend": self.name}


class ServiceBusBroker(BaseBroker):
    name = "servicebus"

    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self._client: Any = None
        self._admin: Any = None
        self._sender: Any = None
        self._main_receiver: Any = None
        self._dlq_receiver: Any = None
        self._initialized = False

    async def connect(self) -> None:
        if self._initialized:
            return

        try:
            from azure.servicebus.aio import ServiceBusClient
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("azure-servicebus is required for the servicebus backend") from exc

        from azure.servicebus import ServiceBusReceiveMode, ServiceBusSubQueue

        self._client = ServiceBusClient.from_connection_string(self.conn_str)
        try:
            from azure.servicebus.aio.management import ServiceBusAdministrationClient

            self._admin = ServiceBusAdministrationClient.from_connection_string(self.conn_str)
            await self._admin.__aenter__()
        except Exception:
            self._admin = None
        self._sender = self._client.get_queue_sender(queue_name=MAIN_QUEUE)
        self._main_receiver = self._client.get_queue_receiver(
            queue_name=MAIN_QUEUE,
            receive_mode=ServiceBusReceiveMode.PEEK_LOCK,
            max_wait_time=WORKER_POLL_INTERVAL,
        )
        self._dlq_receiver = self._client.get_queue_receiver(
            queue_name=MAIN_QUEUE,
            sub_queue=ServiceBusSubQueue.DEAD_LETTER,
            receive_mode=ServiceBusReceiveMode.PEEK_LOCK,
            max_wait_time=WORKER_POLL_INTERVAL,
        )
        await self._sender.__aenter__()
        await self._main_receiver.__aenter__()
        await self._dlq_receiver.__aenter__()
        self._initialized = True
        log.info("broker.connected", backend=self.name)

    async def close(self) -> None:
        if self._dlq_receiver is not None:
            await self._dlq_receiver.__aexit__(None, None, None)
        if self._main_receiver is not None:
            await self._main_receiver.__aexit__(None, None, None)
        if self._sender is not None:
            await self._sender.__aexit__(None, None, None)
        if self._admin is not None:
            await self._admin.__aexit__(None, None, None)
        if self._client is not None:
            await self._client.close()
        self._initialized = False

    async def publish(self, payload: Dict[str, Any], priority: int, attempts: int = 0) -> str:
        await self.connect()
        from azure.servicebus import ServiceBusMessage

        message_id = str(uuid.uuid4())
        message = ServiceBusMessage(
            json.dumps(payload),
            message_id=message_id,
            application_properties={"priority": priority, "attempts": attempts},
            content_type="application/json",
        )
        await self._sender.send_messages(message)
        log.info("mq.published", backend=self.name, queue=MAIN_QUEUE, message_id=message_id, priority=priority)
        return message_id

    async def get_next(self, timeout: float = WORKER_POLL_INTERVAL) -> Optional[QueueMessage]:
        await self.connect()
        from azure.servicebus import ServiceBusReceiveMode

        messages = await self._main_receiver.receive_messages(max_message_count=1, max_wait_time=timeout)
        if not messages:
            return None

        raw = messages[0]
        body = _servicebus_message_to_dict(raw)
        attempts = int((raw.application_properties or {}).get("attempts", 0))
        priority = int((raw.application_properties or {}).get("priority", body.get("priority", 5)))
        return QueueMessage(
            id=raw.message_id or str(uuid.uuid4()),
            body=body,
            priority=priority,
            attempts=attempts,
            delivery_count=attempts,
            transport=(self._main_receiver, raw),
        )

    async def ack(self, message: QueueMessage) -> None:
        receiver, raw = message.transport
        await receiver.complete_message(raw)
        log.info("worker.acked", backend=self.name, message_id=message.id, priority=message.priority)

    async def retry(self, message: QueueMessage, reason: str) -> None:
        receiver, raw = message.transport
        await self.publish(message.body, priority=message.priority, attempts=message.attempts)
        await receiver.complete_message(raw)
        log.warning(
            "worker.nacked",
            backend=self.name,
            message_id=message.id,
            priority=message.priority,
            attempt=message.delivery_count,
            error=reason,
        )

    async def dead_letter(self, message: QueueMessage, reason: str) -> None:
        receiver, raw = message.transport
        await receiver.dead_letter_message(
            raw,
            reason=type(reason).__name__ if isinstance(reason, Exception) else "processing_error",
            error_description=str(reason),
        )
        message.dead_lettered_at = _utc_now()
        message.dead_letter_reason = reason
        log.warning("mq.dead_lettered", backend=self.name, message_id=message.id, reason=reason)

    async def inspect_dlq(self, limit: int = 10) -> List[Dict[str, Any]]:
        await self.connect()
        messages = await self._dlq_receiver.peek_messages(max_message_count=limit)
        items: List[Dict[str, Any]] = []
        for raw in messages:
            body = _servicebus_message_to_dict(raw)
            attempts = int((raw.application_properties or {}).get("attempts", 0))
            priority = int((raw.application_properties or {}).get("priority", body.get("priority", 5)))
            items.append(
                {
                    "id": raw.message_id or str(uuid.uuid4()),
                    "body": body,
                    "priority": priority,
                    "attempts": attempts,
                    "delivery_count": attempts,
                    "dead_lettered_at": getattr(raw, "enqueued_time_utc", None),
                    "reason": getattr(raw, "dead_letter_reason", None),
                }
            )
        return items

    async def replay_dlq(self, limit: int = 5) -> List[str]:
        await self.connect()
        replayed: List[str] = []
        messages = await self._dlq_receiver.receive_messages(max_message_count=limit, max_wait_time=1)
        for raw in messages:
            body = _servicebus_message_to_dict(raw)
            priority = int((raw.application_properties or {}).get("priority", body.get("priority", 5)))
            attempts = int((raw.application_properties or {}).get("attempts", 0))
            await self.publish(body, priority=priority, attempts=attempts)
            await self._dlq_receiver.complete_message(raw)
            replayed.append(raw.message_id or str(uuid.uuid4()))
            log.info("dlq.replayed", backend=self.name, message_id=replayed[-1])
        return replayed

    async def depth(self, queue_name: str) -> int:
        await self.connect()
        if self._admin is not None:
            props = await self._admin.get_queue_runtime_properties(MAIN_QUEUE)
            return int(props.dead_letter_message_count if queue_name == DLQ_QUEUE else props.active_message_count)

        receiver = self._dlq_receiver if queue_name == DLQ_QUEUE else self._main_receiver
        messages = await receiver.peek_messages(max_message_count=1)
        # Azure doesn't expose queue depth cheaply via every SDK surface, so we
        # fall back to a sample when runtime properties are unavailable.
        return len(messages)

    async def health_check(self) -> Dict[str, str]:
        try:
            await self.connect()
            await self.depth(MAIN_QUEUE)
            return {"status": "ok", "backend": self.name}
        except Exception as exc:
            return {"status": f"error: {exc}", "backend": self.name}


def build_broker() -> BaseBroker:
    if BROKER_BACKEND == "rabbitmq":
        rabbit_url = (
            os.getenv("CLOUDAMQP_URL")
            or os.getenv("RABBITMQ_URL")
            or os.getenv("AMQP_URL")
            or "amqp://guest:guest@localhost/"
        )
        return RabbitMQBroker(rabbit_url)
    if BROKER_BACKEND == "servicebus":
        sb_conn = os.getenv("SB_CONN_STR") or os.getenv("SERVICEBUS_CONNECTION_STRING", "")
        if not sb_conn:
            raise RuntimeError("SB_CONN_STR or SERVICEBUS_CONNECTION_STRING is required for servicebus backend")
        return ServiceBusBroker(sb_conn)
    return InMemoryBroker()


broker = build_broker()


def _servicebus_message_to_dict(message: Any) -> Dict[str, Any]:
    body = getattr(message, "body", None)
    if isinstance(body, dict):
        return body
    if isinstance(body, (bytes, bytearray)):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    if hasattr(message, "body_as_str"):
        return json.loads(message.body_as_str())
    if hasattr(message, "body_as_bytes"):
        raw = message.body_as_bytes()
        if isinstance(raw, (bytes, bytearray)):
            return json.loads(raw.decode("utf-8"))
    return json.loads(str(message))


async def process_payment(message: QueueMessage) -> Dict[str, Any]:
    """Simulate payment processing, with a configurable failure mode."""
    await asyncio.sleep(0.05)

    if message.body.get("force_fail") or PROCESSOR_FAIL_MODE == "always":
        raise ValueError(f"processor forced to fail for {message.body.get('account_id')}")

    if PROCESSOR_FAIL_MODE == "random":
        if random.random() < 0.4:
            raise ValueError(f"fraud check timeout for {message.body.get('account_id')}")

    return {
        "processed_id": str(uuid.uuid4()),
        "status": "settled",
        "account_id": message.body.get("account_id"),
    }


async def worker_tick() -> Optional[Dict[str, Any]]:
    """Process one message from the queue."""
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


async def _worker_loop(app: FastAPI) -> None:
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


async def _dlq_alert_loop() -> None:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app.startup", backend=broker.name)
    await broker.connect()

    app.state.worker_task = None
    app.state.dlq_alert_task = None
    app.state.worker_pause = False

    if BACKGROUND_WORKER_ENABLED:
        app.state.worker_task = asyncio.create_task(_worker_loop(app))
        app.state.dlq_alert_task = asyncio.create_task(_dlq_alert_loop())

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


import contextlib


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
    status_code = 200 if all_ok else 503
    return JSONResponse(
        content={
            "status": "ready" if all_ok else "not_ready",
            "backend": broker.name,
            **checks,
            "queue_depth": await broker.depth(MAIN_QUEUE),
            "dlq_depth": await broker.depth(DLQ_QUEUE),
        },
        status_code=status_code,
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
        "max_deliveries": MAX_DELIVERIES,
        "background_worker": BACKGROUND_WORKER_ENABLED,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def priority_ordering_snapshot() -> List[str]:
    """Small helper for tests: highest priority messages are returned first."""
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


if __name__ == "__main__":  # pragma: no cover - convenience runner
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("message_queue:app", host="0.0.0.0", port=port, reload=False)

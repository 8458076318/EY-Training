from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import count
from typing import Any, Dict, List, Optional

from .common import log, utc_now
from .models import QueueMessage
from .settings import DLQ_QUEUE, MAIN_QUEUE, QUEUE_TTL_MS


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
    async def get_next(self, timeout: float = 0.25) -> Optional[QueueMessage]:
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

    async def get_next(self, timeout: float = 0.25) -> Optional[QueueMessage]:
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
        message.dead_lettered_at = utc_now()
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
        except Exception as exc:  # pragma: no cover - optional dependency
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

    async def get_next(self, timeout: float = 0.25) -> Optional[QueueMessage]:
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
        message.dead_lettered_at = utc_now()
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
                        "dead_lettered_at": utc_now(),
                        "reason": (raw.headers or {}).get("dead_letter_reason"),
                    }
                )
                temp.append(raw)
        except asyncio.TimeoutError:
            pass

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
            from azure.servicebus.aio.management import ServiceBusAdministrationClient
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("azure-servicebus is required for the servicebus backend") from exc

        from azure.servicebus import ServiceBusReceiveMode, ServiceBusSubQueue

        self._client = ServiceBusClient.from_connection_string(self.conn_str)
        self._admin = ServiceBusAdministrationClient.from_connection_string(self.conn_str)
        await self._admin.__aenter__()
        self._sender = self._client.get_queue_sender(queue_name=MAIN_QUEUE)
        self._main_receiver = self._client.get_queue_receiver(
            queue_name=MAIN_QUEUE,
            receive_mode=ServiceBusReceiveMode.PEEK_LOCK,
            max_wait_time=0.25,
        )
        self._dlq_receiver = self._client.get_queue_receiver(
            queue_name=MAIN_QUEUE,
            sub_queue=ServiceBusSubQueue.DEAD_LETTER,
            receive_mode=ServiceBusReceiveMode.PEEK_LOCK,
            max_wait_time=0.25,
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

    async def get_next(self, timeout: float = 0.25) -> Optional[QueueMessage]:
        await self.connect()
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
        message.dead_lettered_at = utc_now()
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
        props = await self._admin.get_queue_runtime_properties(MAIN_QUEUE)
        return int(props.dead_letter_message_count if queue_name == DLQ_QUEUE else props.active_message_count)

    async def health_check(self) -> Dict[str, str]:
        try:
            await self.connect()
            await self.depth(MAIN_QUEUE)
            return {"status": "ok", "backend": self.name}
        except Exception as exc:
            return {"status": f"error: {exc}", "backend": self.name}


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


def build_broker() -> BaseBroker:
    from .settings import BROKER_BACKEND

    if BROKER_BACKEND == "rabbitmq":
        import os

        rabbit_url = (
            os.getenv("CLOUDAMQP_URL")
            or os.getenv("RABBITMQ_URL")
            or os.getenv("AMQP_URL")
            or "amqp://guest:guest@localhost/"
        )
        return RabbitMQBroker(rabbit_url)

    if BROKER_BACKEND == "servicebus":
        import os

        sb_conn = os.getenv("SB_CONN_STR") or os.getenv("SERVICEBUS_CONNECTION_STRING", "")
        if not sb_conn:
            raise RuntimeError("SB_CONN_STR or SERVICEBUS_CONNECTION_STRING is required for servicebus backend")
        return ServiceBusBroker(sb_conn)

    return InMemoryBroker()

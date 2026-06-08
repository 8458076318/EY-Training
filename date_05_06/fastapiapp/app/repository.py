from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from .models import Order, OrderStatus


class OrderRepository:
    """Simple in-memory store."""

    def __init__(self) -> None:
        self._store: Dict[UUID, Order] = {}

    def save(self, order: Order) -> Order:
        self._store[order.id] = order
        return order

    def get(self, order_id: UUID) -> Optional[Order]:
        return self._store.get(order_id)

    def list_all(self) -> List[Order]:
        return list(self._store.values())

    def update_status(self, order_id: UUID, new_status: OrderStatus) -> Optional[Order]:
        order = self._store.get(order_id)
        if order is None:
            return None

        updated = order.model_copy(update={"status": new_status, "updated_at": datetime.utcnow()})
        self._store[order_id] = updated
        return updated

    def delete(self, order_id: UUID) -> bool:
        if order_id not in self._store:
            return False

        del self._store[order_id]
        return True


_repo = OrderRepository()


def get_repository() -> OrderRepository:
    return _repo


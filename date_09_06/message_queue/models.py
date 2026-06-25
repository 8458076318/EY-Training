from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from .common import utc_now


@dataclass
class QueueMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    body: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    attempts: int = 0
    delivery_count: int = 0
    created_at: str = field(default_factory=utc_now)
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

from pydantic import (BaseModel, Field,
    field_validator, model_validator)
from typing import Optional
from datetime import datetime
from enum import Enum

class Status(str, Enum):
    active  = "active"
    churned = "churned"

class CustomerRecord(BaseModel):
    customer_id : str
    revenue     : float = Field(gt=0,
        description="Annual revenue £")
    region      : str   = Field(
        pattern=r"^[A-Z]{2}$")
    status      : Status = Status.active
    joined_at   : datetime
    tags        : list[str] = []
    metadata    : Optional[dict] = None

    @field_validator("customer_id")
    @classmethod
    def id_must_be_uppercase(cls, v):
        if not v.isupper():
            raise ValueError("Must be UPPER")
        return v

    @model_validator(mode="after")
    def churned_needs_reason(self):
        if (self.status == Status.churned
                and not self.metadata):
            raise ValueError(
                "Churned record needs metadata")
        return self

# Usage
rec = CustomerRecord(
    customer_id="CUST001",
    revenue=125000,
    region="GB",
    joined_at="2023-01-15T09:00:00")
print(rec.model_dump_json(indent=2))
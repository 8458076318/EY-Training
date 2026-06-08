from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentMethod(str, Enum):
    CARD = "card"
    UPI = "upi"
    CASH = "cash"
    WALLET = "wallet"


class Address(BaseModel):
    """Physical delivery or billing address."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "street": "42 MG Road",
                "city": "Dehradun",
                "state": "Uttarakhand",
                "pincode": "248001",
                "country": "India",
            }
        }
    )

    street: str = Field(..., min_length=3)
    city: str
    state: str
    pincode: str = Field(..., pattern=r"^\d{6}$")
    country: str = Field(default="India")


class Product(BaseModel):
    """Catalogue product."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    price: float = Field(..., gt=0)
    sku: str
    in_stock: bool = True

    @field_validator("price")
    @classmethod
    def price_precision(cls, value: float) -> float:
        return round(value, 2)


class OrderItem(BaseModel):
    """A product line inside an order."""

    product: Product
    quantity: int = Field(..., ge=1, le=100)
    discount: float = Field(default=0.0, ge=0.0, le=100.0)

    @property
    def line_total(self) -> float:
        discounted = self.product.price * (1 - self.discount / 100)
        return round(discounted * self.quantity, 2)


class Customer(BaseModel):
    """Buyer details."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=2)
    email: EmailStr
    phone: str = Field(..., pattern=r"^\+?[1-9]\d{9,14}$")
    shipping_address: Address
    billing_address: Optional[Address] = None

    @model_validator(mode="after")
    def set_billing_address(self) -> "Customer":
        if self.billing_address is None:
            self.billing_address = self.shipping_address
        return self


class OrderCreate(BaseModel):
    """Request body to create a new order."""

    customer: Customer
    items: List[OrderItem] = Field(..., min_length=1)
    payment_method: PaymentMethod
    notes: Optional[str] = None

    @field_validator("items")
    @classmethod
    def items_not_empty(cls, value: List[OrderItem]) -> List[OrderItem]:
        if not value:
            raise ValueError("Order must contain at least one item.")
        return value


class Order(OrderCreate):
    """Full order as stored."""

    id: UUID = Field(default_factory=uuid4)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class OrderSummary(BaseModel):
    """Computed summary returned with each order."""

    subtotal: float
    total_discount: float
    grand_total: float
    item_count: int


class OrderResponse(Order):
    """Order response extended with computed summary values."""

    summary: OrderSummary

    @classmethod
    def from_order(cls, order: Order) -> "OrderResponse":
        subtotal = sum(item.product.price * item.quantity for item in order.items)
        total_discount = sum(
            (item.product.price * item.quantity) - item.line_total for item in order.items
        )

        return cls(
            **order.model_dump(),
            summary=OrderSummary(
                subtotal=round(subtotal, 2),
                total_discount=round(total_discount, 2),
                grand_total=round(subtotal - total_discount, 2),
                item_count=sum(item.quantity for item in order.items),
            ),
        )


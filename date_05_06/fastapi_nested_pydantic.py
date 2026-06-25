# -*- coding: utf-8 -*-
"""FastAPI demo with nested Pydantic models.

Run:
    uvicorn fastapi_nested_pydantic:app --reload

Docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
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

        updated = order.model_copy(
            update={"status": new_status, "updated_at": datetime.utcnow()}
        )
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


router = APIRouter(prefix="/orders", tags=["Orders"])


@router.post(
    "/",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Place a new order",
)
def create_order(
    payload: OrderCreate,
    repo: OrderRepository = Depends(get_repository),
) -> OrderResponse:
    order = Order(**payload.model_dump())
    return OrderResponse.from_order(repo.save(order))


@router.get(
    "/",
    response_model=List[OrderResponse],
    summary="List all orders",
)
def list_orders(
    repo: OrderRepository = Depends(get_repository),
) -> List[OrderResponse]:
    return [OrderResponse.from_order(order) for order in repo.list_all()]


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Get a single order by ID",
)
def get_order(
    order_id: UUID,
    repo: OrderRepository = Depends(get_repository),
) -> OrderResponse:
    order = repo.get(order_id)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found.",
        )
    return OrderResponse.from_order(order)


@router.patch(
    "/{order_id}/status",
    response_model=OrderResponse,
    summary="Update order status",
)
def update_order_status(
    order_id: UUID,
    new_status: OrderStatus,
    repo: OrderRepository = Depends(get_repository),
) -> OrderResponse:
    order = repo.update_status(order_id, new_status)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found.",
        )
    return OrderResponse.from_order(order)


@router.delete(
    "/{order_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel or delete an order",
)
def delete_order(
    order_id: UUID,
    repo: OrderRepository = Depends(get_repository),
) -> None:
    if not repo.delete(order_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found.",
        )


app = FastAPI(
    title="Order Management API",
    description=(
        "End-to-end FastAPI demo with nested Pydantic models and the "
        "extension response pattern."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Health"])
def health() -> dict:
    return {"status": "ok", "message": "Order Management API is running"}


ORDER_PAYLOAD = {
    "customer": {
        "name": "Aarav Sharma",
        "email": "aarav@example.com",
        "phone": "+919876543210",
        "shipping_address": {
            "street": "42 MG Road",
            "city": "Dehradun",
            "state": "Uttarakhand",
            "pincode": "248001",
            "country": "India",
        },
    },
    "items": [
        {
            "product": {
                "name": "Wireless Headphones",
                "price": 2999.00,
                "sku": "WH-001",
            },
            "quantity": 2,
            "discount": 10.0,
        },
        {
            "product": {
                "name": "USB-C Cable",
                "price": 499.00,
                "sku": "UC-002",
            },
            "quantity": 3,
            "discount": 0.0,
        },
    ],
    "payment_method": "upi",
    "notes": "Leave at gate",
}


def test_health() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_order_returns_summary() -> None:
    import pytest
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/orders/", json=ORDER_PAYLOAD)
    assert response.status_code == 201

    summary = response.json()["summary"]
    assert summary["item_count"] == 5
    assert summary["subtotal"] == 7495.0
    assert summary["total_discount"] == pytest.approx(599.8, rel=1e-3)
    assert summary["grand_total"] == pytest.approx(6895.2, rel=1e-3)


def test_create_order_nested_fields() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/orders/", json=ORDER_PAYLOAD)
    assert response.status_code == 201

    data = response.json()
    assert data["customer"]["shipping_address"]["pincode"] == "248001"
    assert data["customer"]["billing_address"]["city"] == "Dehradun"
    assert data["items"][0]["product"]["name"] == "Wireless Headphones"


def test_list_orders() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.post("/orders/", json=ORDER_PAYLOAD)
    response = client.get("/orders/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) >= 1


def test_get_order_by_id() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    create_response = client.post("/orders/", json=ORDER_PAYLOAD)
    order_id = create_response.json()["id"]

    response = client.get(f"/orders/{order_id}")
    assert response.status_code == 200
    assert response.json()["id"] == order_id


def test_get_order_not_found() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get(f"/orders/{uuid4()}")
    assert response.status_code == 404


def test_update_order_status() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    create_response = client.post("/orders/", json=ORDER_PAYLOAD)
    order_id = create_response.json()["id"]

    response = client.patch(f"/orders/{order_id}/status?new_status=confirmed")
    assert response.status_code == 200
    assert response.json()["status"] == "confirmed"


def test_delete_order() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    create_response = client.post("/orders/", json=ORDER_PAYLOAD)
    order_id = create_response.json()["id"]

    delete_response = client.delete(f"/orders/{order_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/orders/{order_id}")
    assert get_response.status_code == 404


def test_invalid_pincode() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    bad_payload = {**ORDER_PAYLOAD}
    bad_payload["customer"] = {
        **ORDER_PAYLOAD["customer"],
        "shipping_address": {
            **ORDER_PAYLOAD["customer"]["shipping_address"],
            "pincode": "12AB",
        },
    }

    response = client.post("/orders/", json=bad_payload)
    assert response.status_code == 422


def test_empty_items_rejected() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    bad_payload = {**ORDER_PAYLOAD, "items": []}
    response = client.post("/orders/", json=bad_payload)
    assert response.status_code == 422

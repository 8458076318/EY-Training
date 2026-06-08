from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from date_05_06.fastapiapp.app.main import app


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
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_order_returns_summary() -> None:
    client = TestClient(app)
    response = client.post("/orders/", json=ORDER_PAYLOAD)
    assert response.status_code == 201

    summary = response.json()["summary"]
    assert summary["item_count"] == 5
    assert summary["subtotal"] == 7495.0
    assert summary["total_discount"] == pytest.approx(599.8, rel=1e-3)
    assert summary["grand_total"] == pytest.approx(6895.2, rel=1e-3)


def test_create_order_nested_fields() -> None:
    client = TestClient(app)
    response = client.post("/orders/", json=ORDER_PAYLOAD)
    assert response.status_code == 201

    data = response.json()
    assert data["customer"]["shipping_address"]["pincode"] == "248001"
    assert data["customer"]["billing_address"]["city"] == "Dehradun"
    assert data["items"][0]["product"]["name"] == "Wireless Headphones"


def test_list_orders() -> None:
    client = TestClient(app)
    client.post("/orders/", json=ORDER_PAYLOAD)
    response = client.get("/orders/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) >= 1


def test_get_order_by_id() -> None:
    client = TestClient(app)
    create_response = client.post("/orders/", json=ORDER_PAYLOAD)
    order_id = create_response.json()["id"]

    response = client.get(f"/orders/{order_id}")
    assert response.status_code == 200
    assert response.json()["id"] == order_id


def test_get_order_not_found() -> None:
    client = TestClient(app)
    response = client.get(f"/orders/{uuid4()}")
    assert response.status_code == 404


def test_update_order_status() -> None:
    client = TestClient(app)
    create_response = client.post("/orders/", json=ORDER_PAYLOAD)
    order_id = create_response.json()["id"]

    response = client.patch(f"/orders/{order_id}/status?new_status=confirmed")
    assert response.status_code == 200
    assert response.json()["status"] == "confirmed"


def test_delete_order() -> None:
    client = TestClient(app)
    create_response = client.post("/orders/", json=ORDER_PAYLOAD)
    order_id = create_response.json()["id"]

    delete_response = client.delete(f"/orders/{order_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/orders/{order_id}")
    assert get_response.status_code == 404


def test_invalid_pincode() -> None:
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
    client = TestClient(app)
    bad_payload = {**ORDER_PAYLOAD, "items": []}
    response = client.post("/orders/", json=bad_payload)
    assert response.status_code == 422

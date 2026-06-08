from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from .models import Order, OrderCreate, OrderResponse, OrderStatus
from .repository import OrderRepository, get_repository


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


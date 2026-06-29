"""Notification routes — fetch unread alerts, mark as read.

The Streamlit frontend polls GET /notifications/unread every 30 s
to drive the notification bell UI.
"""
from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from src.db.session import get_db
from src.db.models import NotificationLog, User
from src.api.v1.deps import get_current_user

router = APIRouter()


@router.get("/unread")
async def get_unread_notifications(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(NotificationLog)
        .where(
            NotificationLog.user_id == current_user.id,
            NotificationLog.is_read == False,  # noqa: E712
        )
        .order_by(NotificationLog.created_at.desc())
        .limit(20)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": str(n.id),
            "title": n.title,
            "message": n.message,
            "sent_at": n.sent_at,
            "extra_data": n.extra_data,
        }
        for n in rows
    ]


@router.patch("/{notification_id}/read")
async def mark_as_read(
    notification_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await db.execute(
        update(NotificationLog)
        .where(
            NotificationLog.id == notification_id,
            NotificationLog.user_id == current_user.id,
        )
        .values(is_read=True, status="read")
    )
    from src.monitoring.metrics import NOTIFICATIONS_READ
    NOTIFICATIONS_READ.inc()
    return {"status": "ok"}


@router.patch("/read-all")
async def mark_all_read(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await db.execute(
        update(NotificationLog)
        .where(
            NotificationLog.user_id == current_user.id,
            NotificationLog.is_read == False,  # noqa: E712
        )
        .values(is_read=True, status="read")
    )
    return {"status": "ok"}


@router.put("/fcm-token")
async def register_fcm_token(
    token: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Called by the frontend after obtaining an FCM registration token."""
    current_user.fcm_token = token
    await db.flush()
    return {"status": "registered"}

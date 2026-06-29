"""Celery tasks + APScheduler cron for in-app push notifications (10 min before each plan event).

Delivery stack:
  - Firebase Cloud Messaging (FCM) for mobile / web push
  - Stored in notification_logs table so the Streamlit UI can poll and display them
"""
from celery import Celery
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.core.config import settings
import structlog

logger = structlog.get_logger(__name__)

celery_app = Celery(
    "day_planner",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "src.workers.notification_worker.send_push_notification": {"queue": "notifications"},
        "src.workers.notification_worker.store_in_app_notification": {"queue": "notifications"},
    },
)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_push_notification(self, user_id: str, title: str, body: str, data: dict | None = None):
    """Send FCM push notification and persist it to the DB for in-app display."""
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging

        if not firebase_admin._apps:
            cred = credentials.Certificate("config/firebase-service-account.json")
            firebase_admin.initialize_app(cred)

        # Fetch the user's FCM token from DB
        import asyncio
        from src.db.session import AsyncSessionLocal
        from src.db.models import User
        from sqlalchemy import select

        async def _get_fcm_token() -> str | None:
            async with AsyncSessionLocal() as db:
                user = await db.get(User, user_id)
                return getattr(user, "fcm_token", None) if user else None

        fcm_token = asyncio.get_event_loop().run_until_complete(_get_fcm_token())

        if fcm_token:
            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
                token=fcm_token,
                android=messaging.AndroidConfig(priority="high"),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(sound="default", badge=1)
                    )
                ),
            )
            response = messaging.send(message)
            logger.info("fcm_push_sent", response=response, user_id=user_id)

        # Always persist to DB so the UI can show it even without FCM token
        store_in_app_notification.delay(user_id=user_id, title=title, body=body, data=data)

        from src.monitoring.metrics import PUSH_SENT
        PUSH_SENT.labels(status="success").inc()

    except Exception as exc:
        from src.monitoring.metrics import PUSH_SENT
        PUSH_SENT.labels(status="failed").inc()
        logger.error("push_notification_failed", error=str(exc), user_id=user_id)
        raise self.retry(exc=exc)


@celery_app.task
def store_in_app_notification(user_id: str, title: str, body: str, data: dict | None = None):
    """Persist notification to DB — Streamlit polls this for the notification bell."""
    import asyncio
    from src.db.session import AsyncSessionLocal
    from src.db.models import NotificationLog
    from datetime import datetime, timezone

    async def _store():
        async with AsyncSessionLocal() as db:
            log = NotificationLog(
                user_id=user_id,
                channel="push",
                message=f"{title}: {body}",
                status="delivered",
                sent_at=datetime.now(timezone.utc).isoformat(),
                extra_data=data or {},
            )
            db.add(log)
            await db.commit()
            logger.info("in_app_notification_stored", user_id=user_id, title=title)

    asyncio.get_event_loop().run_until_complete(_store())


def schedule_plan_reminders():
    """
    Register APScheduler cron job at startup.
    Runs every 5 minutes, checks for plan events starting in ~10 minutes,
    fires push notifications for each.
    """
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _check_and_dispatch_reminders,
        "cron",
        minute="*/5",
        id="plan_reminder_dispatch",
        replace_existing=True,
        misfire_grace_time=60,
    )
    scheduler.start()
    logger.info("reminder_scheduler_started")
    return scheduler


async def _check_and_dispatch_reminders():
    """
    Query DB for today's plan events starting within the next 10 minutes.
    Dispatches a push notification for each unnotified event.
    """
    from datetime import datetime, timezone, timedelta, time as dt_time
    from src.db.session import AsyncSessionLocal
    from src.db.models import DayPlan, WeeklyPlan, User
    from sqlalchemy import select, and_

    now = datetime.now(timezone.utc)
    window_start = now + timedelta(minutes=8)
    window_end = now + timedelta(minutes=12)

    async with AsyncSessionLocal() as db:
        stmt = (
            select(DayPlan, User)
            .join(WeeklyPlan, WeeklyPlan.id == DayPlan.weekly_plan_id)
            .join(User, User.id == WeeklyPlan.user_id)
            .where(
                and_(
                    DayPlan.plan_date == now.date(),
                    DayPlan.reminders_sent == False,  # noqa: E712
                    User.is_active == True,           # noqa: E712
                )
            )
        )
        rows = (await db.execute(stmt)).all()

        for day_plan, user in rows:
            events = _get_upcoming_events(day_plan, window_start.time(), window_end.time())
            for event_name, event_time in events:
                send_push_notification.delay(
                    user_id=str(user.id),
                    title=f"⏰ {event_name} in 10 minutes",
                    body=f"Your {event_name.lower()} is scheduled at {event_time}. Stay on track!",
                    data={"day_plan_id": str(day_plan.id), "event": event_name},
                )
                logger.info("reminder_dispatched", user_id=str(user.id), event=event_name)

            day_plan.reminders_sent = True

        await db.commit()


def _get_upcoming_events(day_plan, window_start, window_end) -> list[tuple[str, str]]:
    """Return (event_name, time_str) pairs whose time falls inside the reminder window."""
    from datetime import time as dt_time

    candidates = []
    if day_plan.wake_time and _in_window(day_plan.wake_time, window_start, window_end):
        candidates.append(("Wake Up", str(day_plan.wake_time)[:5]))
    if day_plan.sleep_time and _in_window(day_plan.sleep_time, window_start, window_end):
        candidates.append(("Sleep Time", str(day_plan.sleep_time)[:5]))

    # Approximate meal times from plan text (stored in workout/meditation)
    meal_schedule = {
        "Breakfast": "08:00",
        "Lunch":     "13:00",
        "Dinner":    "20:00",
        "Workout":   day_plan.workout,
        "Meditation": day_plan.meditation,
    }
    # Workout time is derived from gym_time_minutes offset from wake
    # — a more precise implementation would store actual event times in DayPlan
    return candidates

"""
APScheduler-based reminder scheduler.
Registers one job per plan event, fires 10 minutes before each activity.
"""
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.redis import RedisJobStore
from reminders.sms import send_sms
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
tz = ZoneInfo(settings.DEFAULT_TIMEZONE)

jobstores = {"default": RedisJobStore(host="localhost", port=6379)}
scheduler = AsyncIOScheduler(jobstores=jobstores, timezone=str(tz))


def schedule_plan_reminders(events: list[dict], phone: str, plan_date: str) -> int:
    """
    Register reminder jobs for all events in a daily plan.
    Returns number of jobs scheduled.
    """
    count = 0
    for event in events:
        try:
            event_dt = datetime.strptime(
                f"{plan_date} {event['time']}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=tz)
            remind_at = event_dt - timedelta(minutes=10)

            if remind_at <= datetime.now(tz):
                continue  # skip past events

            job_id = f"reminder_{plan_date}_{event['time']}_{phone}"
            scheduler.add_job(
                send_sms,
                trigger="date",
                run_date=remind_at,
                args=[phone, event["reminder_message"]],
                id=job_id,
                replace_existing=True,
            )
            count += 1
        except Exception as e:
            logger.warning("Could not schedule event %s: %s", event.get("time"), e)

    logger.info("Scheduled %d reminders for %s", count, plan_date)
    return count


def start_scheduler() -> None:
    if not scheduler.running:
        scheduler.start()
        logger.info("APScheduler started")


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown()
        logger.info("APScheduler stopped")

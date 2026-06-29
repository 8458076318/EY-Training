"""Date/time helpers used across the project."""
from datetime import datetime, time, timezone
import pendulum


def parse_time_str(s: str) -> time:
    """Parse '06:30' → time(6, 30)."""
    h, m = map(int, s.split(":"))
    return time(h, m)


def next_week_start() -> str:
    return pendulum.now("UTC").next(pendulum.MONDAY).to_date_string()


def format_reminder_message(day_name: str, activity: str, activity_time: str) -> str:
    return f"⏰ Reminder: Your {activity} for {day_name} starts at {activity_time}. Stay consistent!"

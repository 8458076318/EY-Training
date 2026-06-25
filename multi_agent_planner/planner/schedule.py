"""
Day schedule generator. Calls OpenAI agent once per day.
Outputs a full timeline with events for the APScheduler reminders.
"""
import json
import logging
from datetime import datetime, date
from zoneinfo import ZoneInfo
from orchestrator.graph import AgentOrchestrator
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
tz = ZoneInfo(settings.DEFAULT_TIMEZONE)


SCHEDULE_PROMPT = """
Create a detailed daily schedule for an Indian professional. Return valid JSON with key "schedule"
containing a list of events. Each event must have:
  - time (HH:MM, 24h, IST)
  - activity (string)
  - duration_minutes (int)
  - category: one of [sleep, wake, workout, meditation, meal, reading, work, break]
  - reminder_message (friendly SMS text, ≤160 chars)

Requirements:
- Wake at {wake_time}, sleep at {sleep_time}
- 45 min morning workout (Mon/Wed/Fri) or yoga/meditation on other days
- 10 min evening meditation daily
- Breakfast, lunch, dinner with healthy Indian options
- 30 min book reading before bed
- Realistic for a busy professional in Bengaluru
"""


async def generate_daily_plan(
    wake_time: str = settings.DEFAULT_WAKE_TIME,
    sleep_time: str = settings.DEFAULT_SLEEP_TIME,
    user_prefs: dict | None = None,
) -> dict:
    orchestrator = AgentOrchestrator()
    prompt = SCHEDULE_PROMPT.format(wake_time=wake_time, sleep_time=sleep_time)
    if user_prefs:
        prompt += f"\nUser preferences: {user_prefs}"

    raw = await orchestrator.invoke("generate_schedule", prompt)
    try:
        parsed = json.loads(raw["result"])
        return {"date": date.today().isoformat(), "events": parsed["schedule"]}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to parse schedule: %s", e)
        return {"date": date.today().isoformat(), "events": []}

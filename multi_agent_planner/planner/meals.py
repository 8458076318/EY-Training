"""
Indian meal plan generator.
- Daily: healthy breakfast, lunch, dinner.
- Weekly: one guilt-free cheat meal (Sat or Sun).
"""
import json
import logging
from datetime import date
from orchestrator.graph import AgentOrchestrator

logger = logging.getLogger(__name__)

MEAL_PROMPT = """
Generate a healthy Indian meal plan for {day} ({meal_type}). Return JSON with key "meals" containing:
  - breakfast: {{name, calories, prep_time_mins, ingredients[]}}
  - lunch: {{name, calories, prep_time_mins, ingredients[]}}
  - dinner: {{name, calories, prep_time_mins, ingredients[]}}
  - is_cheat_day: bool

Guidelines:
- Balanced macros, low oil, high protein
- Seasonal vegetables
- Regional variety (South Indian, North Indian, Maharashtra on rotation)
{cheat_note}
"""


async def generate_meal_plan(for_date: date | None = None) -> dict:
    if for_date is None:
        for_date = date.today()

    is_weekend = for_date.weekday() >= 5
    cheat_note = "Today is the weekly cheat meal day — suggest a popular Indian street food or restaurant dish." if is_weekend else ""

    orchestrator = AgentOrchestrator()
    prompt = MEAL_PROMPT.format(
        day=for_date.strftime("%A, %d %b %Y"),
        meal_type="weekend cheat meal" if is_weekend else "weekday healthy",
        cheat_note=cheat_note,
    )
    raw = await orchestrator.invoke("generate_meals", prompt)
    try:
        parsed = json.loads(raw["result"])
        return {"date": for_date.isoformat(), **parsed}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to parse meals: %s", e)
        return {"date": for_date.isoformat(), "meals": {}}

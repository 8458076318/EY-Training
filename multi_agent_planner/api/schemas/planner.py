from pydantic import BaseModel, Field
from typing import Optional


class PlanRequest(BaseModel):
    wake_time: str = Field("06:00", pattern=r"^\d{2}:\d{2}$")
    sleep_time: str = Field("22:30", pattern=r"^\d{2}:\d{2}$")
    phone: Optional[str] = None
    preferences: Optional[dict] = None


class PlanResponse(BaseModel):
    date: str
    events: list[dict]
    meals: dict = {}
    reminders_scheduled: int = 0

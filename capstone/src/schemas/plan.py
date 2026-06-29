"""Pydantic schemas for weekly plan generation requests and responses."""
from datetime import date
from uuid import UUID
from typing import Literal
from pydantic import BaseModel, Field


class PlanPreferences(BaseModel):
    wake_time: str = Field(..., pattern=r"^\d{2}:\d{2}$")     # "06:00"
    sleep_time: str = Field(..., pattern=r"^\d{2}:\d{2}$")    # "22:30"
    diet: Literal["veg", "non-veg", "vegan"]
    includes_gym: bool = False
    includes_yoga: bool = False
    includes_meditation: bool = True
    gym_duration_minutes: int = Field(default=60, ge=30, le=120, multiple_of=15)


class GeneratePlanRequest(BaseModel):
    week_start: date
    preferences: PlanPreferences


class DayPlanResponse(BaseModel):
    day_name: str
    plan_date: date
    wake_time: str
    sleep_time: str
    breakfast: str
    lunch: str
    dinner: str
    workout: str
    meditation: str
    book_recommendation: str
    gym_time_minutes: int | None

    model_config = {"from_attributes": True}


class WeeklyPlanResponse(BaseModel):
    id: UUID
    week_start: date
    llm_provider: str
    days: list[DayPlanResponse]

    model_config = {"from_attributes": True}


class FeedbackRequest(BaseModel):
    day_plan_id: UUID
    rating: int = Field(..., ge=1, le=5)
    feedback_text: str | None = None
    category: Literal["meal", "workout", "schedule", "overall"] = "overall"

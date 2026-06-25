from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from planner.schedule import generate_daily_plan
from planner.meals import generate_meal_plan
from reminders.scheduler import schedule_plan_reminders
from history.store import save_plan
from datetime import date
import logging

router = APIRouter(prefix="/planner", tags=["Day Planner"])
logger = logging.getLogger(__name__)


class PlanRequest(BaseModel):
    wake_time: str = "06:00"
    sleep_time: str = "22:30"
    phone: Optional[str] = None
    user_id: str = "default"
    preferences: Optional[dict] = None


class MealRequest(BaseModel):
    date: Optional[str] = None
    user_id: str = "default"


@router.post("/generate")
async def generate_plan(req: PlanRequest):
    try:
        plan  = await generate_daily_plan(req.wake_time, req.sleep_time, req.preferences)
        meals = await generate_meal_plan()
        if req.phone:
            plan["reminders_scheduled"] = schedule_plan_reminders(plan["events"], req.phone, plan["date"])
        plan["meals"] = meals
        await save_plan(req.user_id, "schedule", plan)
        return plan
    except Exception as e:
        logger.error("Plan generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/meals")
async def meals_endpoint(req: MealRequest):
    try:
        plan_date = date.fromisoformat(req.date) if req.date else date.today()
        result    = await generate_meal_plan(for_date=plan_date)
        await save_plan(req.user_id, "meals", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

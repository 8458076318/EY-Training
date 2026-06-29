"""Plan generation and retrieval routes."""
from uuid import UUID
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.db.session import get_db
from src.db.models import User, WeeklyPlan, DayPlan
from src.schemas.plan import GeneratePlanRequest, WeeklyPlanResponse
from src.agents.orchestrator import AgentOrchestrator
from src.api.v1.deps import get_current_user

router = APIRouter()
orchestrator = AgentOrchestrator()


@router.post("/generate", response_model=WeeklyPlanResponse, status_code=201)
async def generate_plan(
    payload: GeneratePlanRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    context = {
        "user_id": str(current_user.id),
        "age": current_user.age,
        "gender": current_user.gender,
        "height_cm": current_user.height_cm,
        "weight_kg": current_user.weight_kg,
        "profession": current_user.profession,
        "diseases": current_user.diseases,
        "disabilities": current_user.disabilities,
        "health_profile": {
            "diseases": current_user.diseases,
            "disabilities": current_user.disabilities,
        },
        "week_start": str(payload.week_start),
        **payload.preferences.model_dump(),
    }
    plan_data = await orchestrator.generate_weekly_plan(context)

    weekly_plan = WeeklyPlan(
        user_id=current_user.id,
        week_start=payload.week_start,
        preferences=payload.preferences.model_dump(),
        plan_data=plan_data,
        llm_provider=plan_data.get("llm_provider", "unknown"),
        agent_version="1.0.0",
    )
    db.add(weekly_plan)
    await db.flush()

    for day in plan_data.get("days", []):
        db.add(DayPlan(weekly_plan_id=weekly_plan.id, **day))

    return weekly_plan


@router.get("/{plan_id}", response_model=WeeklyPlanResponse)
async def get_plan(plan_id: UUID, db: AsyncSession = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    plan = await db.get(WeeklyPlan, plan_id)
    if not plan or plan.user_id != current_user.id:
        from fastapi import HTTPException
        raise HTTPException(404, "Plan not found")
    return plan

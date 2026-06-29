"""Feedback submission — triggers Agent 3 in background."""
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.db.models import User, UserFeedback
from src.schemas.plan import FeedbackRequest
from src.agents.orchestrator import AgentOrchestrator
from src.api.v1.deps import get_current_user

router = APIRouter()
orchestrator = AgentOrchestrator()


async def _process_feedback_bg(feedback_id: str, feedback_data: dict):
    await orchestrator.process_feedback({"feedback": feedback_data, "id": feedback_id})


@router.post("/", status_code=201)
async def submit_feedback(
    payload: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    fb = UserFeedback(
        user_id=current_user.id,
        day_plan_id=payload.day_plan_id,
        rating=payload.rating,
        feedback_text=payload.feedback_text,
        category=payload.category,
    )
    db.add(fb)
    await db.flush()
    background_tasks.add_task(
        _process_feedback_bg, str(fb.id),
        {"user_id": str(current_user.id), **payload.model_dump()}
    )
    return {"id": str(fb.id), "status": "queued"}

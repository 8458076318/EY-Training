from fastapi import APIRouter
from sqlalchemy import text
from src.db.session import AsyncSessionLocal

router = APIRouter()


@router.get("/health")
async def health_check():
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
    return {"status": "healthy", "db": "ok"}

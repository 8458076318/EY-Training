from fastapi import APIRouter, Query, HTTPException
from history.store import get_history, delete_history

router = APIRouter(prefix="/history", tags=["Plan History"])


@router.get("/{user_id}")
async def list_history(user_id: str, plan_type: str | None = Query(None), limit: int = Query(10, ge=1, le=50)):
    try:
        return {"user_id": user_id, "entries": await get_history(user_id, plan_type, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}")
async def clear_history(user_id: str, plan_type: str | None = Query(None)):
    deleted = await delete_history(user_id, plan_type)
    return {"deleted_keys": deleted}

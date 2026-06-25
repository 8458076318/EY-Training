"""
Agent plan history store backed by Redis. Keeps last 90 plans per user.
"""
import json
import logging
from datetime import datetime, date
from zoneinfo import ZoneInfo
import redis.asyncio as aioredis
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
tz = ZoneInfo(settings.DEFAULT_TIMEZONE)
_redis = None


def _get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def save_plan(user_id: str, plan_type: str, payload: dict) -> str:
    r = _get_redis()
    entry = {
        "id": f"{plan_type}:{date.today().isoformat()}:{user_id}",
        "user_id": user_id, "plan_type": plan_type,
        "created_at": datetime.now(tz).isoformat(), "payload": payload,
    }
    key = f"history:{user_id}:{plan_type}"
    await r.lpush(key, json.dumps(entry))
    await r.ltrim(key, 0, 89)
    await r.expire(key, 90 * 86400)
    return entry["id"]


async def get_history(user_id: str, plan_type=None, limit: int = 10) -> list:
    r = _get_redis()
    keys = [f"history:{user_id}:{plan_type}"] if plan_type else await r.keys(f"history:{user_id}:*")
    entries = []
    for key in keys:
        raw = await r.lrange(key, 0, limit - 1)
        entries.extend(json.loads(x) for x in raw)
    entries.sort(key=lambda x: x["created_at"], reverse=True)
    return entries[:limit]


async def delete_history(user_id: str, plan_type=None) -> int:
    r = _get_redis()
    keys = [f"history:{user_id}:{plan_type}"] if plan_type else await r.keys(f"history:{user_id}:*")
    return sum([await r.delete(k) for k in keys])

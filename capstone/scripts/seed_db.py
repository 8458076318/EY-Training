"""Seed development database with a test user."""
import asyncio
from src.db.session import AsyncSessionLocal
from src.db.models import User
from src.core.security import hash_password


async def seed():
    async with AsyncSessionLocal() as db:
        user = User(
            email="demo@dayplanner.ai",
            hashed_password=hash_password("demo1234"),
            full_name="Demo User",
            age=28, gender="male", height_cm=175, weight_kg=70,
            profession="Software Engineer",
            diseases=[], disabilities=[],
        )
        db.add(user)
        await db.commit()
        print(f"✅ Seeded demo user: {user.email}")


if __name__ == "__main__":
    asyncio.run(seed())

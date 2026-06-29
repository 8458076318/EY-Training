"""Unit tests for agent orchestrator."""
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.orchestrator import AgentOrchestrator


@pytest.fixture
def orchestrator():
    return AgentOrchestrator()


@pytest.fixture
def user_context():
    return {
        "user_id": "test-user-123",
        "age": 28, "gender": "male",
        "height_cm": 175, "weight_kg": 70,
        "profession": "Engineer",
        "diseases": [], "disabilities": [],
        "health_profile": {},
        "week_start": "2025-01-06",
        "wake_time": "06:00", "sleep_time": "22:30",
        "diet": "veg", "includes_gym": True,
        "gym_duration_minutes": 60,
    }


@pytest.mark.asyncio
async def test_generate_weekly_plan(orchestrator, user_context):
    mock_plan = {
        "days": [{"day_name": "Monday", "wake_time": "06:00", "sleep_time": "22:30",
                  "breakfast": "Oats", "lunch": "Dal rice", "dinner": "Soup",
                  "workout": "Gym", "meditation": "10 min", "book_recommendation": "Atomic Habits",
                  "gym_time_minutes": 60}],
        "llm_provider": "gemini",
    }
    with patch.object(orchestrator.analyser, "safe_run", new_callable=AsyncMock) as mock_analyse, \
         patch.object(orchestrator.planner, "safe_run", new_callable=AsyncMock) as mock_plan_agent:
        mock_analyse.return_value = {"feedback_summary": "none", "health_notes": "healthy"}
        mock_plan_agent.return_value = mock_plan

        result = await orchestrator.generate_weekly_plan(user_context)

    assert result["llm_provider"] == "gemini"
    assert len(result["days"]) == 1
    mock_analyse.assert_called_once()
    mock_plan_agent.assert_called_once()

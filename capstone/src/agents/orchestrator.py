"""Agent Orchestrator — wires Agent 1 → 2 → 3 using CrewAI task graph."""
from typing import Any
import structlog

from src.agents.analyser_agent import AnalyserAgent
from src.agents.day_planner_agent import DayPlannerAgent
from src.agents.feedback_agent import FeedbackAgent

logger = structlog.get_logger(__name__)


class AgentOrchestrator:
    def __init__(self):
        self.analyser = AnalyserAgent()
        self.planner = DayPlannerAgent()
        self.feedback_agent = FeedbackAgent()

    async def generate_weekly_plan(self, user_context: dict[str, Any]) -> dict[str, Any]:
        """
        Pipeline:
          1. Analyser enriches context from history + health profile
          2. Day Planner generates the weekly plan (Gemini → GPT-4)
          3. Return plan + metadata
        """
        log = logger.bind(user_id=user_context.get("user_id"))

        log.info("orchestrator_start", stage="analyse")
        enriched = await self.analyser.safe_run(user_context)

        merged_context = {**user_context, **enriched}

        log.info("orchestrator_start", stage="plan")
        plan = await self.planner.safe_run(merged_context)

        log.info("orchestrator_complete", provider=plan.get("llm_provider"))
        return plan

    async def process_feedback(self, feedback_context: dict[str, Any]) -> dict[str, Any]:
        """Agent 3 processes a single feedback entry asynchronously."""
        return await self.feedback_agent.safe_run(feedback_context)

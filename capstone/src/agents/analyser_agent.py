"""Agent 2 — Survey / History Analyser.

Uses Groq for fast inference over user history and health vectors.
Enriches context before Agent 1 generates a plan.
"""
from typing import Any
import json
import structlog
from groq import AsyncGroq

from src.agents.base_agent import BaseAgent
from src.core.config import settings

logger = structlog.get_logger(__name__)


class AnalyserAgent(BaseAgent):
    name = "analyser"

    def __init__(self):
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return an enriched context dict with feedback_summary and health_notes."""
        history_text = self._format_history(context.get("feedback_history", []))
        prompt = f"""
You are a health and wellness analyst. Analyse the user's feedback history and health profile,
then output a concise JSON summary for the planner agent.

Health profile: {json.dumps(context.get('health_profile', {}))}
Feedback history: {history_text}

Return JSON only:
{{
  "feedback_summary": "...",
  "health_notes": "...",
  "avoid_foods": [...],
  "preferred_exercises": [...],
  "optimal_wake_pattern": "..."
}}
"""
        resp = await self._client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "No history yet."
        return "\n".join(
            f"- [{h.get('date')}] Rating {h.get('rating')}/5 ({h.get('category')}): {h.get('feedback_text', '')}"
            for h in history[-20:]   # last 20 feedback entries
        )

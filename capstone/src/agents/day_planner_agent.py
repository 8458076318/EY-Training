"""Agent 1 — Day Planner.

Primary LLM : Gemini (free tier)
Fallback LLM: GPT-4 (paid) — activated only on failure / rate-limit.
"""
from typing import Any
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.agents.base_agent import BaseAgent
from src.core.config import settings
from src.core.exceptions import LLMFallbackExhausted

logger = structlog.get_logger(__name__)

PLAN_SYSTEM_PROMPT = """
You are an expert wellness coach and day planner. Given a user's health profile and
preferences, generate a detailed, realistic 7-day wellness plan.

Output ONLY valid JSON with this shape:
{
  "days": [
    {
      "day_name": "Monday",
      "wake_time": "06:00",
      "sleep_time": "22:30",
      "breakfast": "...",
      "lunch": "...",
      "dinner": "...",
      "workout": "...",
      "meditation": "...",
      "book_recommendation": "...",
      "gym_time_minutes": 60
    }
  ],
  "llm_provider": "gemini"
}
"""


class DayPlannerAgent(BaseAgent):
    name = "day_planner"

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(context)
        try:
            return await self._call_gemini(prompt)
        except Exception as gemini_err:
            logger.warning("gemini_failed_falling_back", error=str(gemini_err))
            try:
                return await self._call_gpt4(prompt)
            except Exception as gpt_err:
                logger.error("gpt4_also_failed", error=str(gpt_err))
                raise LLMFallbackExhausted() from gpt_err

    def _build_prompt(self, ctx: dict) -> str:
        return f"""
User Health Profile:
- Age: {ctx.get('age')}, Gender: {ctx.get('gender')}
- Height: {ctx.get('height_cm')} cm, Weight: {ctx.get('weight_kg')} kg
- Profession: {ctx.get('profession')}
- Health conditions: {', '.join(ctx.get('diseases', []) or ['none'])}
- Disabilities/BP/Sugar/Heart: {', '.join(ctx.get('disabilities', []) or ['none'])}

Preferences:
- Wake: {ctx.get('wake_time')} | Sleep: {ctx.get('sleep_time')}
- Diet: {ctx.get('diet')} | Gym: {ctx.get('includes_gym')}
- Gym duration: {ctx.get('gym_duration_minutes')} mins | Yoga: {ctx.get('includes_yoga')}
- Meditation: {ctx.get('includes_meditation')}
- Week starting: {ctx.get('week_start')}

Previous feedback summary: {ctx.get('feedback_summary', 'None yet')}

Generate a 7-day personalised wellness plan.
"""

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4),
           retry=retry_if_exception_type(Exception))
    async def _call_gemini(self, prompt: str) -> dict:
        import google.generativeai as genai
        import json
        genai.configure(api_key=settings.google_api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=PLAN_SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        raw = response.text.strip().lstrip("```json").rstrip("```")
        data = json.loads(raw)
        data["llm_provider"] = "gemini"
        return data

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8),
           retry=retry_if_exception_type(Exception))
    async def _call_gpt4(self, prompt: str) -> dict:
        from openai import AsyncOpenAI
        import json
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        data["llm_provider"] = "gpt4"
        return data

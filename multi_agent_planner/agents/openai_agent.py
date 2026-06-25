import logging
from typing import Any
from openai import AsyncOpenAI
from agents.base_agent import BaseAgent
from config.settings import get_settings
from observability.metrics import agent_call_counter, agent_latency

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIAgent(BaseAgent):
    """
    Handles high-quality generation tasks:
    - Daily schedule (wake/sleep/workout/meditation)
    - Indian meal planning (breakfast, lunch, dinner, weekly cheat meal)
    """

    name = "openai"

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    @agent_latency.labels(agent="openai").time()
    async def run(self, task: str, context: dict | None = None) -> dict[str, Any]:
        agent_call_counter.labels(agent="openai").inc()
        self.log(f"Task: {task[:80]}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a personal wellness coach and nutritionist. "
                    "You specialise in Indian cuisine and Ayurvedic principles. "
                    "Always respond with structured JSON."
                ),
            },
            {"role": "user", "content": task},
        ]
        if context:
            messages.insert(1, {"role": "system", "content": f"Context: {context}"})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        return {
            "agent": self.name,
            "result": response.choices[0].message.content,
            "usage": response.usage.model_dump(),
        }

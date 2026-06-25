import logging
from typing import Any
from groq import AsyncGroq
from agents.base_agent import BaseAgent
from config.settings import get_settings
from observability.metrics import agent_call_counter, agent_latency

logger = logging.getLogger(__name__)
settings = get_settings()


class GroqAgent(BaseAgent):
    """
    Fast inference for:
    - RAG retrieval ranking
    - Hallucination detection
    - Book reading recommendations
    Uses free Groq tier (llama-3.1-8b-instant).
    """

    name = "groq"

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    @agent_latency.labels(agent="groq").time()
    async def run(self, task: str, context: dict | None = None) -> dict[str, Any]:
        agent_call_counter.labels(agent="groq").inc()
        self.log(f"Task: {task[:80]}")

        messages = [
            {"role": "system", "content": "You are a fast, accurate assistant. Respond concisely in JSON."},
            {"role": "user", "content": task},
        ]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return {
            "agent": self.name,
            "result": response.choices[0].message.content,
            "usage": {"total_tokens": response.usage.total_tokens},
        }

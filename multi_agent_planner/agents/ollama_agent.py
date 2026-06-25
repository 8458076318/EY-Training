import logging
import httpx
from typing import Any
from agents.base_agent import BaseAgent
from config.settings import get_settings
from observability.metrics import agent_call_counter, agent_latency

logger = logging.getLogger(__name__)
settings = get_settings()


class OllamaAgent(BaseAgent):
    """
    Local free inference via Ollama (replaces Ripik.ai).
    Handles: summarisation, evaluation, secondary generation.
    Models: mistral, phi-3-mini, llama3.
    Zero API cost, no rate limits.
    """

    name = "ollama"

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    @agent_latency.labels(agent="ollama").time()
    async def run(self, task: str, context: dict | None = None) -> dict[str, Any]:
        agent_call_counter.labels(agent="ollama").inc()
        self.log(f"Task: {task[:80]}")

        prompt = task
        if context:
            prompt = f"Context: {context}\n\nTask: {task}"

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "agent": self.name,
            "result": data.get("response", ""),
            "usage": {"eval_count": data.get("eval_count", 0)},
        }

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

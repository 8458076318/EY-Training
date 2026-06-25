"""
Hallucination detector: fast Groq check against retrieved context.
"""
import json
import logging
from agents.groq_agent import GroqAgent

logger = logging.getLogger(__name__)


class HallucinationDetector:
    def __init__(self):
        self.agent = GroqAgent()

    async def check(self, answer: str, context_chunks: list[str]) -> dict:
        prompt = f"""
Check if the answer is fully supported by the context. Return JSON:
{{"is_hallucination": bool, "confidence": 0-1, "unsupported_claims": ["..."]}}

Context:
{chr(10).join(context_chunks[:3])}

Answer:
{answer}
"""
        result = await self.agent.run(prompt)
        try:
            return json.loads(result["result"])
        except Exception:
            return {"is_hallucination": False, "confidence": 0.5, "unsupported_claims": []}

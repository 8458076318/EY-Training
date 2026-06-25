"""
Re-ranking agent: uses Groq to score retrieved chunks for relevance.
"""
import json
import logging
from agents.groq_agent import GroqAgent

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self):
        self.agent = GroqAgent()

    async def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        prompt = f"""
Score each chunk for relevance to the query. Return JSON array: [{{"index": 0, "score": 0.9}}, ...]

Query: {query}
Chunks:
{json.dumps([{"index": i, "text": c.get("text", str(c))} for i, c in enumerate(chunks)], indent=2)}
"""
        result = await self.agent.run(prompt)
        try:
            scores = json.loads(result["result"])
            scored = {s["index"]: s["score"] for s in scores}
            reranked = sorted(chunks, key=lambda c: scored.get(chunks.index(c), 0), reverse=True)
            return reranked
        except Exception as e:
            logger.warning("Rerank parse failed: %s — returning original order", e)
            return chunks

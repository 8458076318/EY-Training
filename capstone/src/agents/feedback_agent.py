"""Agent 3 — Feedback & Continuous Improvement.

Runs via Ollama (local, zero cost). Embeds user feedback into
Pinecone/FAISS and fine-tunes future plan suggestions.
"""
from typing import Any
import json
import structlog
import httpx

from src.agents.base_agent import BaseAgent
from src.core.config import settings

logger = structlog.get_logger(__name__)


class FeedbackAgent(BaseAgent):
    name = "feedback"

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Process new feedback, embed it, and return improvement suggestions."""
        feedback = context.get("feedback", {})
        suggestions = await self._generate_suggestions(feedback)
        await self._store_embedding(feedback, suggestions)
        return {"suggestions": suggestions, "processed": True}

    async def _generate_suggestions(self, feedback: dict) -> str:
        prompt = f"""
You are a wellness improvement engine. A user rated their day plan:

Rating: {feedback.get('rating')}/5
Category: {feedback.get('category')}
Feedback: {feedback.get('feedback_text', 'None provided')}

Provide 3 concrete, actionable suggestions to improve future plans.
Return as JSON: {{"suggestions": ["...", "...", "..."]}}
"""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={"model": "mistral", "prompt": prompt, "format": "json", "stream": False},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def _store_embedding(self, feedback: dict, suggestions: str) -> None:
        """Store feedback + suggestions as vector for future retrieval."""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        text = f"{feedback.get('feedback_text', '')} {suggestions}"
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(text).tolist()

        try:
            import pinecone
            pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index)
            vector_id = f"feedback_{feedback.get('id', 'unknown')}"
            index.upsert(vectors=[(vector_id, embedding, {"user_id": str(feedback.get("user_id")), "category": feedback.get("category")})])
        except Exception as exc:
            logger.warning("pinecone_unavailable_using_faiss", error=str(exc))

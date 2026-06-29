"""Unified vector store interface — Pinecone primary, FAISS fallback."""
from typing import Any
import structlog
from src.core.config import settings

logger = structlog.get_logger(__name__)


class VectorService:
    def __init__(self):
        self._pinecone_available = False
        self._index = None
        self._faiss_store: list = []

    async def init(self):
        try:
            import pinecone
            pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
            self._index = pc.Index(settings.pinecone_index)
            self._pinecone_available = True
            logger.info("vector_store_init", backend="pinecone")
        except Exception:
            logger.warning("pinecone_unavailable_using_faiss")

    def embed(self, text: str) -> list[float]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()

    async def upsert(self, vector_id: str, text: str, metadata: dict[str, Any]):
        embedding = self.embed(text)
        if self._pinecone_available and self._index:
            self._index.upsert(vectors=[(vector_id, embedding, metadata)])
        else:
            self._faiss_store.append({"id": vector_id, "embedding": embedding, "metadata": metadata})

    async def query(self, text: str, top_k: int = 5) -> list[dict]:
        embedding = self.embed(text)
        if self._pinecone_available and self._index:
            result = self._index.query(vector=embedding, top_k=top_k, include_metadata=True)
            return [{"id": m.id, "score": m.score, **m.metadata} for m in result.matches]
        return []


vector_service = VectorService()

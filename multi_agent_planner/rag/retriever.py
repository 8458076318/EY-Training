"""
Retriever agent: FAISS (local, free) or Pinecone (cloud).
Swaps based on USE_PINECONE setting.
"""
import logging
import numpy as np
from typing import List
from config.settings import get_settings
from rag.embeddings.embedder import get_embedder

logger = logging.getLogger(__name__)
settings = get_settings()


class Retriever:
    def __init__(self):
        self.embedder = get_embedder()
        self._index = None

    async def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        if settings.USE_PINECONE:
            return await self._pinecone_retrieve(query, top_k)
        return await self._faiss_retrieve(query, top_k)

    async def _faiss_retrieve(self, query: str, top_k: int) -> List[dict]:
        import faiss
        query_vec = np.array([await self.embedder.embed(query)], dtype="float32")
        if self._index is None:
            self._index = faiss.read_index(settings.FAISS_INDEX_PATH)
        distances, indices = self._index.search(query_vec, top_k)
        return [{"index": int(i), "score": float(d)} for i, d in zip(indices[0], distances[0]) if i != -1]

    async def _pinecone_retrieve(self, query: str, top_k: int) -> List[dict]:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX)
        vec = await self.embedder.embed(query)
        results = index.query(vector=vec, top_k=top_k, include_metadata=True)
        return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in results.matches]

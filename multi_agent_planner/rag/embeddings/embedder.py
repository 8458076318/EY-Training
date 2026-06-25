from openai import AsyncOpenAI
from config.settings import get_settings

settings = get_settings()
_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


class Embedder:
    async def embed(self, text: str) -> list[float]:
        r = await _client.embeddings.create(model="text-embedding-3-small", input=text)
        return r.data[0].embedding


_embedder = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

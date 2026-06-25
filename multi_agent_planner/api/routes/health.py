from fastapi import APIRouter
from agents.ollama_agent import OllamaAgent

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    ollama_ok = await OllamaAgent().health_check()
    return {
        "status": "ok",
        "services": {
            "ollama": "up" if ollama_ok else "down",
        },
    }

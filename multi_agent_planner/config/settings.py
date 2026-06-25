from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_DIR / ".env"

# Load the package-local .env first so it wins over any stale shell state.
load_dotenv(ENV_PATH, override=True)


class Settings(BaseSettings):
    APP_NAME: str = "Multi-Agent Day Planner"
    APP_VERSION: str = "1.0.0"
    ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_FROM_NUMBER: Optional[str] = None
    FAST2SMS_API_KEY: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX: str = "day-planner-rag"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    USE_PINECONE: bool = False
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TTL: int = 86400
    OPENAI_MODEL: str = "gpt-4o"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OLLAMA_MODEL: str = "mistral"
    PROMETHEUS_PORT: int = 8001
    LOKI_URL: str = "http://localhost:3100"
    SENTRY_DSN: Optional[str] = None
    DEFAULT_WAKE_TIME: str = "06:00"
    DEFAULT_SLEEP_TIME: str = "22:30"
    DEFAULT_TIMEZONE: str = "Asia/Kolkata"

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()

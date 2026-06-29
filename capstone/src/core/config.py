"""Central settings — loaded once at startup via pydantic-settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # App
    app_env: str = "development"
    app_name: str = "day-planner-ai"
    app_version: str = "1.0.0"
    secret_key: str
    debug: bool = False

    # Database
    database_url: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dayplanner"
    postgres_user: str = "planner_user"
    postgres_password: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # LLM
    google_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Vector DB
    pinecone_api_key: str = ""
    pinecone_env: str = "us-east-1-aws"
    pinecone_index: str = "day-planner-index"
    faiss_index_path: str = "./data/faiss_index"

    # Push Notifications (FCM)
    fcm_server_key: str = ""
    firebase_credentials_path: str = "config/firebase-service-account.json"

    # JWT
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    refresh_token_expire_days: int = 30

    # Monitoring
    log_level: str = "INFO"
    prometheus_port: int = 9090


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

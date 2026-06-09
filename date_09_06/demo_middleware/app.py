from __future__ import annotations

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from fastapi import FastAPI

from .middleware import ObservabilityMiddleware
from .observability import configure_logging
from .routes import router
from .settings import APP_TITLE, APP_VERSION, PROJECT_ROOT
from .tasks import print_task_summary


def _configure_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=True)


def create_app() -> FastAPI:
    _configure_environment()
    log = configure_logging()

    app = FastAPI(title=APP_TITLE, version=APP_VERSION)
    app.add_middleware(ObservabilityMiddleware, log=log)
    app.include_router(router)

    @app.on_event("startup")
    async def _startup_banner() -> None:
        print_task_summary()
        log.info("app.startup", backend="memory")

    @app.on_event("shutdown")
    async def _shutdown_banner() -> None:
        log.info("app.shutdown", backend="memory")

    return app


app = create_app()


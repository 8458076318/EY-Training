"""FastAPI application factory."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.core.config import settings
from src.core.exceptions import AppException
from src.core.logging import setup_logging
from src.api.v1.routes import auth, plans, feedback, health, notifications


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Day Planner AI",
        version=settings.app_version,
        docs_url="/api/docs" if settings.debug else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppException)
    async def app_exc_handler(request: Request, exc: AppException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(plans.router, prefix="/api/v1/plans", tags=["plans"])
    app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])
    app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["notifications"])

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    return app


app = create_app()

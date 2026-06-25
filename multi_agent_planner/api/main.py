from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.middleware.logging import RequestLoggingMiddleware
from api.routes import planner, rag, health
from api.routes.health_routes import router as health_bmi_router
from api.routes.history_routes import router as history_router
from reminders.scheduler import start_scheduler, stop_scheduler
from observability.metrics import start_metrics_server
from observability.loki_logger import setup_loki
from config.logging import setup_logging
from config.settings import get_settings

settings = get_settings()
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_loki()
    start_scheduler()
    if settings.ENV == "production":
        start_metrics_server()
    yield
    stop_scheduler()


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan, docs_url="/docs")
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(health.router)
app.include_router(planner.router)
app.include_router(rag.router)
app.include_router(health_bmi_router)
app.include_router(history_router)

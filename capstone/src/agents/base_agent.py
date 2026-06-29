"""Abstract base for all agents — enforces interface and telemetry."""
import time
from abc import ABC, abstractmethod
from typing import Any
import structlog
from src.monitoring.metrics import AGENT_LATENCY, AGENT_ERRORS

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent and return structured output."""
        ...

    async def safe_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Wrapper: timing, error counting, structured logging."""
        start = time.perf_counter()
        log = logger.bind(agent=self.name)
        try:
            log.info("agent_start")
            result = await self.run(context)
            elapsed = time.perf_counter() - start
            AGENT_LATENCY.labels(agent=self.name).observe(elapsed)
            log.info("agent_success", duration_s=round(elapsed, 3))
            return result
        except Exception as exc:
            AGENT_ERRORS.labels(agent=self.name).inc()
            log.error("agent_error", error=str(exc))
            raise

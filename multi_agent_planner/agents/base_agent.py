from abc import ABC, abstractmethod
from typing import Any
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all agents."""

    name: str = "base"

    @abstractmethod
    async def run(self, task: str, context: dict | None = None) -> dict[str, Any]:
        """Execute the agent task and return a structured result."""
        ...

    def log(self, msg: str) -> None:
        logger.info("[%s] %s", self.name, msg)

    def log_error(self, msg: str) -> None:
        logger.error("[%s] %s", self.name, msg)

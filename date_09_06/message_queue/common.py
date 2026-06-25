from __future__ import annotations

import time

import structlog


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def configure_logging() -> None:
    if getattr(configure_logging, "_configured", False):
        return

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )
    configure_logging._configured = True


configure_logging()
log = structlog.get_logger()

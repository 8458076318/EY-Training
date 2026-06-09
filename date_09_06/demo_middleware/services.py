from __future__ import annotations

import asyncio
from typing import Any

import logging
import httpx
import pybreaker
import structlog
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .correlation import CORRELATION_ID, inject_correlation_id
from .metrics import CIRCUIT_BREAKER_TRIPS
from .settings import (
    FRAUD_BREAKER_FAIL_MAX,
    FRAUD_BREAKER_RESET_TIMEOUT_SECONDS,
    FRAUD_RETRY_ATTEMPTS,
    FRAUD_RETRY_MAX_SECONDS,
    FRAUD_RETRY_MIN_SECONDS,
    FRAUD_RETRY_MULTIPLIER,
)

log = structlog.get_logger("demo_middleware")
retry_logger = logging.getLogger("demo_middleware.retry")

_fraud_attempt_counter = 0


class LoggingBreakerListener(pybreaker.CircuitBreakerListener):
    def state_change(self, cb: pybreaker.CircuitBreaker, old_state: Any, new_state: Any) -> None:
        if getattr(new_state, "name", "") == "open":
            CIRCUIT_BREAKER_TRIPS.labels(breaker=cb.name).inc()
        log.warning(
            "circuit_breaker.state_change",
            breaker=cb.name,
            old=str(old_state),
            new=str(new_state),
        )


fraud_breaker = pybreaker.CircuitBreaker(
    fail_max=FRAUD_BREAKER_FAIL_MAX,
    reset_timeout=FRAUD_BREAKER_RESET_TIMEOUT_SECONDS,
    listeners=[LoggingBreakerListener()],
    name="fraud-api",
)


async def call_mock_downstream_echo(*, app, amount: float, currency: str) -> dict:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        event_hooks={"request": [inject_correlation_id]},
    ) as client:
        response = await client.get(
            "/mock-downstream/echo",
            params={"amount": amount, "currency": currency},
        )
        response.raise_for_status()
        return response.json()


async def flaky_fraud_check(payload: dict) -> dict:
    global _fraud_attempt_counter
    _fraud_attempt_counter += 1

    if payload.get("simulate_timeout", True) and _fraud_attempt_counter < 3:
        log.warning(
            "fraud.retry.simulated_timeout",
            attempt=_fraud_attempt_counter,
            correlation_id=CORRELATION_ID.get(),
        )
        raise ConnectionError(f"Fraud API timeout on attempt {_fraud_attempt_counter}")

    return {
        "fraud_score": 0.02,
        "decision": "approved",
        "attempt": _fraud_attempt_counter,
    }


@retry(
    stop=stop_after_attempt(FRAUD_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=FRAUD_RETRY_MULTIPLIER,
        min=FRAUD_RETRY_MIN_SECONDS,
        max=FRAUD_RETRY_MAX_SECONDS,
    ),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(retry_logger, logging.WARNING),
    reraise=True,
)
async def call_fraud_api_with_retry(payload: dict) -> dict:
    return await flaky_fraud_check(payload)


def check_fraud_cb(payload: dict) -> dict:
    if payload.get("simulate_down", True):
        raise ConnectionError("Fraud service down")

    return {
        "fraud_score": 0.01,
        "decision": "approved",
        "circuit": "closed",
    }


def safe_fraud_check(payload: dict) -> dict:
    try:
        return fraud_breaker.call(check_fraud_cb, payload)
    except pybreaker.CircuitBreakerError:
        log.warning(
            "circuit_breaker.open",
            breaker=fraud_breaker.name,
            correlation_id=CORRELATION_ID.get(),
        )
        return {
            "fraud_score": None,
            "decision": "manual_review",
            "circuit": "open",
        }
    except (ConnectionError, TimeoutError) as exc:
        log.warning(
            "fraud.check.fallback",
            breaker=fraud_breaker.name,
            error=str(exc),
            correlation_id=CORRELATION_ID.get(),
        )
        return {
            "fraud_score": None,
            "decision": "manual_review",
            "circuit": fraud_breaker.current_state,
            "reason": str(exc),
        }
    except Exception as exc:
        log.error(
            "fraud.check.error",
            breaker=fraud_breaker.name,
            error=str(exc),
            correlation_id=CORRELATION_ID.get(),
        )
        return {
            "fraud_score": None,
            "decision": "error",
            "reason": str(exc),
        }


async def assess_fraud(payload: dict) -> dict:
    retry_result = await call_fraud_api_with_retry(payload)
    breaker_result = await asyncio.to_thread(safe_fraud_check, payload)
    return {
        "retry_result": retry_result,
        "breaker_result": breaker_result,
    }

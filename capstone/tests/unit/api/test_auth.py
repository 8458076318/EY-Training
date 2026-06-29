"""Unit tests for auth routes."""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from src.api.main import app


@pytest.mark.asyncio
async def test_health_check():
    with patch("src.api.v1.routes.health.AsyncSessionLocal") as mock_session:
        mock_session.return_value.__aenter__ = AsyncMock()
        mock_session.return_value.__aexit__ = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

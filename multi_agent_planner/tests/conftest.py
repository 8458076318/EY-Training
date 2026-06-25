import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def mock_openai_agent():
    with patch("agents.openai_agent.OpenAIAgent.run", new_callable=AsyncMock) as m:
        m.return_value = {"agent": "openai", "result": "{}","usage": {}}
        yield m


@pytest.fixture
def mock_groq_agent():
    with patch("agents.groq_agent.GroqAgent.run", new_callable=AsyncMock) as m:
        m.return_value = {"agent": "groq", "result": "{}","usage": {}}
        yield m

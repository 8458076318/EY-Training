import pytest
from orchestrator.router import AgentRouter


def test_router_day_planner_tasks():
    r = AgentRouter()
    assert r.select("generate_schedule") == "openai"
    assert r.select("generate_meals") == "openai"


def test_router_rag_tasks():
    r = AgentRouter()
    assert r.select("rag_query") == "groq"
    assert r.select("hallucination_check") == "groq"


def test_router_free_tasks():
    r = AgentRouter()
    assert r.select("summarise") == "ollama"
    assert r.select("evaluate") == "ollama"


def test_router_unknown_defaults_ollama():
    r = AgentRouter()
    assert r.select("unknown_task") == "ollama"

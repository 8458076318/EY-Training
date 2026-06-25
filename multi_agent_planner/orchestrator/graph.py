"""
LangGraph state machine for multi-agent orchestration.
Routes tasks to the cheapest capable agent.
"""
import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from orchestrator.router import AgentRouter
from agents import OpenAIAgent, GroqAgent, OllamaAgent

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    task_type: str
    user_input: str
    context: dict
    openai_result: dict | None
    groq_result: dict | None
    ollama_result: dict | None
    final_output: dict | None
    error: str | None


class AgentOrchestrator:
    def __init__(self):
        self.router = AgentRouter()
        self.openai = OpenAIAgent()
        self.groq = GroqAgent()
        self.ollama = OllamaAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)

        g.add_node("route", self._route_node)
        g.add_node("openai_node", self._openai_node)
        g.add_node("groq_node", self._groq_node)
        g.add_node("ollama_node", self._ollama_node)
        g.add_node("aggregate", self._aggregate_node)

        g.set_entry_point("route")
        g.add_conditional_edges("route", self._dispatch, {
            "openai": "openai_node",
            "groq": "groq_node",
            "ollama": "ollama_node",
        })
        g.add_edge("openai_node", "aggregate")
        g.add_edge("groq_node", "aggregate")
        g.add_edge("ollama_node", "aggregate")
        g.add_edge("aggregate", END)

        return g.compile()

    async def _route_node(self, state: AgentState) -> AgentState:
        agent = self.router.select(state["task_type"])
        state["context"]["selected_agent"] = agent
        return state

    async def _openai_node(self, state: AgentState) -> AgentState:
        result = await self.openai.run(state["user_input"], state["context"])
        state["openai_result"] = result
        return state

    async def _groq_node(self, state: AgentState) -> AgentState:
        result = await self.groq.run(state["user_input"], state["context"])
        state["groq_result"] = result
        return state

    async def _ollama_node(self, state: AgentState) -> AgentState:
        result = await self.ollama.run(state["user_input"], state["context"])
        state["ollama_result"] = result
        return state

    async def _aggregate_node(self, state: AgentState) -> AgentState:
        result = (
            state.get("openai_result")
            or state.get("groq_result")
            or state.get("ollama_result")
        )
        state["final_output"] = result
        return state

    def _dispatch(self, state: AgentState) -> str:
        return state["context"].get("selected_agent", "ollama")

    async def invoke(self, task_type: str, user_input: str, context: dict | None = None) -> dict:
        initial: AgentState = {
            "task_type": task_type,
            "user_input": user_input,
            "context": context or {},
            "openai_result": None,
            "groq_result": None,
            "ollama_result": None,
            "final_output": None,
            "error": None,
        }
        result = await self.graph.ainvoke(initial)
        return result["final_output"]

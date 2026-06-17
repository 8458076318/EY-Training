from __future__ import annotations

import json
import os
import sys
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, StateGraph


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class AgentState(TypedDict):
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iterations: int


def build_llm() -> ChatGroq:
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to C:\\Training\\AI-ML-Training-Projects\\.env "
            "or the current environment before running this script."
        )

    return ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=groq_api_key,
    )


def planner(state: AgentState, llm: ChatGroq) -> AgentState:
    system = (
        "You are a planning agent. Break the user's goal into at most 5 concrete, "
        "actionable tasks. Respond ONLY with a valid JSON array of strings. No preamble, "
        "no markdown."
    )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Goal: {state['goal']}"),
    ]
    response = llm.invoke(messages).content.strip()

    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        tasks = json.loads(clean)
        if not isinstance(tasks, list):
            raise ValueError("Planner response was not a JSON list")
    except Exception:
        tasks = [response]

    print(f"\n[Planner] Generated {len(tasks)} tasks:")
    for i, task in enumerate(tasks, start=1):
        print(f"  {i}. {task}")

    return {**state, "tasks": tasks}


def executor(state: AgentState, llm: ChatGroq, search: DuckDuckGoSearchRun) -> AgentState:
    results: List[str] = []
    critique_ctx = ""
    if state["critique"]:
        critique_ctx = (
            f"\n\nYour previous attempt was rejected. Previous critique: {state['critique']}"
        )

    for task in state["tasks"]:
        system = (
            "You are an execution agent. Complete the task thoroughly. Use web search "
            f"if you need current information. {critique_ctx}"
        )

        search_ctx = ""
        try:
            search_result = search.run(task[:100])
            search_ctx = f"\n\nWeb search result for context:\n{search_result[:800]}"
        except Exception:
            pass

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Task: {task}{search_ctx}"),
        ]

        result = llm.invoke(messages).content
        results.append(result)
        print(f"\n[Executor] Task: {task[:60]}...\n  Result: {result}")

    return {**state, "results": results, "iterations": state["iterations"] + 1}


def verifier(state: AgentState, llm: ChatGroq) -> AgentState:
    # Safety net: approve after 3 iterations regardless.
    if state["iterations"] >= 3:
        print("[Verifier] Max iterations reached - force approving.")
        return {**state, "approved": True}

    combined_results = "\n\n".join(
        f"Task {i + 1}: {task}\nResult: {result}"
        for i, (task, result) in enumerate(zip(state["tasks"], state["results"]))
    )
    system = (
        "You are a quality verifier. Evaluate the results against the original goal "
        "using this rubric:\n"
        "- Completeness: Does it fully address the goal? (0-0.4)\n"
        "- Accuracy: Is the information correct and specific? (0-0.3)\n"
        "- Clarity: Is it well-structured and clear? (0-0.3)\n"
        "- Latency: Does it take a reasonable amount of time to complete? (0-0.3)\n"
        "- Tokens: Number of tokens used (0-10000)\n"
        "Sum the scores for a total between 0.0 and 1.0.\n"
        'Respond ONLY as JSON: {"score": 0.9, "completeness_score": 0.35, '
        '"accuracy_score": 0.2, "clarity_score": 0.15, "approved": true, '
        '"critique": "..."}'
    )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Original goal: {state['goal']}\n\nResults:\n{combined_results}"),
    ]
    raw = llm.invoke(messages).content.strip()

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        verdict = json.loads(clean)
        approved = bool(verdict.get("approved", False))
        critique = verdict.get("critique", "")
        score = float(verdict.get("score", 0))
    except Exception:
        approved, critique, score = False, raw, 0.0

    print(f"\n[Verifier] Score: {score:.2f} | Approved: {approved}")
    if not approved:
        print(f"  Critique: {critique}")

    return {**state, "approved": approved, "critique": critique}


def route_after_verify(state: AgentState) -> str:
    return "end" if state["approved"] else "executor"


def build_app(llm: ChatGroq, search: DuckDuckGoSearchRun):
    graph = StateGraph(AgentState)

    graph.add_node("planner", lambda state: planner(state, llm))
    graph.add_node("executor", lambda state: executor(state, llm, search))
    graph.add_node("verifier", lambda state: verifier(state, llm))

    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verify,
        {"end": END, "executor": "executor"},
    )

    graph.set_entry_point("planner")
    return graph.compile()


def main() -> None:
    llm = build_llm()
    search = DuckDuckGoSearchRun()
    app = build_app(llm, search)

    initial_state: AgentState = {
        "goal": "Research and summarise the top 3 trends in agriculture for 2025",
        "tasks": [],
        "results": [],
        "critique": "",
        "approved": False,
        "iterations": 0,
    }

    final_state = app.invoke(initial_state)
    print("\n[Final State]")
    print(final_state)


if __name__ == "__main__":
    main()

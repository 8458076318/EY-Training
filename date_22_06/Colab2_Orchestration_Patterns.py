"""Colab 2: Orchestration Patterns in Depth.

Converted from the notebook `Colab2_Orchestration_Patterns.ipynb` into a
plain Python script for local execution.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from getpass import getpass

# %% markdown
# # Colab 2 · Orchestration Patterns in Depth
# ### Day 19 — Agent Orchestration with AutoGen Studio & Semantic Kernel
#
# Colab 1 used the simplest pattern (RoundRobin). Now you'll wire the same
# research team four different ways and watch how the control flow changes
# - then peek at the same idea in Semantic Kernel.
#
# You will build:
# 1. SelectorGroupChat — an LLM decides who speaks next.
# 2. Swarm — agents hand off to each other directly.
# 3. GraphFlow — a deterministic researcher -> writer -> reviewer graph.
# 4. A function tool the researcher calls to delegate real work.
# 5. A Semantic Kernel sequential-orchestration mini-example.
#
# ~60 min including the extension tasks at the end.
#
# > The patterns are the lesson. AutoGen, Semantic Kernel and the Microsoft
# > Agent Framework all expose this same family - RoundRobin/Sequential,
# > Selector/GroupChat, Swarm/Handoff, Graph, Magentic.

# %% markdown
# ## 0 · Setup

# Notebook-only install cell:
# %pip install -q -U "autogen-agentchat" "autogen-ext[openai]"
# print("AutoGen installed.")

try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import (
        HandoffTermination,
        MaxMessageTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.ui import Console
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "Missing AutoGen packages. Install 'autogen-agentchat' and "
        "'autogen-ext[openai]' to run this script."
    ) from exc


def _load_repo_env() -> Path | None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value

    return env_path


REPO_ENV_PATH = _load_repo_env()
if REPO_ENV_PATH:
    print(f"Loaded environment variables from {REPO_ENV_PATH}")


def _ensure_openai_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    if os.isatty(0):
        os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI API key: ")
        return
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Export it in the environment or run in an "
        "interactive terminal."
    )


_ensure_openai_key()
MODEL_NAME = (
    os.environ.get("OPENAI_MODEL")
    or os.environ.get("OPENAI_MODEL_ID")
    or os.environ.get("OPENAI_CHAT_MODEL")
    or "gpt-4o-mini"
)
model_client = OpenAIChatCompletionClient(model=MODEL_NAME)
print("Ready.")


def make_specialists():
    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        description="Breaks a topic into 2-3 concrete sub-questions to research.",
        system_message="You plan research. Given a topic, list 2-3 specific sub-questions. Keep it short.",
    )
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        description="Answers factual sub-questions with concise bullet points.",
        system_message="You answer the planner's sub-questions with short factual bullets.",
    )
    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        description="Turns research bullets into a tight 4-sentence summary, ending with APPROVE.",
        system_message="Write a tight 4-sentence summary from the research. End your message with APPROVE.",
    )
    return planner, researcher, writer


print("Specialist factory ready.")


def force_planner_first(messages):
    """Force the planner to speak first, then let the model router take over."""
    if len(messages) <= 1:
        return "planner"
    return None


async def run_selector_group_chat() -> None:
    from autogen_agentchat.teams import SelectorGroupChat

    planner, researcher, writer = make_specialists()
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(8)

    selector_team = SelectorGroupChat(
        participants=[planner, researcher, writer],
        model_client=model_client,  # the "router" brain
        selector_func=force_planner_first,
        termination_condition=termination,
        allow_repeated_speaker=False,
    )

    await Console(
        selector_team.run_stream(
            task="Topic: why are reusable cups better than disposable ones?"
        )
    )


async def run_swarm() -> None:
    from autogen_agentchat.teams import Swarm

    triage = AssistantAgent(
        name="triage",
        model_client=model_client,
        handoffs=["billing", "tech"],
        description="Front desk: routes the user to the right specialist.",
        system_message="Decide if the request is about billing or tech, then hand off to that agent.",
    )
    billing = AssistantAgent(
        name="billing",
        model_client=model_client,
        handoffs=["triage"],
        description="Handles billing and refund questions.",
        system_message="Answer the billing question. If it's not billing, hand back to triage.",
    )
    tech = AssistantAgent(
        name="tech",
        model_client=model_client,
        handoffs=["triage"],
        description="Handles technical troubleshooting.",
        system_message="Answer the tech question concisely, then say DONE.",
    )

    swarm = Swarm(
        participants=[triage, billing, tech],  # Swarm starts with the first agent
        termination_condition=TextMentionTermination("DONE") | MaxMessageTermination(8),
    )

    await Console(
        swarm.run_stream(task="My app keeps crashing when I open the camera.")
    )


async def run_graph_flow() -> None:
    from autogen_agentchat.agents import SocietyOfMindAgent
    from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

    planner, researcher, writer = make_specialists()
    fact_checker = AssistantAgent(
        name="fact_checker",
        model_client=model_client,
        description="Checks the researcher's notes for obvious issues and returns a concise approval.",
        system_message="Review the researcher's notes for accuracy. If the notes look good, reply with APPROVE. Keep it short.",
    )
    from autogen_agentchat.teams import RoundRobinGroupChat

    inner_research_team = RoundRobinGroupChat(
        [researcher, fact_checker],
        termination_condition=TextMentionTermination("APPROVE") | MaxMessageTermination(4),
    )
    research_team = SocietyOfMindAgent(
        name="research_team",
        team=inner_research_team,
        model_client=model_client,
        description="A nested research team that collaborates before handing a synthesized result back to the graph.",
    )

    builder = DiGraphBuilder()
    builder.add_node(planner).add_node(research_team).add_node(writer)
    builder.add_edge(planner, research_team).add_edge(research_team, writer)
    graph = builder.build()

    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
    )

    await Console(flow.run_stream(task="Topic: the benefits of cycling to work."))


def web_search(query: str) -> str:
    """Look up a query and return a short text snippet. (Stub for the workshop.)"""
    canned = {
        "reusable cup co2": "A reusable cup typically breaks even vs. disposables after ~20-100 uses.",
        "default": "No exact match; returning a generic note that reusable goods amortise their footprint with use.",
    }
    return canned.get(query.lower().strip(), canned["default"])


async def run_tool_demo() -> None:
    researcher_with_tool = AssistantAgent(
        name="researcher",
        model_client=model_client,
        tools=[web_search],
        description="Researches facts, calling web_search when it needs evidence.",
        system_message="Use the web_search tool to find a figure, then report it in one bullet. End with APPROVE.",
    )

    from autogen_agentchat.teams import RoundRobinGroupChat

    tool_team = RoundRobinGroupChat(
        [researcher_with_tool],
        termination_condition=TextMentionTermination("APPROVE") | MaxMessageTermination(4),
    )
    await Console(
        tool_team.run_stream(
            task="Find a figure on reusable cup CO2 break-even and report it."
        )
    )


# Notebook-only install cell:
# %pip install -q -U semantic-kernel
# print("Semantic Kernel installed.")


async def run_semantic_kernel_demo() -> None:
    # The shape of an SK sequential orchestration: two agents, output of one feeds the next.
    # Wrapped in try/except because SK's orchestration symbols move between versions.
    try:
        from semantic_kernel.agents import ChatCompletionAgent
        from semantic_kernel.agents.orchestration.sequential import SequentialOrchestration
        from semantic_kernel.agents.runtime import InProcessRuntime
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "Missing Semantic Kernel packages. Install 'semantic-kernel' to run this section."
        ) from exc

    service = OpenAIChatCompletion(ai_model_id=MODEL_NAME)

    sk_writer = ChatCompletionAgent(
        name="writer",
        service=service,
        instructions="Write one short paragraph on the given topic.",
    )
    sk_editor = ChatCompletionAgent(
        name="editor",
        service=service,
        instructions="Tighten the paragraph you receive into two crisp sentences.",
    )

    orchestration = SequentialOrchestration(members=[sk_writer, sk_editor])
    runtime = InProcessRuntime()
    runtime.start()

    try:
        result = await orchestration.invoke(
            task="The benefits of walking meetings.", runtime=runtime
        )
        print(await result.get())
    finally:
        await runtime.stop_when_idle()


async def main() -> None:
    await run_selector_group_chat()
    await run_swarm()
    await run_graph_flow()
    await run_tool_demo()
    await run_semantic_kernel_demo()
    await model_client.close()
    print("Client closed. On to the capstone!")


# === Extension 1 scaffold: custom selector_func ===
# def force_planner_first(messages):
#     # Return "planner" for the first turn, then None to let the model decide.
#     if len(messages) <= 1:
#         return "planner"
#     return None
#
# planner, researcher, writer = make_specialists()
# team = SelectorGroupChat(
#     participants=[planner, researcher, writer],
#     model_client=model_client,
#     selector_func=force_planner_first,
#     termination_condition=TextMentionTermination("APPROVE") | MaxMessageTermination(8),
# )
# await Console(team.run_stream(task="Topic: benefits of a standing desk."))

# === Extension 2 scaffold: conditional edge (writer loops on REVISE) ===
# builder = DiGraphBuilder()
# builder.add_node(planner).add_node(researcher).add_node(writer).add_node(reviewer)
# builder.add_edge(planner, researcher).add_edge(researcher, writer).add_edge(writer, reviewer)
# # Conditional: reviewer -> writer only if "REVISE" appears; else the run ends.
# builder.add_edge(reviewer, writer, condition=lambda msg: "REVISE" in msg.to_text())
# graph = builder.build()
# flow = GraphFlow(participants=builder.get_participants(), graph=graph)

# === Extension 3 scaffold: nest a team inside a node ===
# fact_checker = AssistantAgent(
#     name="fact_checker",
#     model_client=model_client,
#     system_message="Review the researcher's notes and reply with APPROVE if they look good.",
# )
# inner_research_team = RoundRobinGroupChat(
#     [researcher, fact_checker],
#     termination_condition=TextMentionTermination("APPROVE") | MaxMessageTermination(4),
# )
# research_team = SocietyOfMindAgent(
#     name="research_team",
#     team=inner_research_team,
#     model_client=model_client,
# )
# builder = DiGraphBuilder()
# builder.add_node(planner).add_node(research_team).add_node(writer)
# builder.add_edge(planner, research_team).add_edge(research_team, writer)


if __name__ == "__main__":
    asyncio.run(main())

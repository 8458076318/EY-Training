"""Notebook conversion for the researcher/supervisor multi-agent architecture.

This script keeps the original notebook structure but makes it usable as a
plain Python module. It also adds a notebook-friendly Mermaid diagram renderer
so the LangGraph architecture can be shown from IPython when available.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - optional notebook dependency
    Markdown = None  # type: ignore[assignment]

    def display(obj):  # type: ignore[override]
        print(obj)


try:  # pragma: no cover - optional runtime dependency
    from langchain_groq import ChatGroq
except Exception as exc:  # pragma: no cover - optional runtime dependency
    ChatGroq = None  # type: ignore[assignment]
    _CHATGROQ_IMPORT_ERROR = exc


try:  # pragma: no cover - optional runtime dependency
    from langchain_community.tools.tavily_search import TavilySearchResults
except Exception as exc:  # pragma: no cover - optional runtime dependency
    TavilySearchResults = None  # type: ignore[assignment]
    _TAVILY_IMPORT_ERROR = exc


try:  # pragma: no cover - optional runtime dependency
    from langgraph.graph import END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
except Exception as exc:  # pragma: no cover - optional runtime dependency
    END = "END"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    MemorySaver = None  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc


class AgentState(TypedDict):
    task: str
    research_notes: Annotated[List[str], operator.add]
    draft: str
    next_node: str
    retry_count: int
    revision_feedback: str


class Router(BaseModel):
    """Decide which worker to call next."""

    next_worker: Literal["researcher", "writer", "FINISH"] = Field(
        description="The next node to act"
    )
    instructions: str = Field(description="Specific instructions for the worker")
    is_critical: bool = Field(description="If True, system will pause for human review")


def build_mermaid_diagram() -> str:
    """Return a Mermaid diagram for the multi-agent workflow."""

    return "\n".join(
        [
            "flowchart LR",
            "    T([1. Task Input]) --> S{2. Supervisor}",
            "    S -->|research brief| R[3. Researcher]",
            "    R -->|research notes| S",
            "    S -->|write draft| W[4. Writer]",
            "    W -->|final draft| S",
            "    S -->|FINISH| E((5. End))",
            "    S -. interrupt_before writer .-> H[(Human Review / Approval)]",
        ]
    )


def display_langgraph_diagram() -> str:
    """Display the graph in IPython, or print the Mermaid text as a fallback."""

    mermaid = build_mermaid_diagram()
    block = f"```mermaid\n{mermaid}\n```"
    try:
        display(Markdown(block))  # type: ignore[misc]
    except Exception:
        print(mermaid)
    return mermaid


def save_flow_diagram_png(
    output_path: Optional[Path] = None,
) -> Path:
    """Save a simple flow diagram PNG for the architecture."""

    if output_path is None:
        output_path = Path(__file__).with_name("Researcher_Supervisor_flow_diagram.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    title_color = "#0f172a"
    subtitle_color = "#475569"
    nodes = {
        "task": (1.2, 5.0, "1\nTask Input\nUser request"),
        "supervisor": (4.2, 5.0, "2\nSupervisor\nRoutes the work"),
        "researcher": (7.4, 7.7, "3\nResearcher\nCollects sources"),
        "writer": (10.8, 5.0, "4\nWriter\nDrafts the answer"),
        "pause": (10.8, 1.8, "Human Review\nApprove / revise"),
        "end": (14.0, 5.0, "5\nEnd\nFinal result"),
    }

    def box(
        x: float,
        y: float,
        text: str,
        width: float = 2.35,
        height: float = 1.35,
        facecolor: str = "#dbeafe",
        edgecolor: str = "#1f2937",
    ):
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=2.0,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            color="#111827",
            linespacing=1.15,
            fontweight="semibold",
        )

    def arrow(start, end, label: str = "", color: str = "#334155", style: str = "-", y_offset: float = 0.0):
        sx, sy = start
        ex, ey = end
        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="->",
                lw=2.2,
                color=color,
                linestyle=style,
                shrinkA=18,
                shrinkB=18,
            ),
        )
        if label:
            ax.text(
                (sx + ex) / 2,
                (sy + ey) / 2 + 0.25 + y_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9),
            )

    for _, (x, y, label) in nodes.items():
        if label.startswith("Human Review"):
            box(x, y, label, width=2.8, height=1.2, facecolor="#fff7cc", edgecolor="#b45309")
        elif label.startswith("5"):
            box(x, y, label, width=2.1, height=1.25, facecolor="#dcfce7", edgecolor="#166534")
        elif label.startswith("3"):
            box(x, y, label, width=2.4, height=1.4, facecolor="#ede9fe", edgecolor="#6d28d9")
        elif label.startswith("4"):
            box(x, y, label, width=2.4, height=1.4, facecolor="#dbeafe", edgecolor="#1d4ed8")
        elif label.startswith("2"):
            box(x, y, label, width=2.5, height=1.4, facecolor="#e0f2fe", edgecolor="#0369a1")
        else:
            box(x, y, label, width=2.5, height=1.4, facecolor="#f1f5f9", edgecolor="#334155")

    arrow((2.35, 5.0), (3.0, 5.0), "submit task", y_offset=0.05)
    arrow((5.4, 5.0), (6.3, 6.95), "research brief", y_offset=0.25)
    arrow((8.5, 6.9), (5.2, 5.4), "research notes", y_offset=0.4)
    arrow((5.4, 5.0), (9.6, 5.0), "draft request", y_offset=0.05)
    arrow((12.0, 5.0), (13.0, 5.0), "final output", color="#166534")
    arrow((10.8, 4.35), (10.8, 2.55), "manual approval", style="--", color="#b45309", y_offset=0.0)

    ax.text(
        8.0,
        8.9,
        "Researcher / Supervisor Multi-Agent Flow",
        fontsize=18,
        fontweight="bold",
        color=title_color,
        ha="center",
    )
    ax.text(
        8.0,
        8.35,
        "The supervisor routes work between research and writing, with an optional human review pause before drafting.",
        fontsize=11,
        color=subtitle_color,
        ha="center",
    )
    ax.text(1.2, 3.1, "Start", fontsize=10, color=subtitle_color, ha="center")
    ax.text(7.9, 7.25, "Research loop", fontsize=10, color=subtitle_color, ha="center")
    ax.text(10.8, 0.95, "Approval branch", fontsize=10, color=subtitle_color, ha="center")
    ax.text(14.0, 2.95, "Completion", fontsize=10, color=subtitle_color, ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_researcher(search_tool):
    def researcher(state: AgentState):
        print("[RESEARCHER] collecting notes")
        query = state["task"]
        results = search_tool.invoke(query)
        print(results)
        return {"research_notes": [str(results)], "retry_count": 0}

    return researcher


def make_writer(llm):
    def writer(state: AgentState):
        print("[WRITER] drafting the report")
        context = "\n".join(state["research_notes"])
        res = llm.invoke(f"Write a report on {state['task']} using: {context}")
        return {"draft": res.content}

    return writer


def make_supervisor(llm):
    def supervisor(state: AgentState):
        print("[SUPERVISOR] reviewing state")
        structured_llm = llm.with_structured_output(Router)

        prompt = f"""
Task: {state['task']}
Notes collected: {len(state['research_notes'])}
Current Draft: {state['draft'][:100]}...
If you have something in research_notes, select writer
"""
        print(
            f"[SUPERVISOR] sees {len(state['research_notes'])} notes and draft length {len(state['draft'])}"
        )
        decision = structured_llm.invoke(prompt)
        return {
            "next_node": decision.next_worker,
            "revision_feedback": decision.instructions,
        }

    return supervisor


def build_graph():
    """Build and compile the LangGraph workflow when dependencies are available."""

    if StateGraph is None or MemorySaver is None:
        raise RuntimeError(
            "langgraph could not be imported in this environment. "
            "The notebook conversion is still usable for diagram generation, "
            "but workflow execution needs a compatible langgraph install."
        )
    if ChatGroq is None or TavilySearchResults is None:
        raise RuntimeError(
            "The runtime agent dependencies are missing. "
            "Install langchain_groq and langchain_community, then provide the "
            "required API keys before executing the workflow."
        )

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    search_tool = TavilySearchResults(k=2)

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node("researcher", make_researcher(search_tool))
    builder.add_node("writer", make_writer(llm))

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next_node"],
        {"researcher": "researcher", "writer": "writer", "FINISH": END},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["writer"],
    )


def run_demo():
    """Run the original notebook demonstration if the runtime dependencies exist."""

    graph = build_graph()
    config = {"configurable": {"thread_id": "workshop_user_1"}}
    initial_input = {
        "task": "Impact of LPU architecture on AI inference speeds",
        "research_notes": [],
        "retry_count": 0,
        "draft": "",
    }

    print("--- STARTING GRAPH ---")
    for event in graph.stream(initial_input, config, stream_mode="values"):
        if "next_node" in event:
            print(f"Moving to: {event['next_node']}")

    snapshot = graph.get_state(config)
    if snapshot.next:
        print(f"\n[SYSTEM PAUSED] Next step is: {snapshot.next}")
        print(f"Feedback from Supervisor: {snapshot.values['revision_feedback']}")
        print("\n--- RESUMING AFTER PAUSE ---")

    print("--- RESUMING GRAPH ---\n")
    for event in graph.stream(None, config, stream_mode="values"):
        if "next_node" in event:
            print(f"Moving to: {event['next_node']}")
        elif "draft" in event:
            print(f"\n--- FINAL DRAFT ---:\n{event['draft']}")


def main() -> None:
    """Entry point for the converted notebook."""

    print("LangGraph diagram for the researcher/supervisor workflow:")
    display_langgraph_diagram()
    png_path = save_flow_diagram_png()
    print(f"Saved PNG flow diagram to: {png_path}")

    # Keep the notebook demo opt-in so the file remains import-safe.
    if Path(__file__).with_suffix(".run").exists():
        run_demo()


if __name__ == "__main__":
    main()

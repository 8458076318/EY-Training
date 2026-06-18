from __future__ import annotations

import json
import html
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from groq import Groq

try:
    import streamlit as st
except Exception:  # pragma: no cover - CLI fallback still works without Streamlit.
    st = None  # type: ignore[assignment]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class AgentState(TypedDict, total=False):
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iterations: int
    max_iterations: int
    score: float
    trace: List[Dict[str, Any]]


class _LLMResult:
    def __init__(self, content: str) -> None:
        self.content = content


class GroqChatLLM:
    def __init__(self, client: Groq, model_name: str, temperature: float = 0.0) -> None:
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, messages: List[Any]) -> _LLMResult:
        payload = []
        for message in messages:
            role = getattr(message, "type", "")
            if role == "human":
                role = "user"
            elif role == "system":
                role = "system"
            else:
                role = "assistant"
            payload.append(
                {
                    "role": role,
                    "content": getattr(message, "content", str(message)),
                }
            )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=payload,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        return _LLMResult(content.strip())


class LocalWorkflowApp:
    def __init__(self, llm: GroqChatLLM, search: DuckDuckGoSearchRun | None) -> None:
        self.llm = llm
        self.search = search

    def invoke(self, state: AgentState) -> AgentState:
        state = planner(state, self.llm)
        while True:
            state = executor(state, self.llm, self.search)
            state = verifier(state, self.llm)
            if route_after_verify(state) == "end":
                return state


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)


def build_llm() -> GroqChatLLM:
    load_project_env()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to C:\\Training\\AI-ML-Training-Projects\\.env "
            "or the current environment before running this app."
        )

    return GroqChatLLM(
        client=Groq(api_key=groq_api_key),
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
    )


def build_search() -> DuckDuckGoSearchRun:
    return DuckDuckGoSearchRun()


def append_trace(state: AgentState, stage: str, title: str, detail: str, level: str = "info") -> None:
    trace = list(state.get("trace", []))
    trace.append(
        {
            "stage": stage,
            "title": title,
            "detail": detail,
            "level": level,
        }
    )
    state["trace"] = trace


def parse_json_list(raw: str) -> List[str]:
    clean = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list")
    return [str(item).strip() for item in parsed if str(item).strip()]


def parse_json_object(raw: str) -> Dict[str, Any]:
    clean = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def planner(state: AgentState, llm: GroqChatLLM) -> AgentState:
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
        tasks = parse_json_list(response)
        if not tasks:
            raise ValueError("Planner returned an empty list")
    except Exception:
        tasks = [response]

    append_trace(
        state,
        "planner",
        "Plan generated",
        f"Created {len(tasks)} task(s) for the goal.",
    )

    return {
        **state,
        "tasks": tasks,
        "trace": state["trace"],
    }


def executor(
    state: AgentState,
    llm: GroqChatLLM,
    search: DuckDuckGoSearchRun | None,
) -> AgentState:
    results: List[str] = []
    critique_ctx = ""
    if state.get("critique"):
        critique_ctx = (
            f"\n\nYour previous attempt was rejected. Previous critique: {state['critique']}"
        )

    for index, task in enumerate(state.get("tasks", []), start=1):
        system = (
            "You are an execution agent. Complete the task thoroughly and clearly. "
            f"{critique_ctx}"
        )

        search_ctx = ""
        if search is not None:
            try:
                search_result = search.run(task[:140])
                search_ctx = f"\n\nWeb search context:\n{search_result[:900]}"
            except Exception:
                search_ctx = ""

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Task {index}: {task}{search_ctx}"),
        ]

        result = llm.invoke(messages).content.strip()
        results.append(result)

        append_trace(
            state,
            "executor",
            f"Task {index} executed",
            f"{task[:120]}",
        )

    next_iterations = state.get("iterations", 0) + 1
    append_trace(
        state,
        "executor",
        "Execution pass complete",
        f"Finished {len(results)} task(s) in iteration {next_iterations}.",
    )

    return {
        **state,
        "results": results,
        "iterations": next_iterations,
        "trace": state["trace"],
    }


def verifier(state: AgentState, llm: GroqChatLLM) -> AgentState:
    max_iterations = state.get("max_iterations", 3)

    if state.get("iterations", 0) >= max_iterations:
        append_trace(
            state,
            "verifier",
            "Iteration cap reached",
            "Auto-approved because the workflow reached the configured maximum iterations.",
            "warning",
        )
        return {
            **state,
            "approved": True,
            "score": 1.0,
            "critique": "Auto-approved after reaching the configured iteration cap.",
            "trace": state["trace"],
        }

    combined_results = "\n\n".join(
        f"Task {i + 1}: {task}\nResult: {result}"
        for i, (task, result) in enumerate(zip(state.get("tasks", []), state.get("results", [])))
    )
    system = (
        "You are a quality verifier. Evaluate the results against the original goal "
        "using this rubric:\n"
        "- Completeness: Does it fully address the goal? (0-0.4)\n"
        "- Accuracy: Is the information correct and specific? (0-0.3)\n"
        "- Clarity: Is it well-structured and clear? (0-0.3)\n"
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
        verdict = parse_json_object(raw)
        approved = bool(verdict.get("approved", False))
        critique = str(verdict.get("critique", "")).strip()
        score = float(verdict.get("score", 0))
    except Exception:
        approved, critique, score = False, raw, 0.0

    append_trace(
        state,
        "verifier",
        "Verdict produced",
        f"Score {score:.2f} | Approved: {approved}",
        "success" if approved else "warning",
    )

    return {
        **state,
        "approved": approved,
        "critique": critique,
        "score": score,
        "trace": state["trace"],
    }


def route_after_verify(state: AgentState) -> str:
    return "end" if state.get("approved") else "executor"


def build_app(llm: GroqChatLLM, search: DuckDuckGoSearchRun | None):
    return LocalWorkflowApp(llm, search)


def run_workflow(goal: str, max_iterations: int = 3, use_search: bool = True) -> AgentState:
    llm = build_llm()
    search = build_search() if use_search else None
    app = build_app(llm, search)

    initial_state: AgentState = {
        "goal": goal,
        "tasks": [],
        "results": [],
        "critique": "",
        "approved": False,
        "iterations": 0,
        "max_iterations": max_iterations,
        "score": 0.0,
        "trace": [],
    }

    final_state = app.invoke(initial_state)
    return final_state


def format_task_cards(tasks: List[str]) -> str:
    cards = []
    for index, task in enumerate(tasks, start=1):
        cards.append(
            f"""
            <div class="task-card">
              <div class="task-index">Step {index}</div>
              <div class="task-body">{html.escape(task)}</div>
            </div>
            """
        )
    return "".join(cards)


def format_currency(value: float) -> str:
    return f"₹{value:,.0f} cr"


def build_budget_memory(goal: str, total_budget: float, final_state: AgentState | None) -> Dict[str, Any]:
    domain_baselines = {
        "commute": 0.18,
        "defence": 0.34,
        "agriculture": 0.22,
        "government": 0.26,
    }
    domain_labels = {
        "commute": "Commute",
        "defence": "Defence",
        "agriculture": "Agriculture",
        "government": "Government",
    }
    keywords = goal.lower()
    adjustments = {
        "commute": 0.03 if "commute" in keywords or "transport" in keywords else 0.0,
        "defence": 0.03 if "defence" in keywords or "security" in keywords else 0.0,
        "agriculture": 0.03 if "agriculture" in keywords or "farm" in keywords else 0.0,
        "government": 0.02 if "government" in keywords or "public" in keywords else 0.0,
    }

    short_term_shares = {}
    remaining = 1.0
    for key, base in domain_baselines.items():
        short_term_shares[key] = max(0.05, base + adjustments[key])
        remaining -= short_term_shares[key]

    if remaining != 0:
        short_term_shares["government"] = max(0.05, short_term_shares["government"] + remaining)

    total_share = sum(short_term_shares.values()) or 1.0
    short_term_shares = {key: value / total_share for key, value in short_term_shares.items()}

    allocations = {
        key: round(total_budget * share, 2) for key, share in short_term_shares.items()
    }
    baselines = {
        key: round(total_budget * share, 2) for key, share in domain_baselines.items()
    }

    short_term = {
        "active_goal": goal,
        "iteration": int(final_state.get("iterations", 0)) if final_state else 0,
        "latest_verdict": str(final_state.get("critique", "")) if final_state else "",
        "approved": bool(final_state.get("approved", False)) if final_state else False,
        "planner_steps": len(final_state.get("tasks", [])) if final_state else 0,
        "executor_passes": len(final_state.get("results", [])) if final_state else 0,
        "current_allocations": allocations,
        "allocation_shares": short_term_shares,
    }

    long_term = {
        "budget_envelope": total_budget,
        "baseline_allocations": baselines,
        "policy_rules": [
            "Preserve essential service continuity across all four domains.",
            "Keep commute spending aligned with mobility demand and maintenance cycles.",
            "Protect defence commitments that span multiple fiscal years.",
            "Retain agriculture support for seasonal risk, input volatility, and food security.",
            "Cover government overhead, service delivery, and recurring obligations.",
        ],
        "historical_signals": [
            "Commute demand tends to move with population density and fuel prices.",
            "Defence requires longer procurement lead times and more rigid commitments.",
            "Agriculture is sensitive to weather, input costs, and subsidy timing.",
            "Government spending usually has the largest recurring fixed-cost floor.",
        ],
        "domain_labels": domain_labels,
    }

    return {"short_term": short_term, "long_term": long_term}


def budget_card_html(name: str, allocation: float, baseline: float, share: float) -> str:
    delta = allocation - baseline
    delta_class = "status-ok" if delta >= 0 else "status-warn"
    delta_sign = "+" if delta >= 0 else ""
    return f"""
        <div class="metric-card budget-card">
          <div class="metric-label">{html.escape(name)}</div>
          <div class="metric-value">{html.escape(format_currency(allocation))}</div>
          <div class="metric-help">Baseline {html.escape(format_currency(baseline))}</div>
          <div class="budget-line">
            <span class="status-pill status-neutral">{share * 100:.1f}%</span>
            <span class="status-pill {delta_class}">{delta_sign}{delta:,.0f} cr vs baseline</span>
          </div>
        </div>
    """


def render_memory_panel(title: str, items: List[str], accent: str) -> str:
    bullets = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"""
        <div class="metric-card memory-card">
          <div class="metric-label">{html.escape(title)}</div>
          <div class="memory-accent" style="background:{accent};"></div>
          <ul class="memory-list">{bullets}</ul>
        </div>
    """


def render_streamlit_app() -> None:
    load_project_env()

    st.set_page_config(
        page_title="Planner -> Executor -> Validator",
        page_icon="PA",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fb;
            --panel: rgba(255,255,255,0.92);
            --panel-border: rgba(15, 23, 42, 0.08);
            --ink: #102033;
            --muted: #617089;
            --accent: #2563eb;
            --accent-2: #0ea5e9;
            --success: #059669;
            --warning: #d97706;
            --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(14,165,233,0.12), transparent 26%),
                linear-gradient(180deg, #f8fbff 0%, #eef3fb 100%);
            color: var(--ink);
        }

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 48%, #06b6d4 100%);
            color: white;
            padding: 28px 30px;
            border-radius: 24px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.05;
        }

        .hero p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1rem;
        }

        .subtle {
            color: var(--muted);
            font-size: 0.95rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            padding: 18px 18px 10px 18px;
            box-shadow: var(--shadow);
        }

        .metric-card {
            background: white;
            border: 1px solid rgba(37,99,235,0.12);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--ink);
        }

        .metric-help {
            margin-top: 4px;
            color: var(--muted);
            font-size: 0.86rem;
        }

        .task-card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 14px 16px;
            margin: 0 0 12px 0;
            box-shadow: 0 10px 20px rgba(15,23,42,0.04);
        }

        .task-index {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #2563eb;
            margin-bottom: 8px;
        }

        .task-body {
            color: var(--ink);
            line-height: 1.55;
        }

        .budget-card .metric-value {
            font-size: 1.35rem;
        }

        .budget-line {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .memory-card {
            padding-bottom: 14px;
        }

        .memory-accent {
            width: 100%;
            height: 4px;
            border-radius: 999px;
            margin: 6px 0 14px 0;
        }

        .memory-list {
            margin: 0;
            padding-left: 18px;
            color: var(--ink);
            line-height: 1.55;
        }

        .status-pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 8px;
        }

        .status-ok {
            background: rgba(5,150,105,0.12);
            color: var(--success);
        }

        .status-warn {
            background: rgba(217,119,6,0.12);
            color: var(--warning);
        }

        .status-neutral {
            background: rgba(37,99,235,0.12);
            color: var(--accent);
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #081120 0%, #0f172a 100%);
        }

        div[data-testid="stSidebar"] * {
            color: #e5eefb !important;
        }

        div[data-testid="stSidebar"] .stTextArea textarea,
        div[data-testid="stSidebar"] .stNumberInput input,
        div[data-testid="stSidebar"] .stSelectbox div,
        div[data-testid="stSidebar"] .stCheckbox label {
            color: #102033 !important;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <h1>Planner -> Executor -> Validator</h1>
          <p>A multi-agent workflow for turning a goal into a plan, executing each step, and validating the result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Run settings")
        st.caption("Shape the task, then let the agents work through it.")
        goal = st.text_area(
            "Goal",
            value="Research and summarize the top 3 trends in agriculture for 2025",
            height=140,
            help="Enter the outcome you want the agent team to produce.",
        )
        max_iterations = st.slider("Max iterations", 1, 5, 3)
        total_budget = st.number_input(
            "Annual budget envelope (₹ cr)",
            min_value=1000.0,
            max_value=500000.0,
            value=120000.0,
            step=500.0,
        )
        use_search = st.checkbox("Allow web search context", value=True)
        run_button = st.button("Run workflow", type="primary", use_container_width=True)

        st.markdown("### What happens")
        st.write("1. Planner breaks your goal into concrete tasks.")
        st.write("2. Executor completes each task with context.")
        st.write("3. Validator scores the result and decides whether to approve or loop again.")

    st.markdown(
        """
        <div class="panel">
          <div class="subtle">Use the sidebar to configure the run. The output below will show the plan, execution details, and the validator verdict in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "GROQ_API_KEY is not set. Add it to C:\\Training\\AI-ML-Training-Projects\\.env "
            "or your current environment and rerun the app."
        )
        st.stop()

    if "last_run" not in st.session_state:
        st.session_state.last_run = None

    if run_button:
        with st.spinner("Running the planner-executor-validator workflow..."):
            try:
                st.session_state.last_run = run_workflow(goal, max_iterations=max_iterations, use_search=use_search)
            except Exception as exc:
                st.error(f"Workflow failed: {exc}")
                st.stop()

    final_state = st.session_state.last_run
    if not final_state:
        st.info("Run the workflow to see the plan, execution outputs, and verification result.")
        return

    tasks = final_state.get("tasks", [])
    results = final_state.get("results", [])
    trace = final_state.get("trace", [])
    approved = bool(final_state.get("approved", False))
    score = float(final_state.get("score", 0.0))
    iterations = int(final_state.get("iterations", 0))
    critique = final_state.get("critique", "")
    budget_memory = build_budget_memory(goal, float(total_budget), final_state)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Tasks planned</div>
          <div class="metric-value">{len(tasks)}</div>
          <div class="metric-help">Concrete steps created by the planner</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Iterations</div>
          <div class="metric-value">{iterations}</div>
          <div class="metric-help">Execution / validation cycles completed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Validator score</div>
          <div class="metric-value">{score:.2f}</div>
          <div class="metric-help">Higher means stronger alignment with the goal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_class = "status-ok" if approved else "status-warn"
    status_label = "Approved" if approved else "Needs another pass"
    col4.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Status</div>
          <div class="metric-value"><span class="status-pill {status_class}">{status_label}</span></div>
          <div class="metric-help">Final validator decision</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Budget memory")
    mem_left, mem_right = st.columns(2, gap="large")
    with mem_left:
        st.markdown(
            render_memory_panel(
                "Short-term memory",
                [
                    f"Current goal: {budget_memory['short_term']['active_goal']}",
                    f"Current iteration: {budget_memory['short_term']['iteration']}",
                    f"Planner steps: {budget_memory['short_term']['planner_steps']}",
                    f"Executor passes: {budget_memory['short_term']['executor_passes']}",
                    f"Latest verdict: {budget_memory['short_term']['latest_verdict'] or 'No verdict yet'}",
                ],
                "linear-gradient(90deg, #2563eb, #0ea5e9)",
            ),
            unsafe_allow_html=True,
        )
    with mem_right:
        st.markdown(
            render_memory_panel(
                "Long-term memory",
                [
                    f"Budget envelope: {format_currency(budget_memory['long_term']['budget_envelope'])}",
                    "Preserve essentials across all domains.",
                    "Respect multi-year obligations and baseline commitments.",
                    "Use historical signals to avoid unstable annual swings.",
                    "Adjust only where policy or demand meaningfully changes.",
                ],
                "linear-gradient(90deg, #0f172a, #14b8a6)",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### Domain budget cards")
    budget_row_1 = st.columns(2, gap="large")
    budget_row_2 = st.columns(2, gap="large")
    domain_names = ["commute", "defence", "agriculture", "government"]
    domain_columns = [budget_row_1[0], budget_row_1[1], budget_row_2[0], budget_row_2[1]]
    for domain_name, column in zip(domain_names, domain_columns):
        with column:
            st.markdown(
                budget_card_html(
                    budget_memory["long_term"]["domain_labels"][domain_name],
                    budget_memory["short_term"]["current_allocations"][domain_name],
                    budget_memory["long_term"]["baseline_allocations"][domain_name],
                    budget_memory["short_term"]["allocation_shares"][domain_name],
                ),
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### Planner output")
        if tasks:
            st.markdown(format_task_cards(tasks), unsafe_allow_html=True)
        else:
            st.info("No tasks were produced.")

        st.markdown("### Validator feedback")
        if approved:
            st.success(critique or "The workflow was approved.")
        else:
            st.warning(critique or "The validator requested another pass.")

    with right:
        st.markdown("### Executor output")
        if results:
            tabs = st.tabs([f"Task {i}" for i in range(1, len(results) + 1)])
            for tab, task, result in zip(tabs, tasks, results):
                with tab:
                    st.caption(task)
                    st.write(result)
        else:
            st.info("No execution results yet.")

        st.markdown("### Run trace")
        for item in trace[-8:]:
            level = item.get("level", "info")
            pill_class = "status-ok" if level == "success" else "status-warn" if level == "warning" else "status-neutral"
            st.markdown(
                f"""
                <div class="task-card">
                  <div class="task-index"><span class="status-pill {pill_class}">{html.escape(str(item.get('stage', 'event')))}</span>{html.escape(str(item.get('title', '')))}</div>
                  <div class="task-body">{html.escape(str(item.get('detail', '')))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("View raw final state", expanded=False):
        st.json(final_state)


def main() -> None:
    goal = "Research and summarize the top 3 trends in agriculture for 2025"
    final_state = run_workflow(goal=goal, max_iterations=3, use_search=True)

    print("\n[Final State]")
    print(json.dumps(final_state, indent=2, ensure_ascii=False))


def is_streamlit_runtime() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if is_streamlit_runtime():
    render_streamlit_app()
elif __name__ == "__main__":
    main()

"""Notebook conversion for the CrewAI Groq supply-chain pipeline.


This script was generated from the Colab notebook and keeps the notebook

execution order, while replacing Colab-only secret loading with a local

repo-root `.env` first flow for this workspace.

"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_APPDATA = WORKSPACE_ROOT / ".codex_runtime" / "appdata"
WORKSPACE_APPDATA.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(WORKSPACE_APPDATA)
os.environ["APPDATA"] = str(WORKSPACE_APPDATA)
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

try:
    import appdirs

    appdirs._get_win_folder = lambda const: str(WORKSPACE_APPDATA)  # type: ignore[attr-defined]
except Exception:
    pass


def load_project_env() -> Path:
    """Load repo-root .env values into the current process if present."""

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        print(f"[WARN] Repo .env not found at {env_path}")
        return env_path

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)

    print(f"[OK] Loaded environment from {env_path}")
    return env_path


class LocalResult:
    def __init__(self, raw: str, token_usage=None):
        self.raw = raw
        self.token_usage = token_usage


ALLOW_LIVE_GROQ = os.getenv("ALLOW_LIVE_GROQ", "false").lower() == "true"


# # Colab 1 - CrewAI Supply-Chain Pipeline
# ### Day 16: Multi-Agent Coordination Patterns  |  Powered by Groq
#
# **Scenario:** GlobalFlow Logistics moves 4M parcels/day across 38 countries.
# A 2-hour port delay costs EUR 14M. Build a 5-agent crew that detects disruptions
# and coordinates a full response automatically.
#
# **What you will build:**
# - 5 specialised agents (Monitor, Router, Comms, Compliance, Reporter)
# - Task dependency chain with `context=` injection
# - Hierarchical process orchestrated by a Groq manager LLM
# - Long-term memory across crew runs
# - Executive disruption report saved to disk
#
# **LLM:** [Groq](https://console.groq.com) - free tier, fast inference
# **Models:** `llama-3.3-70b-versatile` (reasoning) | `llama-3.1-8b-instant` (simple tasks)
#
# > Get a free Groq API key at https://console.groq.com/keys - no credit card required.
#
# **Time budget:** ~85 min core + 30 min extension tasks

# ## Part 1 - Environment Setup (15 min)

# Cell 2 - Configure Groq API key
load_project_env()

key = os.environ.get("GROQ_API_KEY", "")
if key:
    print("[OK] GROQ_API_KEY available")
    print(f"  Key prefix: {key[:8]}...")
else:
    print("[NO] GROQ_API_KEY not set - cells below will fail until you set it.")

# Cell 4
# Cell 3 - Smoke test: single-agent hello world via Groq
from crewai import Agent, Task, Crew, Process

# In CrewAI v1, Groq models are referenced as 'groq/<model_name>'
# LiteLLM handles the Groq API call under the hood.
GROQ_FAST    = "groq/llama-3.1-8b-instant"       # fast + cheap, good for simple tasks
GROQ_SMART   = "groq/llama-3.3-70b-versatile"    # capable, good for reasoning
GROQ_MANAGER = "groq/llama-3.1-8b-instant"    # hierarchical manager LLM

test_agent = Agent(
    role="Hello World Agent",
    goal="Confirm the CrewAI + Groq environment is working correctly",
    backstory="A simple validation agent. You respond in exactly one sentence.",
    llm=GROQ_FAST,
    verbose=False,
    max_iter=2,
)

test_task = Task(
    description="Confirm you are running on Groq and identify your model in one sentence.",
    expected_output="One sentence confirming the environment works.",
    agent=test_agent,
)

test_crew = Crew(agents=[test_agent], tasks=[test_task], verbose=False)
if ALLOW_LIVE_GROQ:
    try:
        result = test_crew.kickoff()
        print("[OK] Smoke test PASSED - CrewAI + Groq is working")
    except Exception as exc:
        print(f"[WARN] Smoke test failed, using local fallback: {exc}")
        result = LocalResult(
            "Smoke test skipped because live CrewAI/Groq execution is blocked in this environment."
        )
else:
    print("[INFO] Live Groq smoke test disabled in this VM; using local fallback.")
    result = LocalResult(
        "Smoke test skipped because live CrewAI/Groq execution is disabled in this VM."
    )

print()
print(result.raw)

# ## Part 2 - Define 5 GlobalFlow Agents (40 min)

# Cell 6
# Cell 4 - Tools and LLM tier constants
from crewai import Agent
from crewai_tools import FileWriterTool
from crewai.tools import BaseTool
from pydantic import Field

# LLM tiers
GROQ_FAST    = "groq/llama-3.1-8b-instant"
GROQ_SMART   = "groq/llama-3.3-70b-versatile"
GROQ_MANAGER = "groq/llama-3.3-70b-versatile"

file_writer = FileWriterTool()

# Mock search tool - no Serper API key required for this lab
# In production replace with: from crewai_tools import SerperDevTool
class MockSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for supply-chain disruption news, "
        "shipping route data, and regulatory information."
    )

    def _run(self, query: str) -> str:
        return (
            f"[SIMULATED SEARCH] Results for: '{query}'\n"
            "Rotterdam: 18h closure, storm surge, severity 8/10\n"
            "Alternative 1 - Hamburg: +5h, -6% cost, low risk\n"
            "Alternative 2 - Felixstowe: +8h, -10% cost, medium risk\n"
            "Alternative 3 - Antwerp: +3h, +2% cost, low risk\n"
            "Singapore PSA: normal operations, no disruption\n"
        )

search_tool = MockSearchTool()

print("[OK] LLM tiers and tools configured")
print(f"  Fast model:    {GROQ_FAST}")
print(f"  Smart model:   {GROQ_SMART}")
print(f"  Manager model: {GROQ_MANAGER}")

# Cell 7
# Cell 5 - Agent 1: Disruption Monitor
disruption_monitor = Agent(
    role="Supply Chain Disruption Monitor",
    goal=(
        "Continuously scan for logistics disruptions - port closures, weather events, "
        "customs delays, and supplier failures - and assess their severity on a 1-10 scale."
    ),
    backstory=(
        "You are a veteran logistics intelligence analyst with 12 years at Maersk and DHL. "
        "You have seen every kind of supply-chain disruption imaginable, from Suez Canal "
        "blockages to pandemic port shutdowns. You are calm under pressure, deeply data-driven, "
        "and always quantify impact before escalating. You write in crisp bullet points."
    ),
    llm=GROQ_FAST,
    verbose=True,
    max_iter=4,
)

print("[OK] Agent 1:", disruption_monitor.role)

# Cell 8
# Cell 6 - Agent 2: Route Optimiser
route_optimiser = Agent(
    role="Logistics Route Optimiser",
    goal=(
        "Given a disruption report, calculate the 3 best alternative routes for affected "
        "shipments, ranking by total cost + estimated delay. Provide a clear recommendation."
    ),
    backstory=(
        "You are a PhD-level operations research specialist who spent 8 years building "
        "real-time routing algorithms for FedEx. You think in graphs, costs, and probabilities. "
        "You know every major shipping lane, air corridor, and rail route. You always present "
        "a primary recommendation plus two ranked alternatives with a weighted score."
    ),
    llm=GROQ_FAST,
    verbose=True,
    max_iter=4,
)

print("[OK] Agent 2:", route_optimiser.role)

# Cell 9
# Cell 7 - Agent 3: Supplier Communications Specialist
supplier_comms = Agent(
    role="Supplier Communications Specialist",
    goal=(
        "Draft professional, urgent communications to affected suppliers and carriers "
        "explaining the disruption, proposing alternatives, and requesting confirmation "
        "within 4 hours."
    ),
    backstory=(
        "You are a senior procurement manager who has negotiated contracts in 22 countries. "
        "You are culturally fluent, direct but diplomatic, and always frame disruptions as "
        "collaborative problems to solve, never as blame assignments. You know that tone "
        "in a crisis email can make or break a supplier relationship worth millions."
    ),
    llm=GROQ_FAST,      # Drafting emails does not need the 70B model
    verbose=True,
    max_iter=3,
)

print("[OK] Agent 3:", supplier_comms.role)

# Cell 10
# Cell 8 - Agent 4: Compliance Officer
compliance_officer = Agent(
    role="Trade Compliance Officer",
    goal=(
        "For each proposed re-route, verify customs requirements, check for sanctions or "
        "restricted-goods regulations, and flag any compliance risks. Issue a COMPLIANCE "
        "CLEARED or COMPLIANCE HOLD recommendation."
    ),
    backstory=(
        "You are a Certified Customs Specialist (CCS) with deep expertise in EU, US, and "
        "APAC trade regulations. You have worked with the WTO and have a zero-tolerance "
        "approach to compliance shortcuts. A single customs violation can cost more than "
        "the disruption itself."
    ),
    llm=GROQ_FAST,
    verbose=True,
    max_iter=4,
)

print("[OK] Agent 4:", compliance_officer.role)

# Cell 11
# Cell 9 - Agent 5: Executive Report Writer
report_writer = Agent(
    role="Executive Communications Writer",
    goal=(
        "Synthesise the disruption intelligence, route options, supplier actions, and "
        "compliance status into a clear, actionable executive briefing. "
        "Format: Situation -> Impact -> Response -> Next Steps. Maximum 1 page."
    ),
    backstory=(
        "You are a former management consultant who spent 10 years writing board-level "
        "crisis communications for Fortune 500 logistics companies. You eliminate jargon "
        "ruthlessly, lead with the bottom line, and always end with exactly 3 numbered "
        "action items with named owners and deadlines."
    ),
    llm=GROQ_SMART,
    verbose=True,
    max_iter=3,
)

print("[OK] Agent 5:", report_writer.role)
print()
print("All 5 GlobalFlow agents ready.")

# ### 2b - Define Tasks with Context Dependencies

# Cell 13
# Cell 10 - Task 1: Monitor disruptions
from crewai import Task

task_monitor = Task(
    description=(
        "Analyze the supplied Rotterdam disruption alert and identify the operational "
        "impact on GlobalFlow's key corridors: Rotterdam (EU hub), Singapore (APAC hub), "
        "Houston (US hub), and the AE-1 Asia-Europe shipping lane. Report: (1) disruption type and location, "
        "(2) severity score 1-10, (3) estimated duration, (4) shipments likely affected. "
        "Start your report with 'SEVERITY: X/10' on the first line."
    ),
    expected_output=(
        "Structured disruption report: severity score, affected corridors, "
        "shipment count, estimated duration, recommended escalation level."
    ),
    agent=disruption_monitor,
)

print("[OK] Task 1: Monitor disruptions  (no dependencies)")

# Cell 14
# Cell 11 - Task 2: Route optimisation (depends on Task 1)
task_route = Task(
    description=(
        "Using the disruption report in your context, calculate 3 alternative routes "
        "for the 50 highest-priority shipments. For each route: "
        "(1) route name and via-points, (2) cost delta vs standard (%), "
        "(3) delay in hours, (4) risk score 1-5, (5) CO2 delta. "
        "Rank by weighted score: 60% cost, 30% time, 10% risk."
    ),
    expected_output=(
        "Ranked table of 3 alternative routes with cost delta, delay, risk, "
        "weighted score, and a one-sentence rationale for the top choice."
    ),
    agent=route_optimiser,
    context=[task_monitor],   # <- injects task_monitor output into this task
)

print("[OK] Task 2: Route optimisation   (context: task_monitor)")

# Cell 15
# Cell 12 - Task 3: Supplier comms (depends on Tasks 1 + 2)
task_comms = Task(
    description=(
        "Draft communications to the 3 most critical affected suppliers. "
        "For each: (1) subject line, (2) 150-word email body explaining the disruption, "
        "the proposed re-routing option, and requesting confirmation within 4 hours. "
        "Tone: professional, urgent, collaborative."
    ),
    expected_output=(
        "Three complete email drafts formatted as:\n"
        "[SUPPLIER NAME] / [SUBJECT LINE]\n[EMAIL BODY]"
    ),
    agent=supplier_comms,
    context=[task_monitor, task_route],
)

print("[OK] Task 3: Supplier comms       (context: task_monitor, task_route)")

# Cell 16
# Cell 13 - Task 4: Compliance check (depends on Task 2)
task_compliance = Task(
    description=(
        "Review the top-ranked re-routing option from the route optimisation team. "
        "Check: (1) customs requirements per transit country, "
        "(2) sanctions or dual-use goods restrictions, "
        "(3) certificate of origin implications. "
        "Issue COMPLIANCE CLEARED or COMPLIANCE HOLD with detailed reasoning."
    ),
    expected_output=(
        "Compliance status (CLEARED or HOLD), per-country requirements, "
        "flags with remediation steps, estimated customs processing time."
    ),
    agent=compliance_officer,
    context=[task_route],
)

print("[OK] Task 4: Compliance check     (context: task_route)")

# Cell 17
# Cell 14 - Task 5: Executive report (all context)
task_report = Task(
    description=(
        "Compile all outputs into a single executive briefing.\n"
        "Use these exact headings:\n"
        "  SITUATION: what happened and severity\n"
        "  IMPACT: shipments affected, EUR cost exposure\n"
        "  RESPONSE: chosen re-route, supplier actions, compliance status\n"
        "  NEXT STEPS: exactly 3 numbered actions with owners and deadlines\n"
        "Maximum 400 words. Save to file 'globalflow_disruption_report.txt'."
    ),
    expected_output=(
        "Complete executive briefing saved to 'globalflow_disruption_report.txt', "
        "4-section structure, maximum 400 words."
    ),
    agent=report_writer,
    context=[task_monitor, task_route, task_comms, task_compliance],
    output_file="globalflow_disruption_report.txt",
)

print("[OK] Task 5: Executive report     (context: ALL tasks)")
print()
print("Task dependency chain:")
print("  task_monitor")
print("    +-> task_route ---------> task_compliance")
print("    +-> task_route ---+")
print("    +------------------+-> task_comms")
print("  ALL -----------------> task_report -> file output")

# ## Part 3 - Assemble the Crew and Run (30 min)

# Cell 19
# Cell 15 - Assemble the Crew
from crewai import Crew, Process

globalflow_crew = Crew(
    agents=[
        disruption_monitor,
        route_optimiser,
        supplier_comms,
        compliance_officer,
        report_writer,
    ],
    tasks=[
        task_monitor,
        task_route,
        task_comms,
        task_compliance,
        task_report,
    ],
    process=Process.sequential,
    # manager_llm=GROQ_MANAGER,   # Groq 70B orchestrates delegation
    verbose=True,
    memory=False,               # CrewAI memory search can trigger Groq tool-use failures here
    output_log_file="crew_run.log",
)

print("[OK] GlobalFlow Crew assembled")
print(f"  Agents:      {len(globalflow_crew.agents)}")
print(f"  Tasks:       {len(globalflow_crew.tasks)}")
print(f"  Process:     {globalflow_crew.process.value}")
print(f"  Manager LLM: {GROQ_MANAGER}")
print(f"  Memory:      {globalflow_crew.memory}")

# Cell 20
# Cell 16 - Trigger: simulate a Rotterdam port closure
trigger_input = {
    "disruption_alert": (
        "ALERT: Port of Rotterdam (GlobalFlow EU hub) has declared force majeure "
        "due to severe North Sea storm surge. Expected closure: 18-24 hours. "
        "340 containers from GlobalFlow clients are currently docked. "
        "12 Maersk vessels en-route have been diverted to Felixstowe. "
        "Incident started: 2025-06-18 06:30 UTC. "
        "Client SLA breach window opens in 6 hours."
    )
}

print("[ALERT] TRIGGERING GlobalFlow Disruption Response Crew")
print("=" * 60)
print(trigger_input["disruption_alert"])
print("=" * 60)
print()
print("Starting crew kickoff - expect 2-5 minutes on Groq free tier.")
print("Watch each agent Thought -> Action -> Observation loop below.")
print()

def build_fallback_result() -> tuple[str, str]:
    report = (
        "SITUATION: Port of Rotterdam is closed due to severe storm surge.\n"
        "IMPACT: 340 containers affected; SLA breach window opens within 6 hours.\n"
        "RESPONSE: Use Felixstowe and Hamburg as backup routes, notify suppliers, and keep compliance checks active.\n"
        "NEXT STEPS:\n"
        "1. Ops to confirm reroutes within 1 hour.\n"
        "2. Comms to send supplier notices within 2 hours.\n"
        "3. Compliance to approve transit path before shipment release.\n"
    )
    return report, "Fallback executed after CrewAI kickoff failed."

import time

MAX_RETRIES = 3
RETRY_DELAY = 20  # seconds — slightly above Groq's suggested 14.4s

if ALLOW_LIVE_GROQ:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = globalflow_crew.kickoff(inputs=trigger_input)
            kickoff_status = "[OK] Crew kickoff completed successfully"
            break
        except Exception as exc:
            error_str = str(exc)
            if "rate_limit_exceeded" in error_str and attempt < MAX_RETRIES:
                print(f"[WARN] Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                      f"Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
            else:
                kickoff_status = f"[WARN] Crew kickoff failed: {exc}"
                print(kickoff_status)
                fallback_raw, fallback_log = build_fallback_result()
                result = LocalResult(fallback_raw)
                with open("globalflow_disruption_report.txt", "w") as f:
                    f.write(fallback_raw + "\n" + fallback_log)
                break

# if ALLOW_LIVE_GROQ:
#     try:
#         result = globalflow_crew.kickoff(inputs=trigger_input)
#         kickoff_status = "[OK] Crew kickoff completed successfully"
#     except Exception as exc:
#         kickoff_status = f"[WARN] Crew kickoff failed, using fallback: {exc}"
#         print(kickoff_status)
#         fallback_raw, fallback_log = build_fallback_result()

#         result = LocalResult(fallback_raw)
#         with open("globalflow_disruption_report.txt", "w", encoding="utf-8") as f:
#             f.write(fallback_raw)
#             f.write("\n")
#             f.write(fallback_log)
else:
    kickoff_status = "[INFO] Live crew kickoff disabled in this VM; using local simulation."
    print(kickoff_status)
    fallback_raw, fallback_log = build_fallback_result()
    result = LocalResult(fallback_raw)
    with open("globalflow_disruption_report.txt", "w", encoding="utf-8") as f:
        f.write(fallback_raw)
        f.write("\n")
        f.write(fallback_log)

print(kickoff_status)

# Cell 21
# Cell 17 - Review the final output
print("\n" + "=" * 60)
print("FINAL EXECUTIVE BRIEFING")
print("=" * 60)
print(result.raw)
print()
print("-" * 60)
print(f"Token usage: {result.token_usage}")

# Cell 22
# Cell 18 - Inspect the saved report file
import os

report_file = "globalflow_disruption_report.txt"
if os.path.exists(report_file):
    size = os.path.getsize(report_file)
    print(f"[OK] Report saved: '{report_file}'  ({size} bytes)")
    print()
    with open(report_file, "r") as f:
        print(f.read())
else:
    print("[WARN] Report file not found - check verbose output above.")
    print("Falling back to result.raw:")
    print(result.raw)

# Cell 23
# Cell 19 - Inspect crew memory (long-term)
import glob

memory_files = glob.glob("*.db") + glob.glob(".crewai/**/*.db", recursive=True)
if memory_files:
    print("Memory database files found:")
    for f in memory_files:
        size = os.path.getsize(f)
        print(f"  {f}  ({size:,} bytes)")
    print()
    print("[TIP] Re-run Cell 16 - agents will reference previous disruption context.")
else:
    print("No memory DB found yet. Run the crew kickoff first (Cell 16).")
    print("Memory DB appears after first successful run.")

# Cell 24
# Cell 20 - Token usage and rough cost estimate
print("Token Usage Summary")
print("=" * 40)
if hasattr(result, 'token_usage') and result.token_usage:
    tu = result.token_usage
    print(f"  Prompt tokens:     {tu.prompt_tokens:>8,}")
    print(f"  Completion tokens: {tu.completion_tokens:>8,}")
    print(f"  Total tokens:      {tu.total_tokens:>8,}")
    # Groq llama-3.3-70b pricing (approx): $0.59/1M input, $0.79/1M output
    input_cost  = (tu.prompt_tokens     / 1_000_000) * 0.59
    output_cost = (tu.completion_tokens / 1_000_000) * 0.79
    print(f"  Est. cost (70B):   ${input_cost + output_cost:.5f}")
else:
    print("  Token usage not available for this run.")

# ## Extension Tasks
#
# Work through these after the core lab. Estimated time: 30-60 min.
#
# ---
#
# ### Extension 1 - Add a Financial Analyst Agent
#
# Create a 6th agent that calculates total EUR exposure from the disruption:
#
# ```python
# financial_analyst = Agent(
#     role="Supply Chain Financial Analyst",
#     goal=(
#         "Calculate total EUR exposure: rerouting cost delta, "
#         "SLA penalty clauses triggered, insurance deductible, "
#         "and opportunity cost of delayed deliveries."
#     ),
#     backstory=(
#         "CFA-qualified financial analyst specialising in logistics cost modelling. "
#         "Always presents base case, worst case, and best case scenarios."
#     ),
#     llm=GROQ_SMART,
#     verbose=True,
#     max_iter=3,
# )
#
# task_financial = Task(
#     description="Calculate total EUR exposure: rerouting, SLA penalties, insurance.",
#     expected_output="Financial exposure table: base / worst / best case in EUR.",
#     agent=financial_analyst,
#     context=[task_monitor, task_route],
# )
# ```
#
# Add `financial_analyst` to `agents=` and `task_financial` to `tasks=` in the Crew, then re-run.
#
# ---
#
# ### Extension 2 - Parallel Async Execution
#
# `task_comms` and `task_compliance` are independent and can run in parallel:
#
# ```python
# task_comms = Task(..., async_execution=True)
# task_compliance = Task(..., async_execution=True)
# ```
#
# Use `crew.kickoff_async()` and compare wall-clock time vs sequential.
#
# ---
#
# ### Extension 3 - Human-in-the-Loop Gate
#
# Add a human approval step before the report is written:
#
# ```python
# task_report = Task(..., human_input=True)
# ```
#
# Re-run - a prompt will appear asking you to approve before the report is written.
#
# ---
#
# ### Extension 4 - Switch Groq Model Tiers
#
# Try assigning different models per agent based on task complexity:
#
# ```python
# GROQ_FAST  = "groq/llama-3.1-8b-instant"    # supplier_comms (drafting)
# GROQ_SMART = "groq/llama-3.3-70b-versatile" # monitor, router, compliance
# ```
#
# Compare output quality vs token cost across tiers.

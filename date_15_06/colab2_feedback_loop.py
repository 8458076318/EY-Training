# -*- coding: utf-8 -*-
"""Notebook conversion of `colab2_feedback_loop.ipynb`.

This script keeps the lab flow from the notebook but runs as a normal Python
file and reads API keys from the project-root `.env`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401
from tqdm.auto import tqdm

from openai import OpenAI

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None


ROOT = Path(__file__).resolve().parents[1]


def load_project_env(name: str, default: str = "") -> str:
    """Read a value from the repo-root `.env` without requiring dotenv."""
    env_path = ROOT / ".env"
    value = os.getenv(name, default)
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() == name:
                value = raw_value.strip().strip('"').strip("'")
    return value.strip().strip('"').strip("'")


OPENAI_API_KEY = load_project_env("OPENAI_API_KEY")
ANTHROPIC_API_KEY = load_project_env("ANTHROPIC_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = (
    anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    if anthropic is not None and ANTHROPIC_API_KEY
    else None
)

if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY and ANTHROPIC_API_KEY in project .env")

print("Setup complete")


# ── Structured log entry dataclass ───────────────────────────────────
@dataclass
class LLMLogEntry:
    request_id: str
    timestamp: str
    prompt_version: str
    prompt_hash: str
    model: str
    task_type: str
    borrower_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    output_text: str
    output_word_count: int
    auto_score: Optional[float] = None
    hallucination_flag: Optional[bool] = None
    hallucination_count: Optional[int] = None
    missing_sections: Optional[str] = None
    user_correction: Optional[bool] = None
    failure_category: Optional[str] = None
    environment: str = "production"


def init_db(db_path: str = "finsight_logs.db") -> sqlite3.Connection:
    """Create SQLite database with logging schema."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_logs (
            request_id          TEXT PRIMARY KEY,
            timestamp           TEXT,
            prompt_version      TEXT,
            prompt_hash         TEXT,
            model               TEXT,
            task_type           TEXT,
            borrower_id         TEXT,
            input_tokens        INTEGER,
            output_tokens       INTEGER,
            latency_ms          REAL,
            cost_usd            REAL,
            output_text         TEXT,
            output_word_count   INTEGER,
            auto_score          REAL,
            hallucination_flag  INTEGER,
            hallucination_count INTEGER,
            missing_sections    TEXT,
            user_correction     INTEGER,
            failure_category    TEXT,
            environment         TEXT
        )
        """
    )
    conn.commit()
    return conn


def log_entry(conn: sqlite3.Connection, entry: LLMLogEntry) -> None:
    """Insert a log entry into the database."""
    d = asdict(entry)
    d["hallucination_flag"] = (
        int(d["hallucination_flag"]) if d["hallucination_flag"] is not None else None
    )
    d["user_correction"] = (
        int(d["user_correction"]) if d["user_correction"] is not None else None
    )
    conn.execute(
        f"INSERT OR REPLACE INTO llm_logs VALUES ({','.join(['?'] * len(d))})",
        list(d.values()),
    )
    conn.commit()


conn = init_db()
print("SQLite database initialised: finsight_logs.db")
print("Schema:", [col[0] for col in conn.execute("PRAGMA table_info(llm_logs)").fetchall()])


# ── Prompt variants to test ────────────────────────────────────────────
PROMPT_V1_0 = {
    "version": "v1.0",
    "system": "You are a credit analyst. Generate a credit risk memo based on the borrower data.",
    "user_template": "Borrower data:\n{data}\n\nWrite a credit memo.",
}

PROMPT_V1_1 = {
    "version": "v1.1",
    "system": """You are a credit analyst AI at FinSight AI.
Generate a credit risk memo (150-250 words) structured as:
1. BORROWER OVERVIEW
2. KEY FINANCIAL METRICS
3. RISK ASSESSMENT
4. RECOMMENDATION
Use only the data provided. Do not extrapolate or add information not given.""",
    "user_template": "Generate a credit memo for:\n{data}",
}

PROMPT_V2_0 = {
    "version": "v2.0",
    "system": """You are a senior credit analyst at FinSight AI, operating in a regulated lending environment.

PROCESS:
Step 1: List every numeric fact explicitly stated in the borrower data.
Step 2: Identify any internal inconsistencies or missing data.
Step 3: Write the credit memo using ONLY those stated facts.

MEMO FORMAT (150-250 words):
BORROWER OVERVIEW: [company name, industry, loan request]
KEY FINANCIAL METRICS: [cite exact figures from data; flag any gaps]
RISK ASSESSMENT: [strengths and risk factors based only on provided data]
RECOMMENDATION: [approve/decline/conditional with rationale]

COMPLIANCE RULES:
- Never state a number not explicitly given in the input
- If data is inconsistent, note it explicitly: "Note: figures appear inconsistent - [detail]"
- Use hedged language for projections: "based on provided data" not "will" or "certainly"
- If insufficient data exists for a section, write: "Insufficient data provided for [section]""",
    "user_template": "Generate a credit memo for the following borrower.\n\nBORROWER DATA:\n{data}",
}

ALL_PROMPTS = [PROMPT_V1_0, PROMPT_V1_1, PROMPT_V2_0]


def hash_prompt(system: str, template: str) -> str:
    return hashlib.md5((system + template).encode()).hexdigest()[:8]


for p in ALL_PROMPTS:
    h = hash_prompt(p["system"], p["user_template"])
    print(f"{p['version']} -> hash: {h}")
print("Prompt variants defined")


from datetime import datetime, timedelta  # noqa: E402


BORROWER_PROFILES = [
    "Northgate Logistics. Revenue $8.2M. EBITDA $1.1M. Debt $2.5M. DSCR 1.7x. Loan request $1.5M.",
    "Sunrise Bakeries. Revenue $3.4M. Net profit $280K. No existing debt. Loan request $400K equipment.",
    "Harbor Bridge Tech. ARR $2.1M. Growth 18% YoY. Burn $85K/mo. Runway 18mo. Loan request $300K.",
    "Westfield Industrial. Revenue $19.5M. EBITDA $3.2M. Existing debt $6.8M. Collateral $9.1M. Loan $2M.",
    "Blue Ridge Farms. Revenue $5.9M. Seasonal. DSCR 1.4x. Crop insurance. Loan request $900K operating line.",
    "Delta Medical Supplies. Revenue $11.2M. Gross margin 34%. DSCR 2.1x. 8yr history. Loan $1.8M.",
    "Summit Construction. Revenue $28M. Gross margin 9%. Active contracts $12M. DSCR 1.5x. Loan $3.5M.",
    "Coastal Realty Trust. Properties 8. NAV $14.2M. LTV 58%. Cash coverage 1.8x. Loan $1.5M.",
    "Ironwood Software. ARR $4.8M. NRR 118%. No debt. Founder personal guarantee. Loan $600K.",
    "Pacific Fisheries. Revenue $7.1M. EBITDA $900K. Fleet value $3.8M. DSCR 1.3x. Loan $1.1M.",
    "NovaStar Retail. Revenue $15M (down 12% YoY). EBITDA $800K. High lease obligations $3.2M/yr. Loan $2M.",
    "Vertex Energy. Revenue $9.2M. EBITDA $1.4M. BUT Q3 one-time gain $600K not excluded. True EBITDA $800K. Loan $1.5M.",
    "Cascade Healthcare. Revenue $6.8M. Receivables $2.1M (90-day overdue 35%). DSCR 1.2x. Loan $800K.",
    "Alpine Hotels. Revenue $12M. Seasonality: 60% summer. Q4 DSCR 0.7x. Annual DSCR 1.4x. Loan $2.2M.",
    "TechVenture Alpha. Pre-revenue. VC backed $3M. Burn $180K/mo. Patent pending. Loan $400K bridge.",
    "Heritage Textiles. Revenue $4.1M. Stable. Owner age 68 no succession plan. DSCR 1.9x. Loan $600K.",
    "Meridian Imports. Revenue $22M. Single customer 65% concentration. DSCR 2.3x. Loan $3M.",
    "Quantum Logistics. Revenue $17M. Growing. Tight working capital. Current ratio 0.9. Loan $2.5M.",
    "Bluestone Mining. Revenue $8.7M. Commodity price sensitive. Insurance in place. DSCR 1.6x. Loan $1.4M.",
    "RapidGrow Foods. Revenue $6.2M. EBITDA margin stated 28% - typical for sector is 12-15%. Loan $1M.",
    "Unnamed LLC. Some revenue. Requesting $500K. Business started recently.",
    "Apex Corp. Revenue not disclosed. Assets $2M. Loan $800K.",
    "XYZ Holdings. Revenue $5M. No other financial data provided. Need $750K urgently.",
    "Global Partners. Profitable. Good DSCR. Loan $1.2M.",
    "MegaBuild Inc. Revenue $100M. EBITDA $50M. (Implausibly high margin for construction at 50%). Loan $5M.",
    "BrightPath Energy. Revenue $12M. Net income $3.6M (30% margin). But tax paid $50K only. DSCR 1.8x. Loan $2M.",
    "Frontier Exports. Revenue $18M. EBITDA $2.7M. Debt service $2.1M. Stated DSCR 1.8x (actual: 1.3x). Loan $3M.",
    "ClearView Media. ARR $1.8M. MRR $200K. Note: $200K x 12 = $2.4M not $1.8M. Loan $400K.",
    "Skyline Builders. Properties appraised $20M. Debt $15M. Stated LTV 60% (actual 75%). Loan $2.5M.",
    "Neptune Pharma. Revenue $9.1M. Gross margin 72%. SG&A $6.8M. Implied operating margin 2% but memo claims 15%. Loan $1.5M.",
] * 2

BORROWER_PROFILES = BORROWER_PROFILES[:50]


def call_model_with_prompt(prompt_cfg: dict, borrower_data: str) -> dict:
    user_msg = prompt_cfg["user_template"].format(data=borrower_data)
    start = time.time()

    if openai_client is not None:
        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": prompt_cfg["system"]},
                    {"role": "user", "content": user_msg},
                ],
            )
            latency = (time.time() - start) * 1000
            output = resp.choices[0].message.content or ""
            prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
            return {
                "output": output,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "latency_ms": round(latency, 1),
                "cost": 0.0,
                "error": None,
            }
        except Exception as exc:
            return {
                "output": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": (time.time() - start) * 1000,
                "cost": 0,
                "error": str(exc),
            }

    if anthropic_client is not None:
        try:
            resp = anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=500,
                system=prompt_cfg["system"],
                messages=[{"role": "user", "content": user_msg}],
            )
            latency = (time.time() - start) * 1000
            output = resp.content[0].text
            return {
                "output": output,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "latency_ms": round(latency, 1),
                "cost": (resp.usage.input_tokens * 3 + resp.usage.output_tokens * 15)
                / 1_000_000,
                "error": None,
            }
        except Exception as exc:
            return {
                "output": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": (time.time() - start) * 1000,
                "cost": 0,
                "error": str(exc),
            }

    raise RuntimeError("No usable API client configured")


# ── Run simulation: v1.0 prompt on 50 requests ────────────────────────
print("Simulating 50 production requests with prompt v1.0...")
print("(Using a subset of 10 for speed - change SIMULATE_N to 50 for full run)")

SIMULATE_N = 10
PROMPT_CFG = PROMPT_V1_0
sim_start = datetime(2025, 6, 1, 9, 0, 0)

for i, borrower_data in enumerate(tqdm(BORROWER_PROFILES[:SIMULATE_N], desc="Simulating")):
    result = call_model_with_prompt(PROMPT_CFG, borrower_data)

    entry = LLMLogEntry(
        request_id=str(uuid.uuid4()),
        timestamp=(sim_start + timedelta(hours=i * 0.5)).isoformat(),
        prompt_version=PROMPT_CFG["version"],
        prompt_hash=hash_prompt(PROMPT_CFG["system"], PROMPT_CFG["user_template"]),
        model=ANTHROPIC_MODEL if anthropic_client is not None else OPENAI_MODEL,
        task_type="credit_memo",
        borrower_id=f"BRW-{i + 1:04d}",
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        latency_ms=result["latency_ms"],
        cost_usd=result["cost"],
        output_text=result["output"],
        output_word_count=len(result["output"].split()),
        environment="simulation",
    )
    log_entry(conn, entry)
    time.sleep(0.2)

count = conn.execute("SELECT COUNT(*) FROM llm_logs").fetchone()[0]
print(f"\n{count} log entries written to SQLite")


# ── Quality probe suite ───────────────────────────────────────────────
REQUIRED_SECTIONS = [
    "borrower",
    "financial",
    "risk",
    "recommend",
]


def probe_hallucination(borrower_data: str, output: str) -> tuple[bool, int]:
    """Check for numeric hallucination (values in output not in source)."""

    def extract_nums(text: str) -> set[float]:
        raw = re.findall(r"\$?[\d,]+\.?\d*[KMB]?", text)
        nums: set[float] = set()
        for r in raw:
            r_clean = r.replace("$", "").replace(",", "")
            try:
                if r_clean.endswith("K"):
                    nums.add(float(r_clean[:-1]) * 1e3)
                elif r_clean.endswith("M"):
                    nums.add(float(r_clean[:-1]) * 1e6)
                elif r_clean.endswith("B"):
                    nums.add(float(r_clean[:-1]) * 1e9)
                else:
                    nums.add(float(r_clean))
            except Exception:
                pass
        return nums

    source_nums = extract_nums(borrower_data)
    output_nums = extract_nums(output)

    hallucinated = [
        v
        for v in output_nums
        if v > 1000
        and not any(abs(v - s) / max(s, 0.01) < 0.05 for s in source_nums if s > 0)
    ]
    return len(hallucinated) > 0, len(hallucinated)


def probe_missing_sections(output: str) -> list[str]:
    """Detect which required sections are absent."""
    output_lower = output.lower()
    return [s for s in REQUIRED_SECTIONS if s not in output_lower]


def probe_output_length(output: str) -> bool:
    """Flag outputs outside the 150-250 word target range."""
    wc = len(output.split())
    return wc < 100 or wc > 350


def simulate_user_correction(output: str, borrower_data: str) -> bool:
    """Simulate analyst feedback: 1 in 4 chance if output has issues."""
    has_issue = (
        probe_missing_sections(output)
        or probe_hallucination(borrower_data, output)[0]
        or probe_output_length(output)
    )
    return bool(has_issue and random.random() < 0.6)


df_logs = pd.read_sql("SELECT * FROM llm_logs", conn)

for _, row in df_logs.iterrows():
    if not row["output_text"]:
        continue

    borrower_data = BORROWER_PROFILES[int(row["borrower_id"].split("-")[1]) - 1]
    hall_flag, hall_count = probe_hallucination(borrower_data, row["output_text"])
    missing = probe_missing_sections(row["output_text"])
    correction = simulate_user_correction(row["output_text"], borrower_data)

    conn.execute(
        """
        UPDATE llm_logs
        SET hallucination_flag=?, hallucination_count=?, missing_sections=?, user_correction=?
        WHERE request_id=?
        """,
        (int(hall_flag), hall_count, json.dumps(missing), int(correction), row["request_id"]),
    )

conn.commit()
df_logs = pd.read_sql("SELECT * FROM llm_logs", conn)

print("\nQuality Probe Results (Prompt v1.0 Baseline)")
print("=" * 50)
print(f"  Total requests:          {len(df_logs)}")
print(f"  Hallucination rate:      {df_logs['hallucination_flag'].mean()*100:.1f}%")
print(f"  Missing sections rate:   {(df_logs['missing_sections'] != '[]').mean()*100:.1f}%")
print(f"  User correction rate:    {df_logs['user_correction'].mean()*100:.1f}%")
print(f"  Avg latency:             {df_logs['latency_ms'].mean():.0f}ms")
print(f"  Avg cost/memo:           ${df_logs['cost_usd'].mean():.5f}")


# ── Failure categorisation ────────────────────────────────────────────
def categorise_failure(row) -> Optional[str]:
    def _get_value(key: str):
        if isinstance(row, dict):
            return row.get(key)
        return getattr(row, key, None)

    missing_sections = (
        _get_value("missing_sections")
    )
    hallucination_flag = _get_value("hallucination_flag")
    output_word_count = _get_value("output_word_count")
    user_correction = _get_value("user_correction")

    if isinstance(missing_sections, str) and missing_sections.strip():
        missing = json.loads(missing_sections)
    else:
        missing = []
    if hallucination_flag:
        return "HALLUCINATION"
    if missing:
        return f"MISSING_SECTION:{missing[0].upper()}"
    if output_word_count and (
        output_word_count < 100 or output_word_count > 350
    ):
        return "LENGTH_VIOLATION"
    if user_correction:
        return "USER_CORRECTION_OTHER"
    return None


for _, row in df_logs.iterrows():
    cat = categorise_failure(row)
    conn.execute(
        "UPDATE llm_logs SET failure_category=? WHERE request_id=?",
        (cat, row["request_id"]),
    )
conn.commit()

df_logs = pd.read_sql("SELECT * FROM llm_logs", conn)

failures = df_logs[df_logs["failure_category"].notna()]
fail_counts = failures["failure_category"].value_counts()

print("\nFAILURE TRIAGE REPORT")
print("=" * 45)
print(
    f"  Failed outputs:   {len(failures)} / {len(df_logs)} "
    f"({len(failures) / len(df_logs) * 100:.1f}%)"
)
print("\n  Failure breakdown:")
for cat, count in fail_counts.items():
    print(f"    {cat:<35} {count:>3}  ({count / len(df_logs) * 100:.1f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("FinSight Prompt v1.0 - Failure Analysis", fontsize=13, fontweight="bold")

if len(fail_counts) > 0:
    colors = ["#F97316", "#8B5CF6", "#F59E0B", "#EF4444", "#10B981"]
    ax1.barh(fail_counts.index, fail_counts.values, color=colors[: len(fail_counts)])
    ax1.set_xlabel("Count")
    ax1.set_title("Failure Categories")
    ax1.invert_yaxis()

ax2.hist(df_logs["output_word_count"].dropna(), bins=15, color="#0D9488", edgecolor="white")
ax2.axvline(100, color="red", linestyle="--", label="Min 100w")
ax2.axvline(350, color="orange", linestyle="--", label="Max 350w")
ax2.set_xlabel("Word Count")
ax2.set_title("Output Length Distribution")
ax2.legend()

plt.tight_layout()
plt.savefig("failure_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Failure analysis chart saved")


# ── Inspect worst performers to inform prompt fix ─────────────────────
print("WORST PERFORMER EXAMPLES (Prompt v1.0)\n")
print("These are the outputs that drove the prompt revision.\n")

worst = df_logs[df_logs["failure_category"].notna()].head(3)
for _, row in worst.iterrows():
    borrower_data = BORROWER_PROFILES[int(row["borrower_id"].split("-")[1]) - 1]
    print(f"{'─' * 60}")
    print(f"Borrower: {row['borrower_id']} | Failure: {row['failure_category']}")
    print(f"Source data: {borrower_data[:100]}...")
    print(f"Output (first 300 chars):\n{row['output_text'][:300]}")
    print(
        f"Word count: {row['output_word_count']} | Hallucination: {bool(row['hallucination_flag'])}"
    )
    print()

print()
print("HYPOTHESIS FOR PROMPT REVISION")
print("=" * 60)
print(
    """
Based on the failure analysis:

PROBLEM 1: HALLUCINATION - Model is inferring/inventing figures
  Root cause: v1.0 prompt doesn't explicitly forbid extrapolation
  Fix: Add explicit anti-hallucination instruction + chain-of-thought

PROBLEM 2: MISSING SECTIONS - Model skips structure
  Root cause: v1.0 only says 'write a memo' - no format requirement
  Fix: Require explicit headers in system prompt

HYPOTHESIS: v2.0 (chain-of-thought + anti-hallucination) will reduce
  hallucination rate by >=50% and missing section rate by >=70%
"""
)


# ── Run v2.0 on the SAME borrower set ─────────────────────────────────
print("Running v2.0 prompt on same inputs...\n")

PROMPT_CFG_V2 = PROMPT_V2_0

for i, borrower_data in enumerate(tqdm(BORROWER_PROFILES[:SIMULATE_N], desc="v2.0")):
    result = call_model_with_prompt(PROMPT_CFG_V2, borrower_data)

    entry = LLMLogEntry(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        prompt_version=PROMPT_CFG_V2["version"],
        prompt_hash=hash_prompt(PROMPT_CFG_V2["system"], PROMPT_CFG_V2["user_template"]),
        model=ANTHROPIC_MODEL if anthropic_client is not None else OPENAI_MODEL,
        task_type="credit_memo",
        borrower_id=f"BRW-{i + 1:04d}",
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        latency_ms=result["latency_ms"],
        cost_usd=result["cost"],
        output_text=result["output"],
        output_word_count=len(result["output"].split()),
        environment="simulation",
    )

    hall_flag, hall_count = probe_hallucination(borrower_data, result["output"])
    missing = probe_missing_sections(result["output"])
    correction = simulate_user_correction(result["output"], borrower_data)
    cat = categorise_failure(
        type(
            "R",
            (),
            {
                "hallucination_flag": int(hall_flag),
                "missing_sections": json.dumps(missing),
                "output_word_count": len(result["output"].split()),
                "user_correction": int(correction),
            },
        )()
    )

    entry.hallucination_flag = hall_flag
    entry.hallucination_count = hall_count
    entry.missing_sections = json.dumps(missing)
    entry.user_correction = correction
    entry.failure_category = cat

    log_entry(conn, entry)
    time.sleep(0.2)

print("v2.0 run complete")


# ── Compare before/after metrics ──────────────────────────────────────
df_all = pd.read_sql("SELECT * FROM llm_logs", conn)

comparison = (
    df_all.groupby("prompt_version")
    .agg(
        n_requests=("request_id", "count"),
        hallucin_rate=("hallucination_flag", "mean"),
        missing_sec_rate=("missing_sections", lambda x: (x != "[]").mean()),
        user_correction=("user_correction", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        avg_cost=("cost_usd", "mean"),
        avg_word_count=("output_word_count", "mean"),
        failure_rate=("failure_category", lambda x: x.notna().mean()),
    )
    .round(4)
    .reset_index()
)

comparison = comparison[comparison["prompt_version"].isin(["v1.0", "v2.0"])]

print("\nBEFORE / AFTER PROMPT COMPARISON")
print("=" * 70)
print(
    comparison[
        [
            "prompt_version",
            "hallucin_rate",
            "missing_sec_rate",
            "user_correction",
            "failure_rate",
            "avg_cost",
        ]
    ].to_string(index=False)
)

if len(comparison) >= 2:
    metrics = ["hallucin_rate", "missing_sec_rate", "user_correction", "failure_rate"]
    labels = [
        "Hallucination\nRate",
        "Missing\nSections",
        "User\nCorrections",
        "Overall\nFailure Rate",
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    width = 0.35

    v1 = comparison[comparison["prompt_version"] == "v1.0"]
    v2 = comparison[comparison["prompt_version"] == "v2.0"]

    if len(v1) and len(v2):
        ax.bar(
            x - width / 2,
            v1[metrics].values[0] * 100,
            width,
            label="v1.0 (Baseline)",
            color="#F97316",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            v2[metrics].values[0] * 100,
            width,
            label="v2.0 (Improved)",
            color="#0D9488",
            alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Rate (%)")
        ax.set_title("Prompt Improvement: v1.0 -> v2.0 (Lower is Better)", fontsize=13, fontweight="bold")
        ax.legend()
        y_values = np.concatenate([v1[metrics].to_numpy().ravel(), v2[metrics].to_numpy().ravel()])
        finite_values = y_values[np.isfinite(y_values)]
        y_max = float(finite_values.max()) if finite_values.size else 0.01
        ax.set_ylim(0, max(y_max, 0.01) * 130)

        for i, metric in enumerate(metrics):
            v1_val = v1[metric].values[0] * 100
            v2_val = v2[metric].values[0] * 100
            delta = v2_val - v1_val
            color = "#10B981" if delta < 0 else "#EF4444"
            ax.text(
                i + width / 2,
                v2_val + 0.3,
                f"{delta:+.1f}%",
                ha="center",
                va="bottom",
                color=color,
                fontweight="bold",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig("before_after_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

print("Comparison chart saved: before_after_comparison.png")


print(
    """
REFLECTION QUESTIONS (discuss in pairs)

1. Which failure category had the highest business impact for FinSight? Why?
2. The v2.0 prompt is longer - does that mean higher cost? Look at your avg_cost comparison.
3. If hallucination rate is still > 0% after v2.0, what would your next experiment be?
4. How would you set up alerting so the team is notified within 1 hour if hallucination rate exceeds 2%?
5. What data would you need to prove this improvement is statistically significant?
"""
)

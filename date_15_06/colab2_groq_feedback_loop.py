# -*- coding: utf-8 -*-
"""Notebook conversion of `colab2_groq_feedback_loop.ipynb`.

This script keeps the lab flow from the notebook but runs as a normal Python
file and reads the Groq API key from the project-root `.env`.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401
from tqdm.auto import tqdm

from groq import Groq

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


GROQ_API_KEY = load_project_env("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in project .env")

groq_client = Groq(api_key=GROQ_API_KEY)

# Model to use for memo generation (from Lab 1 winner)
PRODUCTION_MODEL = "llama-3.3-70b-versatile"

# Groq cost helper
GROQ_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}


def compute_cost(model_id, input_tokens, output_tokens):
    p = GROQ_PRICING.get(model_id, {"input": 0.5, "output": 0.5})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


print("? Setup complete")
print(f"   Production model: {PRODUCTION_MODEL}")


# ------------------------------------------------------------------------------
# # 🔁 Lab 2 — Prompt Feedback Loop with Automatic Iteration (Groq Edition)
# **Day 13 · Module 5 · FinSight AI Credit Risk Scenario**
#
# ---
#
# ## 🎯 Learning Objectives
# By the end of this lab you will be able to:
# 1. Instrument LLM calls with structured logging to a persistent SQLite store
# 2. Build an automated quality probe to detect failure modes at scale
# 3. Triage failure patterns using frequency analysis
# 4. Write a revised prompt based on failure evidence, not intuition
# 5. Demonstrate measurable improvement through a before/after evaluation
#
# ## 🏦 Backbone Scenario: FinSight AI — Production Feedback Loop
# FinSight has deployed **LLaMA 3.3 70B on Groq** (from Lab 1). After 2 weeks in production,
# the team has collected 50 memo requests. Your job: **find what's failing and fix it**.
#
# > 💡 All API calls in this lab use **Groq only** — a single `GROQ_API_KEY` is all you need.
#
# **⏱ Estimated time: 50 minutes**  
# **🔧 Extension task: Connect to LangSmith or Weights & Biases for live tracing — see bottom**
#
# ---

# ------------------------------------------------------------------------------
# ## Step 0 — Install & Import

# ------------------------------------------------------------------------------
# !pip install groq bert-score pandas matplotlib seaborn numpy scikit-learn tqdm -q
# print('✅ Dependencies installed')

# ------------------------------------------------------------------------------
import os, sqlite3, json, time, re, hashlib, random, uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, timedelta

# ── API Key ───────────────────────────────────────────────────────
GROQ_API_KEY = load_project_env("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in project .env")

from groq import Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Model to use for memo generation (from Lab 1 winner) ─────────
# Change this to experiment with different Groq models
PRODUCTION_MODEL = 'llama-3.3-70b-versatile'

# ── Groq cost helper ─────────────────────────────────────────────
GROQ_PRICING = {
    'llama-3.3-70b-versatile': {'input': 0.59, 'output': 0.79},
    'llama-3.1-8b-instant':    {'input': 0.05, 'output': 0.08},
    'mixtral-8x7b-32768':      {'input': 0.24, 'output': 0.24},
    'gemma2-9b-it':            {'input': 0.20, 'output': 0.20},
}

def compute_cost(model_id, input_tokens, output_tokens):
    p = GROQ_PRICING.get(model_id, {'input': 0.5, 'output': 0.5})
    return (input_tokens * p['input'] + output_tokens * p['output']) / 1_000_000

print('✅ Setup complete')
print(f'   Production model: {PRODUCTION_MODEL}')

# ------------------------------------------------------------------------------
# ## Step 1 — Design the Logging Schema & SQLite Database

# ------------------------------------------------------------------------------
# ── Structured log entry dataclass ───────────────────────────────────
@dataclass
class LLMLogEntry:
    request_id:        str     # Unique ID for this call
    timestamp:         str     # ISO 8601
    prompt_version:    str     # e.g. 'v1.0', 'v1.1', 'v2.0'
    prompt_hash:       str     # MD5 of prompt template (for deduplication)
    model:             str     # Groq model ID
    task_type:         str     # 'credit_memo'
    borrower_id:       str     # Anonymised ID
    input_tokens:      int
    output_tokens:     int
    latency_ms:        float
    cost_usd:          float
    output_text:       str
    output_word_count: int
    # Quality signals (populated post-call)
    auto_score:           Optional[float] = None   # BERTScore or rubric score
    hallucination_flag:   Optional[bool]  = None
    hallucination_count:  Optional[int]   = None
    missing_sections:     Optional[str]   = None   # JSON list of absent sections
    user_correction:      Optional[bool]  = None   # Simulated analyst feedback
    failure_category:     Optional[str]   = None   # Populated during triage
    environment:          str = 'production'


def init_db(db_path: str = 'finsight_logs_groq.db') -> sqlite3.Connection:
    """Create SQLite database with logging schema."""
    conn = sqlite3.connect(db_path)
    conn.execute('''
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
    ''')
    conn.commit()
    return conn


def log_entry(conn: sqlite3.Connection, entry: LLMLogEntry) -> None:
    """Insert a log entry into the database."""
    d = asdict(entry)
    d['hallucination_flag'] = int(d['hallucination_flag']) if d['hallucination_flag'] is not None else None
    d['user_correction']    = int(d['user_correction'])    if d['user_correction']    is not None else None
    conn.execute(
        f"INSERT OR REPLACE INTO llm_logs VALUES ({','.join(['?']*len(d))})",
        list(d.values())
    )
    conn.commit()


conn = init_db()
print('✅ SQLite database initialised: finsight_logs_groq.db')
print('   Schema:', [col[0] for col in conn.execute('PRAGMA table_info(llm_logs)').fetchall()])

# ------------------------------------------------------------------------------
# ## Step 2 — Define Prompt Variants

# ------------------------------------------------------------------------------
# ── Prompt variants to test ────────────────────────────────────────────
# v1.0: Minimal prompt (baseline — simulates what a rushed team ships)
PROMPT_V1_0 = {
    'version': 'v1.0',
    'system':  'You are a credit analyst. Generate a credit risk memo based on the borrower data.',
    'user_template': 'Borrower data:\n{data}\n\nWrite a credit memo.',
}

# v1.1: Structured output prompt
PROMPT_V1_1 = {
    'version': 'v1.1',
    'system': '''You are a credit analyst AI at FinSight AI.
Generate a credit risk memo (150-250 words) structured as:
1. BORROWER OVERVIEW
2. KEY FINANCIAL METRICS  
3. RISK ASSESSMENT
4. RECOMMENDATION
Use only the data provided. Do not extrapolate or add information not given.''',
    'user_template': 'Generate a credit memo for:\n{data}',
}

# v2.0: Chain-of-thought + anti-hallucination guardrails
PROMPT_V2_0 = {
    'version': 'v2.0',
    'system': '''You are a senior credit analyst at FinSight AI, operating in a regulated lending environment.

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
- If data is inconsistent, note it: "Note: figures appear inconsistent — [detail]"
- Use hedged language for projections: "based on provided data" not "will" or "certainly"
- If insufficient data exists for a section, write: "Insufficient data provided for [section]"''',
    'user_template': 'Generate a credit memo for the following borrower.\n\nBORROWER DATA:\n{data}',
}

ALL_PROMPTS = [PROMPT_V1_0, PROMPT_V1_1, PROMPT_V2_0]

def hash_prompt(system: str, template: str) -> str:
    return hashlib.md5((system + template).encode()).hexdigest()[:8]

for p in ALL_PROMPTS:
    h = hash_prompt(p['system'], p['user_template'])
    print(f"  {p['version']} → hash: {h}")
print('✅ Prompt variants defined')

# ------------------------------------------------------------------------------
# ## Step 3 — Simulate 50 Production Requests with Full Logging

# ------------------------------------------------------------------------------
# ── 50 FinSight borrower profiles (varied to exercise prompt weaknesses) ──
BORROWER_PROFILES = [
    # Clean cases
    'Northgate Logistics. Revenue $8.2M. EBITDA $1.1M. Debt $2.5M. DSCR 1.7x. Loan request $1.5M.',
    'Sunrise Bakeries. Revenue $3.4M. Net profit $280K. No existing debt. Loan request $400K equipment.',
    'Harbor Bridge Tech. ARR $2.1M. Growth 18% YoY. Burn $85K/mo. Runway 18mo. Loan request $300K.',
    'Westfield Industrial. Revenue $19.5M. EBITDA $3.2M. Existing debt $6.8M. Collateral $9.1M. Loan $2M.',
    'Blue Ridge Farms. Revenue $5.9M. Seasonal. DSCR 1.4x. Crop insurance. Loan request $900K operating line.',
    'Delta Medical Supplies. Revenue $11.2M. Gross margin 34%. DSCR 2.1x. 8yr history. Loan $1.8M.',
    'Summit Construction. Revenue $28M. Gross margin 9%. Active contracts $12M. DSCR 1.5x. Loan $3.5M.',
    'Coastal Realty Trust. Properties 8. NAV $14.2M. LTV 58%. Cash coverage 1.8x. Loan $1.5M.',
    'Ironwood Software. ARR $4.8M. NRR 118%. No debt. Founder personal guarantee. Loan $600K.',
    'Pacific Fisheries. Revenue $7.1M. EBITDA $900K. Fleet value $3.8M. DSCR 1.3x. Loan $1.1M.',
    # Slightly tricky
    'NovaStar Retail. Revenue $15M (down 12% YoY). EBITDA $800K. High lease obligations $3.2M/yr. Loan $2M.',
    'Vertex Energy. Revenue $9.2M. EBITDA $1.4M. BUT Q3 one-time gain $600K not excluded. True EBITDA $800K. Loan $1.5M.',
    'Cascade Healthcare. Revenue $6.8M. Receivables $2.1M (90-day overdue 35%). DSCR 1.2x. Loan $800K.',
    'Alpine Hotels. Revenue $12M. Seasonality: 60% summer. Q4 DSCR 0.7x. Annual DSCR 1.4x. Loan $2.2M.',
    'TechVenture Alpha. Pre-revenue. VC backed $3M. Burn $180K/mo. Patent pending. Loan $400K bridge.',
    'Heritage Textiles. Revenue $4.1M. Stable. Owner age 68 no succession plan. DSCR 1.9x. Loan $600K.',
    'Meridian Imports. Revenue $22M. Single customer 65% concentration. DSCR 2.3x. Loan $3M.',
    'Quantum Logistics. Revenue $17M. Growing. Tight working capital. Current ratio 0.9. Loan $2.5M.',
    'Bluestone Mining. Revenue $8.7M. Commodity price sensitive. Insurance in place. DSCR 1.6x. Loan $1.4M.',
    'RapidGrow Foods. Revenue $6.2M. EBITDA margin stated 28% — typical for sector is 12-15%. Loan $1M.',
    # Adversarial: missing data
    'Unnamed LLC. Some revenue. Requesting $500K. Business started recently.',
    'Apex Corp. Revenue not disclosed. Assets $2M. Loan $800K.',
    'XYZ Holdings. Revenue $5M. No other financial data provided. Need $750K urgently.',
    'Global Partners. Profitable. Good DSCR. Loan $1.2M.',
    'MegaBuild Inc. Revenue $100M. EBITDA $50M. (Implausibly high margin for construction at 50%). Loan $5M.',
    # Adversarial: inconsistent numbers
    'BrightPath Energy. Revenue $12M. Net income $3.6M (30% margin). But tax paid $50K only. DSCR 1.8x. Loan $2M.',
    'Frontier Exports. Revenue $18M. EBITDA $2.7M. Debt service $2.1M. Stated DSCR 1.8x (actual: 1.3x). Loan $3M.',
    'ClearView Media. ARR $1.8M. MRR $200K. Note: $200K x 12 = $2.4M not $1.8M. Loan $400K.',
    'Skyline Builders. Properties appraised $20M. Debt $15M. Stated LTV 60% (actual 75%). Loan $2.5M.',
    'Neptune Pharma. Revenue $9.1M. Gross margin 72%. SG&A $6.8M. Implied operating margin 2% but claims 15%. Loan $1.5M.',
] * 2  # Repeat to reach 50 entries

BORROWER_PROFILES = BORROWER_PROFILES[:50]


# ── Groq caller with full telemetry ──────────────────────────────
def call_model_with_prompt(prompt_cfg: dict, borrower_data: str,
                            model_id: str = PRODUCTION_MODEL) -> dict:
    """Call Groq and return output + token + latency + cost data."""
    user_msg = prompt_cfg['user_template'].format(data=borrower_data)
    start = time.time()
    try:
        resp = groq_client.chat.completions.create(
            model=model_id,
            max_tokens=500,
            messages=[
                {'role': 'system', 'content': prompt_cfg['system']},
                {'role': 'user',   'content': user_msg}
            ],
            temperature=0.0,
        )
        latency_ms  = (time.time() - start) * 1000
        output      = resp.choices[0].message.content
        in_tokens   = resp.usage.prompt_tokens
        out_tokens  = resp.usage.completion_tokens
        return {
            'output':        output,
            'input_tokens':  in_tokens,
            'output_tokens': out_tokens,
            'latency_ms':    round(latency_ms, 1),
            'cost':          compute_cost(model_id, in_tokens, out_tokens),
            'error':         None,
        }
    except Exception as e:
        return {'output': '', 'input_tokens': 0, 'output_tokens': 0,
                'latency_ms': (time.time()-start)*1000, 'cost': 0, 'error': str(e)}


# ── Run simulation: v1.0 prompt on 50 requests ────────────────────
print('Simulating 50 production requests with prompt v1.0...')
print('(Set SIMULATE_N = 50 for the full run; 10 shown for speed)')

SIMULATE_N = 10   # ← Change to 50 for full simulation
PROMPT_CFG = PROMPT_V1_0  # Baseline prompt

sim_start = datetime(2025, 6, 1, 9, 0, 0)

for i, borrower_data in enumerate(tqdm(BORROWER_PROFILES[:SIMULATE_N], desc='Simulating')):
    result = call_model_with_prompt(PROMPT_CFG, borrower_data)

    entry = LLMLogEntry(
        request_id      = str(uuid.uuid4()),
        timestamp       = (sim_start + timedelta(hours=i*0.5)).isoformat(),
        prompt_version  = PROMPT_CFG['version'],
        prompt_hash     = hash_prompt(PROMPT_CFG['system'], PROMPT_CFG['user_template']),
        model           = PRODUCTION_MODEL,
        task_type       = 'credit_memo',
        borrower_id     = f'BRW-{i+1:04d}',
        input_tokens    = result['input_tokens'],
        output_tokens   = result['output_tokens'],
        latency_ms      = result['latency_ms'],
        cost_usd        = result['cost'],
        output_text     = result['output'],
        output_word_count = len(result['output'].split()),
        environment     = 'simulation',
    )
    log_entry(conn, entry)
    time.sleep(0.1)

count = conn.execute('SELECT COUNT(*) FROM llm_logs').fetchone()[0]
print(f'\n✅ {count} log entries written to SQLite')

# ------------------------------------------------------------------------------
# ## Step 4 — Run Automated Quality Probes

# ------------------------------------------------------------------------------
# ── Quality probe suite ───────────────────────────────────────────────

REQUIRED_SECTIONS = [
    'borrower',    # Must mention borrower name or overview
    'financial',   # Must mention financial figures
    'risk',        # Must contain risk assessment
    'recommend',   # Must contain a recommendation
]


def probe_hallucination(borrower_data: str, output: str) -> tuple:
    """Check for numeric hallucination (values in output not in source)."""
    def extract_nums(text):
        raw  = re.findall(r'\$?[\d,]+\.?\d*[KMB]?', text)
        nums = set()
        for r in raw:
            r_clean = r.replace('$','').replace(',','')
            try:
                if r_clean.endswith('K'):   nums.add(float(r_clean[:-1]) * 1e3)
                elif r_clean.endswith('M'): nums.add(float(r_clean[:-1]) * 1e6)
                elif r_clean.endswith('B'): nums.add(float(r_clean[:-1]) * 1e9)
                else:                        nums.add(float(r_clean))
            except: pass
        return nums

    source_nums = extract_nums(borrower_data)
    output_nums = extract_nums(output)

    hallucinated = [
        v for v in output_nums
        if v > 1000  # Only flag substantial figures
        and not any(abs(v - s) / max(s, 0.01) < 0.05 for s in source_nums if s > 0)
    ]
    return len(hallucinated) > 0, len(hallucinated)


def probe_missing_sections(output: str) -> list:
    """Detect which required sections are absent."""
    output_lower = output.lower()
    return [s for s in REQUIRED_SECTIONS if s not in output_lower]


def probe_output_length(output: str) -> bool:
    """Flag outputs outside the 150-250 word target range."""
    wc = len(output.split())
    return wc < 100 or wc > 350


def simulate_user_correction(output: str, borrower_data: str) -> bool:
    """Simulate analyst feedback: ~60% chance of flagging if output has issues."""
    has_issue = (probe_missing_sections(output) or
                 probe_hallucination(borrower_data, output)[0] or
                 probe_output_length(output))
    return has_issue and random.random() < 0.6


# ── Apply probes to all logged entries ────────────────────────────
df_logs = pd.read_sql('SELECT * FROM llm_logs', conn)

for idx, row in df_logs.iterrows():
    if not row['output_text']: continue

    borrower_data = BORROWER_PROFILES[int(row['borrower_id'].split('-')[1]) - 1]
    hall_flag, hall_count = probe_hallucination(borrower_data, row['output_text'])
    missing    = probe_missing_sections(row['output_text'])
    correction = simulate_user_correction(row['output_text'], borrower_data)
    length_vio = probe_output_length(row['output_text'])

    # Categorise failure
    if hall_flag:        cat = 'hallucination'
    elif missing:        cat = 'missing_sections'
    elif length_vio:     cat = 'length_violation'
    elif correction:     cat = 'analyst_correction'
    else:                cat = None

    # Update in-memory dataframe
    df_logs.at[idx, 'hallucination_flag']  = int(hall_flag)
    df_logs.at[idx, 'hallucination_count'] = hall_count
    df_logs.at[idx, 'missing_sections']    = json.dumps(missing)
    df_logs.at[idx, 'user_correction']     = int(correction)
    df_logs.at[idx, 'failure_category']    = cat

    # Persist to SQLite
    conn.execute('''UPDATE llm_logs SET
        hallucination_flag=?, hallucination_count=?,
        missing_sections=?, user_correction=?, failure_category=?
        WHERE request_id=?''',
        (int(hall_flag), hall_count, json.dumps(missing),
         int(correction), cat, row['request_id']))

conn.commit()

print('✅ Quality probes applied')
print(f'   Total entries: {len(df_logs)}')
print(f'   Failures: {df_logs["failure_category"].notna().sum()} '
      f'({df_logs["failure_category"].notna().mean()*100:.0f}%)')
print(f'\nFailure breakdown:')
print(df_logs['failure_category'].value_counts(dropna=False))

# ------------------------------------------------------------------------------
# ## Step 5 — Failure Triage

# ------------------------------------------------------------------------------
# ── Analyse failure patterns ─────────────────────────────────────────
df_logs = pd.read_sql('SELECT * FROM llm_logs', conn)

failures = df_logs[df_logs['failure_category'].notna()].copy()

print(f'\n📊 FAILURE TRIAGE REPORT — Prompt {PROMPT_V1_0["version"]}')
print('=' * 60)
print(f'Total requests:   {len(df_logs)}')
print(f'Failures:         {len(failures)} ({len(failures)/len(df_logs)*100:.0f}%)')
print(f'\nFailure categories:')
print(failures['failure_category'].value_counts().to_string())

print(f'\n📋 Sample failing outputs (worst first by hallucination):')
worst = df_logs.sort_values('hallucination_count', ascending=False).head(3)
for _, row in worst.iterrows():
    if row['output_text']:
        print(f'\n  borrower_id: {row["borrower_id"]} | failure: {row["failure_category"]}')
        print(f'  Output preview: {row["output_text"][:200]}...')

# ------------------------------------------------------------------------------
# ## Step 6 — Inspect Worst Performers

# ------------------------------------------------------------------------------
# ── Deep-dive into highest-hallucination outputs ─────────────────────
print('\n🔍 WORST PERFORMERS — Hallucination Deep-Dive')
print('=' * 60)

worst_5 = df_logs.sort_values('hallucination_count', ascending=False).head(5)

for _, row in worst_5.iterrows():
    if not row['output_text']: continue
    borrower_data = BORROWER_PROFILES[int(row['borrower_id'].split('-')[1]) - 1]
    print(f'\n📋 Request: {row["request_id"][:8]}... | Borrower: {row["borrower_id"]}')
    print(f'   Hallucination count: {row["hallucination_count"]}')
    print(f'   Missing sections: {row["missing_sections"]}')
    print(f'   Word count: {row["output_word_count"]}')
    print(f'   Latency: {row["latency_ms"]:.0f}ms | Cost: ${row["cost_usd"]:.6f}')
    print(f'\n   BORROWER DATA: {borrower_data[:150]}...')
    print(f'   MODEL OUTPUT:  {row["output_text"][:300]}...')

# ── Identify root cause ──────────────────────────────────────────
print('\n\n🔑 ROOT CAUSE ANALYSIS')
print('The v1.0 prompt provides no structural guidance or anti-hallucination rules.')
print('This causes models to:')
print('  1. Fill in missing data with plausible-sounding fabricated numbers')
print('  2. Skip required sections (no template enforced)')
print('  3. Vary output length unpredictably')
print('\n→ Solution: upgrade to v2.0 with explicit structure and compliance rules.')

# ------------------------------------------------------------------------------
# ## Step 7 — Run v2.0 Prompt on Same Inputs

# ------------------------------------------------------------------------------
# ── Re-run the same 50 borrower profiles with the improved prompt ─────
print('Running v2.0 prompt on same inputs...')

PROMPT_CFG_V2 = PROMPT_V2_0

for i, borrower_data in enumerate(tqdm(BORROWER_PROFILES[:SIMULATE_N], desc='v2.0 run')):
    result = call_model_with_prompt(PROMPT_CFG_V2, borrower_data)

    hall_flag, hall_count = probe_hallucination(borrower_data, result['output'])
    missing    = probe_missing_sections(result['output'])
    correction = simulate_user_correction(result['output'], borrower_data)
    length_vio = probe_output_length(result['output'])

    if hall_flag:        cat = 'hallucination'
    elif missing:        cat = 'missing_sections'
    elif length_vio:     cat = 'length_violation'
    elif correction:     cat = 'analyst_correction'
    else:                cat = None

    entry = LLMLogEntry(
        request_id      = str(uuid.uuid4()),
        timestamp       = (sim_start + timedelta(hours=i*0.5 + 100)).isoformat(),
        prompt_version  = PROMPT_CFG_V2['version'],
        prompt_hash     = hash_prompt(PROMPT_CFG_V2['system'], PROMPT_CFG_V2['user_template']),
        model           = PRODUCTION_MODEL,
        task_type       = 'credit_memo',
        borrower_id     = f'BRW-{i+1:04d}',
        input_tokens    = result['input_tokens'],
        output_tokens   = result['output_tokens'],
        latency_ms      = result['latency_ms'],
        cost_usd        = result['cost'],
        output_text     = result['output'],
        output_word_count = len(result['output'].split()),
        environment     = 'simulation',
    )

    entry.hallucination_flag  = hall_flag
    entry.hallucination_count = hall_count
    entry.missing_sections    = json.dumps(missing)
    entry.user_correction     = correction
    entry.failure_category    = cat

    log_entry(conn, entry)
    time.sleep(0.1)

print('✅ v2.0 run complete')

# ------------------------------------------------------------------------------
# ## Step 8 — Before/After Comparison Dashboard

# ------------------------------------------------------------------------------
df_all = pd.read_sql('SELECT * FROM llm_logs', conn)

# ── Compute metrics per prompt version ───────────────────────────
comparison = df_all.groupby('prompt_version').agg(
    n_requests       = ('request_id',         'count'),
    hallucin_rate    = ('hallucination_flag',  'mean'),
    missing_sec_rate = ('missing_sections',    lambda x: (x != '[]').mean()),
    user_correction  = ('user_correction',     'mean'),
    avg_latency_ms   = ('latency_ms',          'mean'),
    avg_cost         = ('cost_usd',            'mean'),
    avg_word_count   = ('output_word_count',   'mean'),
    failure_rate     = ('failure_category',    lambda x: x.notna().mean()),
).round(4).reset_index()

comparison = comparison[comparison['prompt_version'].isin(['v1.0', 'v2.0'])]

print('\n📊 BEFORE / AFTER PROMPT COMPARISON')
print('=' * 70)
print(comparison[['prompt_version','hallucin_rate','missing_sec_rate',
                  'user_correction','failure_rate','avg_cost']].to_string(index=False))

# ── Visualise ─────────────────────────────────────────────────────
if len(comparison) >= 2:
    metrics = ['hallucin_rate', 'missing_sec_rate', 'user_correction', 'failure_rate']
    labels  = ['Hallucination\nRate', 'Missing\nSections', 'User\nCorrections', 'Overall\nFailure Rate']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    width = 0.35

    v1 = comparison[comparison['prompt_version']=='v1.0']
    v2 = comparison[comparison['prompt_version']=='v2.0']

    if len(v1) and len(v2):
        ax.bar(x - width/2, v1[metrics].values[0]*100, width,
               label='v1.0 (Baseline)',  color='#F97316', alpha=0.85)
        ax.bar(x + width/2, v2[metrics].values[0]*100, width,
               label='v2.0 (Improved)', color='#0D9488', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Rate (%)')
        ax.set_title(f'Prompt Improvement: v1.0 → v2.0 (Groq: {PRODUCTION_MODEL})', fontsize=13, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, max(v1[metrics].values[0].max(), 0.01) * 130)

        # Add delta labels
        for i, metric in enumerate(metrics):
            v1_val = v1[metric].values[0] * 100
            v2_val = v2[metric].values[0] * 100
            delta  = v2_val - v1_val
            color  = '#10B981' if delta < 0 else '#EF4444'
            ax.text(i + width/2, v2_val + 0.3, f'{delta:+.1f}%',
                    ha='center', va='bottom', color=color, fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('before_after_comparison_groq.png', dpi=150, bbox_inches='tight')
    plt.show()

print('✅ Comparison chart saved: before_after_comparison_groq.png')

# ------------------------------------------------------------------------------
# ## Step 9 — Reflection Questions

# ------------------------------------------------------------------------------
print("""
╔══════════════════════════════════════════════════════════╗
║           REFLECTION QUESTIONS (with answers)           ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. Which failure category had the highest business      ║
║     impact for FinSight? Why?                            ║
║     Answer: Hallucination, because fabricated numbers   ║
║     can directly distort credit decisions and increase   ║
║     compliance risk.                                    ║
║                                                          ║
║  2. The v2.0 prompt is longer - does that mean higher    ║
║     cost? Look at your avg_cost comparison.              ║
║     (Hint: Groq's pricing is per token, not per call)    ║
║     Answer: Not necessarily. Cost depends on total       ║
║     tokens, so a longer prompt may cost a bit more, but  ║
║     better structure can reduce retries and waste.       ║
║                                                          ║
║  3. If hallucination rate is still > 0% after v2.0,      ║
║     what would your next experiment be?                  ║
║     Answer: Add a post-generation fact checker or a      ║
║     stricter validation pass for every numeric claim.    ║
║                                                          ║
║  4. How would you set up alerting so the team is         ║
║     notified within 1 hour if hallucination rate         ║
║     exceeds 2%?                                          ║
║     Answer: Track a rolling 1-hour window, compare the   ║
║     hallucination rate to the 2% threshold, and send a   ║
║     Slack/email/page alert with the failing request IDs. ║
║                                                          ║
║  5. Groq delivers sub-second latency - what new          ║
║     product features does this enable for FinSight?      ║
║     Answer: Interactive memo drafting, live analyst      ║
║     review, faster underwriting, and real-time borrower  ║
║     feedback.                                            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

# ------------------------------------------------------------------------------
# ---
# ## 🚀 Extension Task — LangSmith / Weights & Biases Integration
#
# **⏱ +10 minutes | Advanced participants**
#
# ### Option A: LangSmith Tracing (Groq via OpenAI-compatible endpoint)
# 1. Install `langsmith` and `langchain-groq`
# 2. Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in secrets
# 3. Wrap your `call_model_with_prompt` function with a LangSmith tracer
# 4. View traces in the LangSmith UI — compare v1.0 vs v2.0 side by side
#
# ### Option B: Weights & Biases
# 1. Install `wandb`
# 2. Log each run as a W&B artifact: prompt version, Groq model, metrics, outputs
# 3. Create a W&B Table comparing model outputs
# 4. Use W&B Sweeps to auto-tune temperature for minimum hallucination rate

# ------------------------------------------------------------------------------
# ── EXTENSION A: LangSmith with Groq ──────────────────────────────────
# !pip install langsmith langchain-groq -q

import os
LANGCHAIN_API_KEY = load_project_env("LANGCHAIN_API_KEY")
ENABLE_LANGSMITH_TRACING = load_project_env("ENABLE_LANGSMITH_TRACING", "false").lower() in {
    "1",
    "true",
    "yes",
}
LANGSMITH_AVAILABLE = False
if ENABLE_LANGSMITH_TRACING and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = "training"
    try:
        from langsmith import traceable
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        LANGSMITH_AVAILABLE = True
    except ImportError:
        print("Skipping LangSmith tracing because langsmith/langchain-groq is not installed")
elif LANGCHAIN_API_KEY and not ENABLE_LANGSMITH_TRACING:
    print("LangSmith tracing is disabled by default. Set ENABLE_LANGSMITH_TRACING=true in .env to enable it.")
else:
    print("Skipping LangSmith tracing because LANGCHAIN_API_KEY is missing in .env")

if LANGSMITH_AVAILABLE:
    @traceable(name="credit_memo_groq")
    def call_model_with_prompt_traced(prompt_cfg, borrower_data):
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
        messages = [
            SystemMessage(content=prompt_cfg["system"]),
            HumanMessage(content=prompt_cfg["user_template"].format(data=borrower_data)),
        ]
        return llm.invoke(messages).content

    traced_v1 = call_model_with_prompt_traced(PROMPT_V1_0, BORROWER_PROFILES[0])
    traced_v2 = call_model_with_prompt_traced(PROMPT_V2_0, BORROWER_PROFILES[0])
    print("Traced v1.0 output:", traced_v1[:200])
    print("Traced v2.0 output:", traced_v2[:200])
    print("View traces at: https://smith.langchain.com")
else:
    print("Using local execution only for the LangSmith extension section.")


# ── EXTENSION B: Weights & Biases ─────────────────────────────────
# !pip install wandb -q
WANDB_API_KEY = load_project_env("WANDB_API_KEY")
WANDB_AVAILABLE = False
try:
    import tempfile
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Skipping W&B logging because wandb is not installed")

if WANDB_AVAILABLE:
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    else:
        print("Skipping W&B login because WANDB_API_KEY is missing in .env")

    run = wandb.init(
        project="finsight-prompt-eval-groq",
        config={"model": PRODUCTION_MODEL, "version": "2.0"},
    )
    table = wandb.Table(
        columns=["borrower_id", "prompt_version", "output", "hallucination", "failure_cat", "cost_usd"]
    )
    for _, row in df_all.iterrows():
        table.add_data(
            row["borrower_id"],
            row["prompt_version"],
            row["output_text"][:200],
            bool(row["hallucination_flag"]),
            row["failure_category"],
            row["cost_usd"],
        )
    wandb.log({"results": table})

    artifact = wandb.Artifact("finsight-prompt-eval", type="evaluation")
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "finsight_prompt_eval_summary.csv"
        df_all[[
            "request_id",
            "timestamp",
            "prompt_version",
            "model",
            "borrower_id",
            "hallucination_flag",
            "failure_category",
            "cost_usd",
            "output_text",
        ]].to_csv(artifact_path, index=False)
        artifact.add_file(str(artifact_path))
        run.log_artifact(artifact)

    SWEEP_CONFIG = {
        "method": "bayes",
        "metric": {"name": "hallucination_rate", "goal": "minimize"},
        "parameters": {"temperature": {"values": [0.0, 0.1, 0.2, 0.3]}},
    }
    print("W&B artifact logged: finsight-prompt-eval")
    print("Sweep config ready for temperature tuning:", SWEEP_CONFIG)
    run.finish()


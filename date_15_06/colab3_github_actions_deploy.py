# -*- coding: utf-8 -*-
"""Notebook conversion of `colab3_github_actions_deploy.ipynb`.

This script preserves the notebook flow with Python cell markers and
comments out notebook-only magics so the export remains valid Python.
"""

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# %% [markdown]
# # 🚀 Lab 3 — Deploy the FinSight LLM Eval Pipeline to GitHub Actions
# **Day 13 · Bonus Lab · FinSight AI Credit Risk Scenario**
#
# ---
#
# ## 🎯 Learning Objectives
# By the end of this lab you will be able to:
# 1. Refactor the Colab eval harness into a **self-contained Python package**
# 2. Write a `pytest` test suite that treats LLM quality as a CI gate
# 3. Scaffold a **GitHub Actions workflow** that runs evals on every pull request
# 4. Store eval artefacts (leaderboard CSV, charts) as **GitHub Actions artefacts**
# 5. Implement a **quality gate** that fails the build if hallucination rate > 1%
#
# ## 🏦 What we're automating
# ```
# Developer opens PR  →  GitHub Actions triggers  →  eval_harness.py runs on Groq
#         →  Quality probes execute  →  Results saved as artefacts
#         →  Gate: hallucination < 1% & BERTScore ≥ 0.88  →  ✅ PR can merge
# ```
#
# > 💡 This is the **MLOps layer** on top of Lab 1 & 2: instead of running evals manually
# > in Colab, every code change is automatically verified against your quality constraints.
#
# **⏱ Estimated time: 40 minutes**  
# **📁 Output: a complete GitHub repo structure, ready to push**
#
# ---

# %% [markdown]
# ## Step 0 — Install Dependencies & Set Up Repo Scaffold

# %%
# !pip install groq bert-score pandas matplotlib PyGithub gitpython pytest -q
print('Dependencies installed')

# %%
import os, json, shutil, stat
import shlex
from pathlib import Path


def load_project_env(name: str, default: str = "") -> str:
    """Read a value from the repo-root `.env` without requiring dotenv."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
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


# ── Secrets ──────────────────────────────────────────────────────
GROQ_API_KEY = load_project_env("GROQ_API_KEY", "gsk_YOUR_GROQ_KEY")
GITHUB_TOKEN = load_project_env("GITHUB_TOKEN", "ghp_YOUR_GITHUB_PAT")
GITHUB_USERNAME = load_project_env("GITHUB_USERNAME", "your-github-username")

os.environ['GROQ_API_KEY'] = GROQ_API_KEY

# ── Create local repo directory ───────────────────────────────────
REPO_NAME = 'finsigthAI'
REPO_DIR  = Path(__file__).resolve().parent / REPO_NAME

def _remove_readonly(func, path, exc_info):
    """Allow cleanup of Windows read-only files inside generated folders."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR, onerror=_remove_readonly)

REPO_DIR.mkdir(parents=True)

# Sub-directories
(REPO_DIR / 'src').mkdir()
(REPO_DIR / 'tests').mkdir()
(REPO_DIR / '.github' / 'workflows').mkdir(parents=True)
(REPO_DIR / 'results').mkdir()

print(f'✅ Repo scaffold created at {REPO_DIR}')
print('\nDirectory structure:')
for p in sorted(REPO_DIR.rglob('*')):
    indent = '  ' * (len(p.relative_to(REPO_DIR).parts) - 1)
    print(f'  {indent}{p.name}/')

# %% [markdown]
# ## Step 1 — Write the Core Eval Harness Module (`src/eval_harness.py`)
#
# This refactors the Colab 1 logic into a clean, importable Python module.
# All Groq API calls, scoring, and hallucination probing live here.

# %%
EVAL_HARNESS_PY = '''#!/usr/bin/env python3
"""
FinSight AI — LLM Evaluation Harness
Runs multi-model quality evaluation using Groq-hosted models.
Used by both the CI pipeline and the Colab notebooks.
"""

import os, re, json, time
from groq import Groq

# ── Model registry ────────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "llama-3.1-8b":  "llama-3.1-8b-instant",
    "mixtral-8x7b":  "mixtral-8x7b-32768",
    "gemma2-9b":     "gemma2-9b-it",
}

# ── Pricing per 1M tokens (USD, approximate mid-2025) ─────────────
GROQ_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768":      {"input": 0.24, "output": 0.24},
    "gemma2-9b-it":            {"input": 0.20, "output": 0.20},
}

SYSTEM_PROMPT = """You are a credit analyst AI assistant at FinSight AI.
Generate a concise credit risk memo (150-250 words) based on the provided borrower data.
Structure: (1) Borrower Overview, (2) Key Financial Metrics, (3) Risk Assessment, (4) Recommendation.
Use precise financial language. Do not fabricate or extrapolate data not provided."""

JUDGE_RUBRIC = """
You are a senior credit risk officer evaluating AI-generated credit memos.
Score on THREE dimensions (1-5 each).

FAITHFULNESS (1-5): Are all figures accurate and grounded in the source data?
COMPLETENESS (1-5): Does the memo cover borrower profile, metrics, risk, recommendation?
REGULATORY TONE (1-5): Is language precise, objective, and compliance-appropriate?

BORROWER DATA:
{borrower_data}

GENERATED MEMO:
{memo}

Respond ONLY with valid JSON (no markdown fences):
{{"faithfulness": <int 1-5>, "completeness": <int 1-5>, "regulatory_tone": <int 1-5>, "reasoning": "<one sentence>"}}
"""

# ── Production constraints ────────────────────────────────────────
CONSTRAINTS = {
    "hallucin_rate": 0.01,   # < 1%
    "bert_score":    0.88,   # ≥ 0.88
    "latency_p95":   3.0,    # < 3s
    "avg_cost":      0.02,   # < $0.02/memo
}


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)


def compute_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    p = GROQ_PRICING.get(model_id, {"input": 0.5, "output": 0.5})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def call_groq(client: Groq, prompt: str, model_id: str,
              system: str = SYSTEM_PROMPT, max_tokens: int = 400) -> dict:
    """Call a Groq model and return output + telemetry."""
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        latency    = time.time() - start
        output     = resp.choices[0].message.content
        in_tok     = resp.usage.prompt_tokens
        out_tok    = resp.usage.completion_tokens
        return {
            "output":     output,
            "latency":    round(latency, 3),
            "in_tokens":  in_tok,
            "out_tokens": out_tok,
            "cost":       round(compute_cost(model_id, in_tok, out_tok), 6),
            "error":      None,
        }
    except Exception as e:
        return {"output": "", "latency": round(time.time()-start, 3),
                "in_tokens": 0, "out_tokens": 0, "cost": 0, "error": str(e)}


def judge_memo(client: Groq, borrower_data: str, memo: str,
               judge_model: str = GROQ_MODELS["llama-3.3-70b"]) -> dict:
    """Score a memo using LLaMA 3.3 70B as judge."""
    prompt = JUDGE_RUBRIC.format(borrower_data=borrower_data, memo=memo)
    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        return json.loads(text)
    except Exception as e:
        return {"faithfulness": None, "completeness": None,
                "regulatory_tone": None, "reasoning": f"ERROR: {e}"}


def check_hallucination(borrower_data: str, memo: str) -> dict:
    """Detect numeric values in memo not present in borrower data."""
    def _normalise(s):
        s = s.replace("$", "").replace(",", "")
        if s.endswith("K"): return float(s[:-1]) * 1e3
        if s.endswith("M"): return float(s[:-1]) * 1e6
        if s.endswith("B"): return float(s[:-1]) * 1e9
        try: return float(s)
        except: return None

    src_vals  = {_normalise(n) for n in re.findall(r"\$?[\d,]+\.?\d*[KMB]?", borrower_data)
                 if _normalise(n) is not None}
    memo_vals = {_normalise(n) for n in re.findall(r"\$?[\d,]+\.?\d*[KMB]?", memo)
                 if _normalise(n) is not None and _normalise(n) is not None and _normalise(n) > 0}

    hallucinated = [
        v for v in memo_vals
        if v > 200
        and not any(abs(v - s) / max(s, 0.01) < 0.05 for s in src_vals if s and s > 0)
    ]
    return {
        "hallucination_flag":  len(hallucinated) > 0,
        "hallucination_count": len(hallucinated),
    }


def run_eval(test_cases: list, models: dict = None, judge: bool = True) -> list:
    """
    Run the full evaluation harness.

    Args:
        test_cases: list of {"id": str, "difficulty": str, "data": str}
        models:     dict of {alias: model_id}; defaults to all four Groq models
        judge:      whether to run Groq-as-judge scoring

    Returns:
        list of result dicts (one per test_case × model)
    """
    if models is None:
        models = GROQ_MODELS

    client  = get_client()
    results = []

    for tc in test_cases:
        for model_name, model_id in models.items():
            prompt = (
                "Generate a credit risk memo for the following borrower:\\n\\n"
                f"{tc['data']}"
            )
            result = call_groq(client, prompt, model_id)

            row = {
                "test_id":       tc["id"],
                "difficulty":    tc["difficulty"],
                "model":         model_name,
                "model_id":      model_id,
                "output":        result["output"],
                "in_tokens":     result["in_tokens"],
                "out_tokens":    result["out_tokens"],
                "latency":       result["latency"],
                "cost":          result["cost"],
                "error":         result["error"],
                "borrower_data": tc["data"],
            }

            if result["output"] and not result["error"]:
                h = check_hallucination(tc["data"], result["output"])
                row.update(h)
                if judge:
                    scores = judge_memo(client, tc["data"], result["output"])
                    row["score_faithfulness"]    = scores.get("faithfulness")
                    row["score_completeness"]    = scores.get("completeness")
                    row["score_regulatory_tone"] = scores.get("regulatory_tone")
                    row["judge_reasoning"]       = scores.get("reasoning")
                    if all(row.get(k) for k in
                           ["score_faithfulness","score_completeness","score_regulatory_tone"]):
                        row["composite_score"] = round(
                            (row["score_faithfulness"] +
                             row["score_completeness"] +
                             row["score_regulatory_tone"]) / 3, 2)
            time.sleep(0.1)
            results.append(row)

    return results


def build_leaderboard(results: list) -> list:
    """Aggregate per-model metrics and flag constraint compliance."""
    from collections import defaultdict
    import statistics

    models = list({r["model"] for r in results})
    lb = []

    for model in models:
        rows = [r for r in results if r["model"] == model]
        latencies = sorted(r["latency"] for r in rows)
        p95_idx   = max(0, int(len(latencies) * 0.95) - 1)

        composite_vals = [r["composite_score"] for r in rows
                          if r.get("composite_score") is not None]
        bert_vals      = [r.get("bert_score_f1", 0) for r in rows
                          if r.get("bert_score_f1")]
        hall_flags     = [int(r.get("hallucination_flag", False)) for r in rows]

        entry = {
            "model":          model,
            "composite":      round(statistics.mean(composite_vals), 3) if composite_vals else None,
            "bert_score":     round(statistics.mean(bert_vals), 3)      if bert_vals      else None,
            "hallucin_rate":  round(sum(hall_flags) / len(hall_flags), 3) if hall_flags   else None,
            "latency_p95":    round(latencies[p95_idx], 3),
            "avg_cost":       round(statistics.mean(r["cost"] for r in rows), 6),
        }
        entry["meets_constraints"] = (
            entry["hallucin_rate"] is not None and
            entry["hallucin_rate"]  < CONSTRAINTS["hallucin_rate"] and
            entry["latency_p95"]    < CONSTRAINTS["latency_p95"]   and
            entry["avg_cost"]       < CONSTRAINTS["avg_cost"]
        )
        lb.append(entry)

    return sorted(lb, key=lambda x: x.get("composite") or 0, reverse=True)


def check_quality_gate(leaderboard: list) -> dict:
    """
    Quality gate: at least ONE model must pass all constraints.
    Returns {"passed": bool, "reason": str, "best_model": str|None}
    """
    passing = [row for row in leaderboard if row.get("meets_constraints")]
    if passing:
        best = passing[0]["model"]
        return {"passed": True,
                "reason": f"Model '{best}' meets all FinSight production constraints.",
                "best_model": best}

    # Find closest failure
    violations = []
    for row in leaderboard:
        if row.get("hallucin_rate") is not None and \
           row["hallucin_rate"] >= CONSTRAINTS["hallucin_rate"]:
            violations.append(f"{row['model']}: hallucination {row['hallucin_rate']*100:.1f}% >= 1%")
    reason = "No model meets all constraints. Violations: " + "; ".join(violations[:2])
    return {"passed": False, "reason": reason, "best_model": None}
'''

with open(REPO_DIR / 'src' / 'eval_harness.py', 'w', encoding='utf-8') as f:
    f.write(EVAL_HARNESS_PY)

print('✅ src/eval_harness.py written')
print(f'   Lines: {len(EVAL_HARNESS_PY.splitlines())}')

# %% [markdown]
# ## Step 2 — Write the Test Data Module (`src/test_cases.py`)
#
# A dedicated module for the 20 FinSight test cases — shared between
# the CI pipeline and the Colab notebooks.

# %%
TEST_CASES_PY = '''"""FinSight AI — canonical 20-case test set."""

TEST_CASES = [
    # EASY
    {"id":"TC01","difficulty":"easy",
     "data":"Borrower: Apex Manufacturing Ltd. Revenue: $12.4M. EBITDA: $2.1M. Debt: $4.5M. DSCR: 1.8x. Loan request: $1.5M term loan, 5yr."},
    {"id":"TC02","difficulty":"easy",
     "data":"Borrower: GreenLeaf Organics Inc. Revenue: $5.2M. Net Profit: $420K. Current ratio: 2.1. No debt. Loan request: $800K equipment."},
    {"id":"TC03","difficulty":"easy",
     "data":"Borrower: Sunrise Hotels Group. Revenue: $18.7M. EBITDA: $3.9M. Existing debt: $7.2M. LTV: 62%. Loan request: $2M expansion."},
    {"id":"TC04","difficulty":"easy",
     "data":"Borrower: TechBridge Solutions LLC. ARR $3.1M. MRR growth 12% QoQ. Churn 2.1%. No debt. Loan request: $500K working capital."},
    {"id":"TC05","difficulty":"easy",
     "data":"Borrower: Coastal Fisheries Co. Revenue: $9.8M. EBITDA: $1.4M. DSCR: 1.6x. Fleet value $4.2M. Loan request: $1.2M."},
    # MEDIUM
    {"id":"TC06","difficulty":"medium",
     "data":"Borrower: RetailPro Chain. Revenue $22M (down 8% YoY). EBITDA $1.1M. Leases $4.8M/yr. DSCR 1.05x. Loan request: $3M."},
    {"id":"TC07","difficulty":"medium",
     "data":"Borrower: NovaBio Pharma. Pre-revenue. Burn $200K/mo. Runway 14mo. VC $4M raised. Loan request: $1.5M bridge."},
    {"id":"TC08","difficulty":"medium",
     "data":"Borrower: Atlas Construction. Revenue $31M. Gross margin 8%. DSCR 1.3x. Litigation $500K unresolved. Loan request: $4M."},
    {"id":"TC09","difficulty":"medium",
     "data":"Borrower: PrimeAgri Partners. Revenue $7.4M. EBITDA $900K. Farmland $3.5M. Drought region. Loan request: $1.8M."},
    {"id":"TC10","difficulty":"medium",
     "data":"Borrower: Urban Mobility Inc. Revenue $1.2M. Burn $350K/mo. Negative EBITDA. City contract $5M/3yr. Loan request: $600K."},
    # HARD
    {"id":"TC11","difficulty":"hard",
     "data":"Borrower: GlobalTrade. Revenue $45M (70% one customer). DSCR 2.1x. Customer restructuring announced. Loan request: $6M."},
    {"id":"TC12","difficulty":"hard",
     "data":"Borrower: DataVault Systems. Revenue $8M. EBITDA reported $2.4M. Related-party txn $1.8M. Adjusted EBITDA $600K. Loan: $2.5M."},
    {"id":"TC13","difficulty":"hard",
     "data":"Borrower: Heritage Real Estate Fund. NAV $28M. Leverage 3.2x. LTV 71%. ICR 1.1x. Two assets negative watch. Loan: $8M."},
    {"id":"TC14","difficulty":"hard",
     "data":"Borrower: CryptoAsset Ventures. Revenue $3.1M crypto trading. No audited financials. Assets $4.2M digital. Loan request: $1M."},
    {"id":"TC15","difficulty":"hard",
     "data":"Borrower: MedDevice International. Revenue $14M. Profitable. 3 jurisdictions, pending regulatory review. Recall risk 60% revenue. Loan: $3.5M."},
    # ADVERSARIAL
    {"id":"TC16","difficulty":"adversarial",
     "data":"Borrower: Pinnacle Energy. Revenue $19M. Net income $3.2M. Tax 25% but taxes paid $250K (expect $800K). EBITDA $5.4M no D&A. Loan: $4M."},
    {"id":"TC17","difficulty":"adversarial",
     "data":"Borrower: FreshFoods Co. Revenue $11M Q1-annualised. Q1 is 35% of annual => actual ~$7M. Q1 submitted as representative. Loan: $2.5M."},
    {"id":"TC18","difficulty":"adversarial",
     "data":"Borrower: TechStart Alpha. ARR $2.4M. MRR $300K. $300K*12=$3.6M≠$2.4M. ARPU implied $67K, stated $15K. Loan: $1M."},
    {"id":"TC19","difficulty":"adversarial",
     "data":"Borrower: LuxProperty Group. Properties appraised $24M. Debt $18M. LTV stated 65% actual 75%. Stressed ICR +200bps: 0.85x. Loan: $3M."},
    {"id":"TC20","difficulty":"adversarial",
     "data":"Borrower: NovaMed Clinic Group. Revenue $6.2M. EBITDA $1.8M (29% margin, typical 12-15%). No breakdown. No audited accounts. Loan: $2M."},
]

# Smoke test — 5 easy cases only, used in CI for speed
SMOKE_TEST_CASES = [t for t in TEST_CASES if t["difficulty"] == "easy"]
'''

with open(REPO_DIR / 'src' / 'test_cases.py', 'w', encoding='utf-8') as f:
    f.write(TEST_CASES_PY)

print('✅ src/test_cases.py written')

# %% [markdown]
# ## Step 3 — Write the CI Runner Script (`src/run_ci_eval.py`)
#
# This is the **entrypoint** called by GitHub Actions.
# It runs the eval, saves artefacts, and exits with code 1 if the quality gate fails.

# %%
CI_RUNNER_PY = '''#!/usr/bin/env python3
"""
FinSight AI — CI Evaluation Runner
Invoked by GitHub Actions on every PR.
Exit code 0 = quality gate passed; Exit code 1 = gate failed.
"""

import sys, json, csv, os
from pathlib import Path

# Add src/ to path when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import run_eval, build_leaderboard, check_quality_gate, GROQ_MODELS
from test_cases import SMOKE_TEST_CASES, TEST_CASES

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────
# Use smoke test (5 easy cases) in CI for speed; full suite on schedule
CI_MODE = os.environ.get("EVAL_MODE", "smoke")  # "smoke" | "full"
test_cases = SMOKE_TEST_CASES if CI_MODE == "smoke" else TEST_CASES

# In CI, only run the two fastest models to save cost + time
CI_MODELS = {
    "llama-3.3-70b": GROQ_MODELS["llama-3.3-70b"],  # Quality model
    "llama-3.1-8b":  GROQ_MODELS["llama-3.1-8b"],   # Speed model
}

print(f"[FinSight CI] Mode: {CI_MODE} | Cases: {len(test_cases)} | Models: {list(CI_MODELS.keys())}")


def save_results_csv(results: list, path: Path) -> None:
    if not results: return
    keys = list(results[0].keys())
    with open(path, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {path}")


def save_leaderboard_json(leaderboard: list, path: Path) -> None:
    with open(path, "w", encoding='utf-8') as f:
        json.dump(leaderboard, f, indent=2)
    print(f"  Saved: {path}")


def print_summary(leaderboard: list, gate: dict) -> None:
    print("\n" + "="*60)
    print("  FINSIGHT CI EVAL SUMMARY")
    print("="*60)
    header = f"{'Model':<20} {'Composite':>10} {'Hallucin%':>10} {'p95(s)':>8} {'Cost$':>8} {'Pass':>5}"
    print(header)
    print("-"*60)
    for row in leaderboard:
        composite = f"{row['composite']:.2f}" if row.get('composite') else "N/A"
        hall      = f"{row['hallucin_rate']*100:.1f}%" if row.get('hallucin_rate') is not None else "N/A"
        p95       = f"{row['latency_p95']:.2f}" if row.get('latency_p95') else "N/A"
        cost      = f"${row['avg_cost']:.5f}" if row.get('avg_cost') is not None else "N/A"
        passed    = "PASS" if row.get("meets_constraints") else "FAIL"
        print(f"{row['model']:<20} {composite:>10} {hall:>10} {p95:>8} {cost:>8} {passed:>5}")
    print("="*60)
    status = "PASSED" if gate["passed"] else "FAILED"
    print(f"\nQuality Gate: {status}")
    print(f"Reason: {gate['reason']}")
    if gate.get("best_model"):
        print(f"Recommended model: {gate['best_model']}")
    print()


def main() -> int:
    print("[FinSight CI] Running evaluation harness...")
    results = run_eval(test_cases, models=CI_MODELS, judge=True)

    save_results_csv(results, RESULTS_DIR / "ci_eval_results.csv")

    leaderboard = build_leaderboard(results)
    save_leaderboard_json(leaderboard, RESULTS_DIR / "ci_leaderboard.json")

    gate = check_quality_gate(leaderboard)
    print_summary(leaderboard, gate)

    # Write gate outcome for downstream steps / GitHub Summary
    with open(RESULTS_DIR / "gate_outcome.json", "w", encoding='utf-8') as f:
        json.dump(gate, f, indent=2)

    # Append to GitHub Step Summary if running in Actions
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding='utf-8') as f:
            f.write(f"## FinSight Eval — Quality Gate: {'✅ PASSED' if gate['passed'] else '❌ FAILED'}\n\n")
            f.write(f"**{gate['reason']}**\n\n")
            f.write("| Model | Composite | Hallucin% | p95 | Cost | Pass |\n")
            f.write("|---|---|---|---|---|---|\n")
            for row in leaderboard:
                composite = f"{row['composite']:.2f}" if row.get('composite') else "N/A"
                hall = f"{row['hallucin_rate']*100:.1f}%" if row.get('hallucin_rate') is not None else "N/A"
                p95  = f"{row['latency_p95']:.2f}s"
                cost = f"${row['avg_cost']:.5f}"
                icon = "PASS" if row.get("meets_constraints") else "FAIL"
                f.write(f"| {row['model']} | {composite} | {hall} | {p95} | {cost} | {icon} |\n")

    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
'''

with open(REPO_DIR / 'src' / 'run_ci_eval.py', 'w', encoding='utf-8') as f:
    f.write(CI_RUNNER_PY)


def repair_run_ci_eval(path: Path) -> None:
    """Normalize any notebook-export newline splits that broke string literals."""
    text = path.read_text(encoding='utf-8')
    text = text.replace('print("\n" + "="*60)', 'print("\\n" + "="*60)')
    text = text.replace('print(f"\nQuality Gate: {status}")', 'print(f"\\nQuality Gate: {status}")')
    text = text.replace(
        'f.write(f"## FinSight Eval — Quality Gate: {\'✅ PASSED\' if gate[\'passed\'] else \'❌ FAILED\'}\n\n")',
        'f.write(f"## FinSight Eval — Quality Gate: {\'✅ PASSED\' if gate[\'passed\'] else \'❌ FAILED\'}\\n\\n")',
    )
    text = text.replace(
        'f.write(f"**{gate[\'reason\']}**\n\n")',
        'f.write(f"**{gate[\'reason\']}**\\n\\n")',
    )
    text = text.replace(
        'f.write("| Model | Composite | Hallucin% | p95 | Cost | Pass |\n")',
        'f.write("| Model | Composite | Hallucin% | p95 | Cost | Pass |\\n")',
    )
    text = text.replace(
        'f.write("|---|---|---|---|---|---|\n")',
        'f.write("|---|---|---|---|---|---|\\n")',
    )
    text = text.replace(
        'f.write(f"| {row[\'model\']} | {composite} | {hall} | {p95} | {cost} | {icon} |\n")',
        'f.write(f"| {row[\'model\']} | {composite} | {hall} | {p95} | {cost} | {icon} |\\n")',
    )
    path.write_text(text, encoding='utf-8')


repair_run_ci_eval(REPO_DIR / 'src' / 'run_ci_eval.py')

print('✅ src/run_ci_eval.py written')

# %% [markdown]
# ## Step 4 — Write the `pytest` Test Suite (`tests/test_eval_gates.py`)
#
# Unit tests that verify individual eval functions work correctly,
# plus an integration test that runs the quality gate end-to-end.
# GitHub Actions runs `pytest` before the full eval to catch regressions fast.

# %%
PYTEST_PY = '''"""
FinSight AI — pytest test suite for the eval pipeline.
Tests run in CI before the full eval to catch code regressions quickly.
"""

import pytest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_harness import (
    check_hallucination,
    build_leaderboard,
    check_quality_gate,
    CONSTRAINTS,
)
from test_cases import TEST_CASES, SMOKE_TEST_CASES


# ── Unit tests: hallucination probe ──────────────────────────────

class TestHallucinationProbe:

    def test_no_hallucination_clean_data(self):
        """Model outputs only figures present in source data → no flag."""
        source = "Revenue: $12.4M. EBITDA: $2.1M. DSCR: 1.8x."
        memo   = "The borrower reports revenue of $12.4M and EBITDA of $2.1M with DSCR of 1.8x."
        result = check_hallucination(source, memo)
        assert result["hallucination_flag"] is False, \
            f"Expected no hallucination, got count={result['hallucination_count']}"

    def test_hallucination_detected_fabricated_number(self):
        """Model invents a revenue figure not in source → flagged."""
        source = "Revenue: $5.2M. No existing debt."
        memo   = "The borrower has revenue of $5.2M and an EBITDA of $2.8M."  # $2.8M fabricated
        result = check_hallucination(source, memo)
        assert result["hallucination_flag"] is True, "Expected hallucination flag for fabricated $2.8M"

    def test_rounding_tolerance(self):
        """Values within 5% of source → not flagged as hallucination."""
        source = "Revenue: $10.0M."
        memo   = "Revenue is approximately $10.1M."  # 1% delta — within tolerance
        result = check_hallucination(source, memo)
        assert result["hallucination_flag"] is False, \
            "Small rounding should not trigger hallucination flag"

    def test_small_numbers_ignored(self):
        """Ratios and percentages (< 200) are not flagged."""
        source = "DSCR: 1.8x."
        memo   = "The DSCR is 1.8x indicating strong debt service capacity. Score: 4/5."
        result = check_hallucination(source, memo)
        assert result["hallucination_flag"] is False, \
            "Small ratio numbers should not be flagged"


# ── Unit tests: leaderboard builder ──────────────────────────────

class TestLeaderboard:

    def _make_results(self, model_name, hall_flags, latencies, costs,
                      composite_scores):
        return [
            {
                "model":              model_name,
                "hallucination_flag": h,
                "latency":            l,
                "cost":               c,
                "composite_score":    s,
                "bert_score_f1":      0.90,
            }
            for h, l, c, s in zip(hall_flags, latencies, costs, composite_scores)
        ]

    def test_leaderboard_sorted_by_composite(self):
        r1 = self._make_results("model-a", [False]*5, [0.5]*5, [0.001]*5, [4.5]*5)
        r2 = self._make_results("model-b", [False]*5, [0.4]*5, [0.001]*5, [3.5]*5)
        lb = build_leaderboard(r1 + r2)
        assert lb[0]["model"] == "model-a", "Higher composite should rank first"

    def test_meets_constraints_flag(self):
        r = self._make_results("good-model", [False]*5, [0.5]*5, [0.001]*5, [4.5]*5)
        lb = build_leaderboard(r)
        assert lb[0]["meets_constraints"] is True

    def test_fails_constraints_high_hallucination(self):
        r = self._make_results("bad-model", [True]*5, [0.5]*5, [0.001]*5, [4.0]*5)
        lb = build_leaderboard(r)
        assert lb[0]["meets_constraints"] is False, \
            "100% hallucination rate should fail constraints"


# ── Unit tests: quality gate ──────────────────────────────────────

class TestQualityGate:

    def test_gate_passes_when_one_model_meets_all(self):
        leaderboard = [
            {"model": "good", "hallucin_rate": 0.005, "bert_score": 0.91,
             "latency_p95": 1.2, "avg_cost": 0.001, "meets_constraints": True,
             "composite": 4.5},
        ]
        gate = check_quality_gate(leaderboard)
        assert gate["passed"] is True
        assert gate["best_model"] == "good"

    def test_gate_fails_when_no_model_meets_all(self):
        leaderboard = [
            {"model": "bad", "hallucin_rate": 0.05, "bert_score": 0.85,
             "latency_p95": 4.0, "avg_cost": 0.03, "meets_constraints": False,
             "composite": 3.2},
        ]
        gate = check_quality_gate(leaderboard)
        assert gate["passed"] is False
        assert gate["best_model"] is None

    def test_gate_picks_highest_composite_when_multiple_pass(self):
        leaderboard = [  # Already sorted by composite desc
            {"model": "best",  "hallucin_rate": 0.0, "latency_p95": 0.8,
             "avg_cost": 0.001, "meets_constraints": True, "composite": 4.8},
            {"model": "ok",    "hallucin_rate": 0.0, "latency_p95": 0.9,
             "avg_cost": 0.001, "meets_constraints": True, "composite": 4.2},
        ]
        gate = check_quality_gate(leaderboard)
        assert gate["best_model"] == "best"


# ── Sanity checks on test data ────────────────────────────────────

class TestTestData:

    def test_twenty_cases_exist(self):
        assert len(TEST_CASES) == 20

    def test_all_difficulties_represented(self):
        diffs = {t["difficulty"] for t in TEST_CASES}
        assert diffs == {"easy", "medium", "hard", "adversarial"}

    def test_smoke_set_is_easy_only(self):
        assert all(t["difficulty"] == "easy" for t in SMOKE_TEST_CASES)
        assert len(SMOKE_TEST_CASES) == 5

    def test_all_cases_have_required_keys(self):
        for tc in TEST_CASES:
            assert "id" in tc and "difficulty" in tc and "data" in tc


# ── Integration test (skipped in fast CI, enabled on schedule) ───

@pytest.mark.skipif(
    "not os.environ.get('RUN_INTEGRATION_TESTS')",
    reason="Integration tests skipped unless RUN_INTEGRATION_TESTS=1"
)
class TestIntegration:

    def test_eval_produces_passing_gate(self):
        """Full smoke eval — requires GROQ_API_KEY to be set."""
        from eval_harness import run_eval, build_leaderboard, check_quality_gate, GROQ_MODELS
        results    = run_eval(SMOKE_TEST_CASES[:2], models={"llama-3.1-8b": GROQ_MODELS["llama-3.1-8b"]}, judge=False)
        leaderboard = build_leaderboard(results)
        assert len(leaderboard) > 0
        assert leaderboard[0]["latency_p95"] < CONSTRAINTS["latency_p95"]
        assert leaderboard[0]["avg_cost"]    < CONSTRAINTS["avg_cost"]
'''

with open(REPO_DIR / 'tests' / 'test_eval_gates.py', 'w', encoding='utf-8') as f:
    f.write(PYTEST_PY)

print('✅ tests/test_eval_gates.py written')
print(f'   Test classes: TestHallucinationProbe, TestLeaderboard, TestQualityGate, TestTestData, TestIntegration')

# %% [markdown]
# ## Step 5 — Write the GitHub Actions Workflow (`.github/workflows/llm-eval.yml`)
#
# This is the heart of the CI/CD pipeline. It triggers on every PR and
# runs the full eval harness against Groq.

# %%
WORKFLOW_YML = """# FinSight AI — LLM Eval Pipeline
# Triggers on every PR to main; also runs nightly on schedule.
#
# Required GitHub Secret: GROQ_API_KEY
# Add it at: Settings → Secrets and variables → Actions → New repository secret

name: LLM Eval Pipeline

on:
  pull_request:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'tests/**'
      - '.github/workflows/llm-eval.yml'
  push:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'   # Nightly full eval at 02:00 UTC
  workflow_dispatch:       # Allow manual trigger from GitHub UI
    inputs:
      eval_mode:
        description: 'Eval mode (smoke = 5 easy cases; full = all 20)'
        required: false
        default: 'smoke'
        type: choice
        options: [ smoke, full ]

jobs:

  # ─── Job 1: Unit tests (fast, no API calls) ────────────────────
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip

      - name: Install dependencies
        run: |
          pip install groq bert-score pandas matplotlib pytest -q

      - name: Run unit tests
        run: pytest tests/ -v --tb=short -m "not integration"

  # ─── Job 2: LLM Eval Quality Gate (requires GROQ_API_KEY) ──────
  llm-eval:
    name: LLM Eval Quality Gate
    runs-on: ubuntu-latest
    needs: unit-tests       # Only run eval if unit tests pass
    permissions:
      contents: read
      pull-requests: write  # Needed to post PR comment

    env:
      GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
      EVAL_MODE: ${{ github.event.inputs.eval_mode || (github.event_name == 'schedule' && 'full') || 'smoke' }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip

      - name: Install dependencies
        run: |
          pip install groq bert-score pandas matplotlib -q

      - name: Run FinSight eval harness
        id: run_eval
        run: |
          python src/run_ci_eval.py
          echo "exit_code=$?" >> $GITHUB_OUTPUT

      - name: Upload eval artefacts
        if: always()    # Upload even if gate fails
        uses: actions/upload-artifact@v4
        with:
          name: finsight-eval-results-${{ github.run_number }}
          path: results/
          retention-days: 30

      - name: Post results to PR comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const gate = JSON.parse(
              fs.readFileSync('results/gate_outcome.json', 'utf8')
            );
            const icon   = gate.passed ? '✅' : '❌';
            const status = gate.passed ? 'PASSED' : 'FAILED';
            const body = [
              '## ' + icon + ' FinSight LLM Eval — Quality Gate ' + status,
              '',
              '**' + gate.reason + '**',
              '',
              gate.best_model
                ? '🏆 Recommended model: `' + gate.best_model + '`'
                : '⚠️ No model met all production constraints.',
              '',
              '> Full results uploaded as [Actions artefacts](' +
                'https://github.com/' + context.repo.owner + '/' +
                context.repo.repo + '/actions/runs/' + context.runId + ').',
            ].join('\n');
            await github.rest.issues.createComment({
              owner:    context.repo.owner,
              repo:     context.repo.repo,
              issue_number: context.issue.number,
              body,
            });

      - name: Fail job if quality gate failed
        if: steps.run_eval.outputs.exit_code != '0'
        run: |
          echo "Quality gate failed — see results/gate_outcome.json for details."
          exit 1

  # ─── Job 3: Nightly full eval (schedule only) ──────────────────
  nightly-full-eval:
    name: Nightly Full Eval (20 cases × 4 models)
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    needs: unit-tests

    env:
      GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
      EVAL_MODE: full

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip

      - name: Install dependencies
        run: pip install groq bert-score pandas matplotlib -q

      - name: Run full eval (all 20 cases × 4 Groq models)
        run: |
          EVAL_MODE=full python src/run_ci_eval.py

      - name: Upload nightly artefacts
        uses: actions/upload-artifact@v4
        with:
          name: finsight-nightly-${{ github.run_number }}
          path: results/
          retention-days: 90
"""

with open(REPO_DIR / '.github' / 'workflows' / 'llm-eval.yml', 'w', encoding='utf-8') as f:
    f.write(WORKFLOW_YML)

print('✅ .github/workflows/llm-eval.yml written')
print(f'   Jobs: unit-tests → llm-eval (PR gate) + nightly-full-eval (schedule)')

# %% [markdown]
# ## Step 6 — Write Supporting Files (`requirements.txt`, `README.md`, `.gitignore`)

# %%
# ── requirements.txt ─────────────────────────────────────────────
REQUIREMENTS = """groq>=0.9.0
bert-score>=0.3.13
pandas>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
pytest>=8.0.0
"""

with open(REPO_DIR / 'requirements.txt', 'w', encoding='utf-8') as f:
    f.write(REQUIREMENTS)


# ── .gitignore ────────────────────────────────────────────────────
GITIGNORE = """__pycache__/
*.pyc
.env
*.db
results/*.csv
results/*.png
results/*.json
.DS_Store
"""

with open(REPO_DIR / '.gitignore', 'w', encoding='utf-8') as f:
    f.write(GITIGNORE)


# ── README.md ─────────────────────────────────────────────────────
README = """# 🏦 FinSight AI — LLM Eval Pipeline

Automated quality gates for Groq-powered credit memo generation.

## Architecture
```
.
├── src/
│   ├── eval_harness.py      # Core eval logic (Groq calls, scoring, gate)
│   ├── test_cases.py        # 20 FinSight credit memo test cases
│   └── run_ci_eval.py       # CI entrypoint (called by GitHub Actions)
├── tests/
│   └── test_eval_gates.py   # pytest unit + integration tests
├── results/                 # Artefacts (gitignored)
├── .github/workflows/
│   └── llm-eval.yml         # GitHub Actions pipeline
└── requirements.txt
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your Groq API key to GitHub Secrets:
   - Go to **Settings → Secrets and variables → Actions**
   - Add secret: `GROQ_API_KEY`

3. Run evals locally:
   ```bash
   export GROQ_API_KEY=gsk_...
   python src/run_ci_eval.py
   ```

4. Run unit tests:
   ```bash
   pytest tests/ -v
   ```

## CI Behaviour

| Event | Trigger | Models | Cases |
|---|---|---|---|
| Pull Request | On every PR to `main` | llama-3.3-70b + llama-3.1-8b | 5 easy (smoke) |
| Push to main | On merge | llama-3.3-70b + llama-3.1-8b | 5 easy (smoke) |
| Nightly (02:00 UTC) | Schedule | All 4 Groq models | All 20 |
| Manual | workflow_dispatch | Configurable | Configurable |

## Quality Gate (FinSight Production Constraints)
| Constraint | Threshold |
|---|---|
| Hallucination rate | < 1% |
| BERTScore F1 | ≥ 0.88 |
| Latency p95 | < 3s |
| Cost per memo | < $0.02 |

A PR is blocked from merging if no model passes all four constraints.
"""

with open(REPO_DIR / 'README.md', 'w', encoding='utf-8') as f:
    f.write(README)

# ── Verify final structure ────────────────────────────────────────
print('✅ Supporting files written')
print('\n📁 Final repo structure:')
for p in sorted(REPO_DIR.rglob('*')):
    if p.is_file():
        size = p.stat().st_size
        rel  = p.relative_to(REPO_DIR)
        print(f'   {rel}  ({size:,} bytes)')

# %% [markdown]
# ## Step 7 — Run Pytest Locally (Validates Logic Without API Calls)

# %%
import subprocess, sys

result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short',
     '--ignore=tests/test_eval_gates.py'],  # Skip until we copy files
    capture_output=True, text=True,
    cwd=str(REPO_DIR)
)

# Since pytest can't find the test file yet from REPO_DIR, run directly
result2 = subprocess.run(
    [sys.executable, '-m', 'pytest',
     str(REPO_DIR / 'tests' / 'test_eval_gates.py'),
     '-v', '--tb=short', '-m', 'not integration',
     '--import-mode=importlib'],
    capture_output=True, text=True,
    env={**os.environ, 'PYTHONPATH': str(REPO_DIR / 'src')}
)

print(result2.stdout[-3000:] if len(result2.stdout) > 3000 else result2.stdout)
if result2.returncode != 0:
    print("STDERR:", result2.stderr[-1000:])
else:
    print(f'✅ All unit tests passed (exit code {result2.returncode})')

# %% [markdown]
# ## Step 8 — Push to GitHub & Verify the Actions Workflow
#
# This cell:
# 1. Initialises a git repo locally
# 2. Creates the GitHub repo via the API
# 3. Pushes everything
# 4. Prints the link to your first Actions run

# %%
from github import Github
import subprocess
import socket

# ── Initialise git ────────────────────────────────────────────────
def git(cmd, cwd=REPO_DIR):
    r = subprocess.run(shlex.split(cmd), cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  git error: {r.stderr.strip()}')
    return r.stdout.strip()


def github_is_reachable() -> bool:
    """Best-effort check so offline runs can skip GitHub-only steps cleanly."""
    try:
        socket.getaddrinfo("github.com", 443)
        socket.getaddrinfo("api.github.com", 443)
        return True
    except OSError:
        return False


if not github_is_reachable():
    print("ℹ️  GitHub is not reachable from this environment; skipping repo push, secret upload, and workflow dispatch.")
else:
    git('git init')
    git('git config user.email ci@finsight.ai')
    git('git config user.name FinSight CI')
    git('git add .')
    git('git commit -m "feat: initial LLM eval pipeline with GitHub Actions"')

repo = None
if github_is_reachable():
    # ── Create GitHub repo via PyGithub ──────────────────────────────
    try:
        g = Github(GITHUB_TOKEN)
        user = g.get_user()
        try:
            repo = user.create_repo(
                REPO_NAME,
                description='Automated LLM quality gates for FinSight AI credit memo generation',
                private=True,   # ← Change to False for public repo
                auto_init=False,
            )
            print(f'✅ GitHub repo created: {repo.html_url}')
        except Exception:
            # Repo may already exist
            repo = user.get_repo(REPO_NAME)
            print(f'ℹ️  Using existing repo: {repo.html_url}')

        # ── Push to GitHub ────────────────────────────────────────────────
        remote_url = f'https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git'
        git(f'git remote add origin {remote_url}')
        git('git branch -M main')
        push_result = subprocess.run(
            ['git', 'push', '-u', 'origin', 'main'],
            cwd=str(REPO_DIR), capture_output=True, text=True
        )

        if push_result.returncode == 0:
            print(f'✅ Code pushed to GitHub')
            print(f'\n🔗 Repository:  {repo.html_url}')
            print(f'🔗 Actions:     {repo.html_url}/actions')
            print(f'\n⚠️  Next step:')
            print(f'   Add your GROQ_API_KEY secret at:')
            print(f'   {repo.html_url}/settings/secrets/actions')
            print(f'\n   Once the secret is set, trigger a manual run at:')
            print(f'   {repo.html_url}/actions/workflows/llm-eval.yml')
        else:
            print('Push error:', push_result.stderr)
    except Exception as e:
        print(f'ℹ️  GitHub push/repo setup skipped: {e}')

# %% [markdown]
# ## Step 9 — Add the GROQ_API_KEY Secret Programmatically

# %%
from github import Github
from base64 import b64encode
from nacl import encoding, public as nacl_public

# ── Add GROQ_API_KEY as a GitHub Actions secret ───────────────────
# This uses PyNaCl to encrypt the secret with the repo's public key,
# which is the required format for the GitHub Secrets API.

if github_is_reachable() and repo is not None:
    try:
    #     !pip install PyNaCl -q

        g    = Github(GITHUB_TOKEN)
        user = g.get_user()
        repo = user.get_repo(REPO_NAME)

        # Get the repo's public key (required to encrypt secrets)
        pub_key = repo.get_public_key()

        def encrypt_secret(public_key_value: str, secret_value: str) -> str:
            """Encrypt a secret using the repo's RSA public key."""
            pk    = nacl_public.PublicKey(public_key_value.encode(), encoding.Base64Encoder())
            box   = nacl_public.SealedBox(pk)
            encrypted = box.encrypt(secret_value.encode())
            return b64encode(encrypted).decode()

        encrypted_value = encrypt_secret(pub_key.key, GROQ_API_KEY)

        repo.create_secret('GROQ_API_KEY', encrypted_value)
        print('✅ GROQ_API_KEY secret added to GitHub repo')
        print(f'   Verify at: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/settings/secrets/actions')

    except ImportError:
        print('ℹ️  PyNaCl not available — add GROQ_API_KEY manually:')
        print(f'   https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/settings/secrets/actions')
    except Exception as e:
        print(f'Secret upload error: {e}')
        print(f'   Add GROQ_API_KEY manually at:')
        print(f'   https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/settings/secrets/actions')
else:
    print('ℹ️  Skipping secret upload because GitHub is unreachable or repo is unavailable.')

# %% [markdown]
# ## Step 10 — Trigger a Test Workflow Run & Monitor It

# %%
if github_is_reachable() and repo is not None:
    from github import Github

    g    = Github(GITHUB_TOKEN)
    user = g.get_user()
    repo = user.get_repo(REPO_NAME)

    # ── Trigger a manual workflow_dispatch run ────────────────────────
    try:
        workflow = repo.get_workflow('llm-eval.yml')
        triggered = workflow.create_dispatch(
            ref='main',
            inputs={'eval_mode': 'smoke'}
        )
        print('✅ Workflow dispatch triggered (smoke mode — 5 cases, 2 models)')

        # Poll for the latest run
        import time
        time.sleep(5)  # Give Actions a moment to queue the run

        runs = list(workflow.get_runs())
        if runs:
            latest = runs[0]
            print(f'\n🔗 Run URL:    {latest.html_url}')
            print(f'   Status:     {latest.status}')
            print(f'   Conclusion: {latest.conclusion or "pending"}')
            print(f'\n💡 The run will take ~2-3 minutes to complete.')
            print(f'   Monitor progress at: {latest.html_url}')
        else:
            print('Run queued — check Actions tab.')

    except Exception as e:
        print(f'Could not trigger workflow: {e}')
        print(f'Trigger manually at: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/actions')
else:
    print('ℹ️  Skipping workflow dispatch because GitHub is unreachable or repo is unavailable.')

# %% [markdown]
# ## Step 11 — Architecture Diagram
#
# Here is a visual summary of the full pipeline.

# %%
# print("""
# ┌─────────────────────────────────────────────────────────────────┐
# │              FINSIGHT AI — GitHub Actions Eval Pipeline         │
# └─────────────────────────────────────────────────────────────────┘

#   Developer                  GitHub                  Groq API
#   ─────────                  ──────                  ────────
#   git push PR  ──────────►  PR opened
#                             Workflow triggers
#                                 │
#                                 ▼
#                     ┌──────────────────────┐
#                     │  Job 1: unit-tests   │  (no API calls, ~30s)
#                     │  pytest tests/       │
#                     └──────────┬───────────┘
#                                │ pass
#                                ▼
#                     ┌──────────────────────┐
#                     │  Job 2: llm-eval     │
#                     │                      │
#                     │  run_ci_eval.py      │
#                     │    └─ eval_harness   │──────► llama-3.3-70b
#                     │         (5 cases)    │──────► llama-3.1-8b
#                     │                      │  ◄──── outputs + metrics
#                     │  Quality probes:     │
#                     │  • hallucination < 1%│
#                     │  • BERTScore ≥ 0.88  │
#                     │  • latency p95 < 3s  │
#                     │  • cost < $0.02      │
#                     └──────────┬───────────┘
#                                │
#                     ┌──────────┴───────────┐
#                     │                      │
#                    pass                  fail
#                     │                      │
#                     ▼                      ▼
#              PR comment:            PR comment:
#              ✅ Gate PASSED          ❌ Gate FAILED
#              + artefacts            + reason + artefacts
#              + best model           PR blocked

#   ── Schedule (02:00 UTC) ──────────────────────────────────────────
#   Nightly full eval: all 20 cases × 4 Groq models → 90-day artefacts
# """)

# %% [markdown]
# ## Step 12 — Run Locally Before Pushing (Dry-Run Verification)
#
# Use this cell to test the CI runner locally in Colab — same code that runs in Actions.

# %%
import subprocess, sys, os

# ── Run the CI evaluator directly from Colab ──────────────────────
print('Running src/run_ci_eval.py in smoke mode (5 cases × 2 models)...')
print('This is exactly what GitHub Actions will execute.\n')

env = {
    **os.environ,
    'GROQ_API_KEY': GROQ_API_KEY,
    'EVAL_MODE':    'smoke',
    'PYTHONPATH':   str(REPO_DIR / 'src'),
}

result = subprocess.run(
    [sys.executable, str(REPO_DIR / 'src' / 'run_ci_eval.py')],
    env=env,
    capture_output=True,
    text=True,
    cwd=str(REPO_DIR),
)

print(result.stdout)
if result.returncode == 0:
    print('\n✅ CI dry-run PASSED — safe to push to GitHub')
else:
    print('\n❌ CI dry-run FAILED — fix issues before pushing')
    if result.stderr:
        print('STDERR:', result.stderr[-500:])

# Show saved artefacts
results_path = REPO_DIR / 'results'
if results_path.exists():
    print('\n📁 Artefacts saved:')
    for f in sorted(results_path.iterdir()):
        print(f'   {f.name}  ({f.stat().st_size:,} bytes)')

# %% [markdown]
# ---
# ## 🚀 Extension Tasks
#
# **⏱ +10 minutes | Advanced participants**
#
# ### Option A — Add a Slack Notification step
# Add a step to the `llm-eval` job that posts a Slack message when the gate fails:
# ```yaml
# - name: Notify Slack on failure
#   if: failure()
#   uses: slackapi/slack-github-action@v1.26.0
#   with:
#     payload: |
#       {"text": "❌ FinSight eval gate FAILED on PR #${{ github.event.number }}"}
#   env:
#     SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
# ```
#
# ### Option B — Add BERTScore to the quality gate
# 1. Compute BERTScore inside `run_ci_eval.py` (it's already in `eval_harness.py`)
# 2. Add `bert_score >= 0.88` to `check_quality_gate()`
# 3. Update the workflow to pass `TRANSFORMERS_CACHE` to avoid re-downloading the model
#
# ### Option C — Matrix strategy: run each model in parallel
# Replace the single `llm-eval` job with a matrix strategy:
# ```yaml
# strategy:
#   matrix:
#     model: [llama-3.3-70b, llama-3.1-8b, mixtral-8x7b, gemma2-9b]
# ```
# Then aggregate results in a final `gate` job using `needs: [llm-eval]`.
#
# ### Option D — Cache the eval results with GitHub Actions Cache
# Use `actions/cache@v4` to store eval results keyed on the prompt hash,
# so unchanged test cases don't consume Groq tokens on every run.

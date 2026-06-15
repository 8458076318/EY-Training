"""Lab 1 - Build a Multi-Model LLM Evaluation Harness.

Converted from `colab1_evaluation_harness.ipynb` into a standalone Python script.
"""

# Lab overview:
# - Define structured evaluation dimensions with explicit rubrics
# - Build a test harness that calls multiple LLM APIs
# - Score outputs using BERTScore and a GPT-4o-as-judge rubric
# - Produce a ranked model leaderboard with visualisation
# - Inspect sample outputs and export results

# Step 0 - Install Dependencies
# In Colab you can run:
#   !pip install openai anthropic google-generativeai bert-score pandas matplotlib seaborn tqdm -q
# For local execution, install the dependencies through your environment manager.

import json
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from bert_score import score as bert_score
from tqdm.auto import tqdm

try:
    from google.colab import userdata  # type: ignore
except Exception:
    userdata = None

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise ImportError(
        "openai is required to run this script. Install the notebook dependencies first."
    ) from exc

try:
    import anthropic
except Exception as exc:  # pragma: no cover - import-time dependency guard
    raise ImportError(
        "anthropic is required to run this script. Install the notebook dependencies first."
    ) from exc


# Step 1 - Configure API Keys & Model Clients
def load_env_file(path: Path) -> dict:
    """Load simple KEY=VALUE pairs from a .env file without extra dependencies."""
    values = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            values[key] = value

    return values


_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_ENV_VALUES = load_env_file(_ENV_PATH)
for _key, _value in _ENV_VALUES.items():
    os.environ.setdefault(_key, _value)


def _read_secret(name: str, fallback: str) -> str:
    if userdata is not None:
        try:
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass
    return os.getenv(name, fallback)


OPENAI_API_KEY = _read_secret("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = _read_secret("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = _read_secret("GOOGLE_API_KEY", "")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

if not OPENAI_API_KEY:
    raise RuntimeError(f"OPENAI_API_KEY not found. Set it in {_ENV_PATH} or export it in the environment.")

if not ANTHROPIC_API_KEY:
    raise RuntimeError(f"ANTHROPIC_API_KEY not found. Set it in {_ENV_PATH} or export it in the environment.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

print("Dependencies and API clients ready")


# Step 2 - Define Evaluation Dimensions & Rubric
EVAL_RUBRIC = """
You are a senior credit risk officer evaluating AI-generated credit memos.
Score the following memo on THREE dimensions (1-5 scale each).

DIMENSION 1 - FAITHFULNESS (1-5)
Does the memo accurately reflect the financial data provided?
Are all stated figures, ratios, and dates correct?
5 = all facts accurate; 1 = multiple fabricated or wrong facts

DIMENSION 2 - COMPLETENESS (1-5)
Does the memo cover: borrower profile, key financial ratios, risk flags, recommendation?
5 = comprehensive; 1 = major sections missing

DIMENSION 3 - REGULATORY TONE (1-5)
Is the language precise, objective, and appropriate for a regulated lending environment?
Are hedges and uncertainties appropriately flagged?
5 = professional and compliance-ready; 1 = vague, promotional, or inappropriate

BORROWER DATA:
{borrower_data}

GENERATED MEMO:
{memo}

Respond ONLY with valid JSON:
{
  "faithfulness": <int 1-5>,
  "completeness": <int 1-5>,
  "regulatory_tone": <int 1-5>,
  "reasoning": "<one-sentence justification>"
}
"""

SYSTEM_PROMPT = """You are a credit analyst AI assistant at FinSight AI.
Generate a concise credit risk memo (150-250 words) based on the provided borrower data.
Structure: (1) Borrower Overview, (2) Key Financial Metrics, (3) Risk Assessment, (4) Recommendation.
Use precise financial language. Do not fabricate or extrapolate data not provided."""

print("Rubric and system prompt defined")


# Step 3 - Load the FinSight Test Set (20 Credit Memo Prompts)
TEST_CASES = [
    {"id": "TC01", "difficulty": "easy", "data": "Borrower: Apex Manufacturing Ltd. Revenue: $12.4M. EBITDA: $2.1M. Debt: $4.5M. DSCR: 1.8x. Industry: Industrial. Loan request: $1.5M term loan, 5yr. No payment history issues."},
    {"id": "TC02", "difficulty": "easy", "data": "Borrower: GreenLeaf Organics Inc. Revenue: $5.2M. Net Profit: $420K. Current ratio: 2.1. No existing debt. Loan request: $800K equipment financing. 3 years in operation."},
    {"id": "TC03", "difficulty": "easy", "data": "Borrower: Sunrise Hotels Group. Revenue: $18.7M. EBITDA: $3.9M. Existing debt: $7.2M. LTV on collateral: 62%. Loan request: $2M expansion loan. Strong cash reserves."},
    {"id": "TC04", "difficulty": "easy", "data": "Borrower: TechBridge Solutions LLC. SaaS recurring revenue: $3.1M ARR. MRR growth: 12% QoQ. Churn: 2.1%. No debt. Loan request: $500K working capital. 2yr operating history."},
    {"id": "TC05", "difficulty": "easy", "data": "Borrower: Coastal Fisheries Co. Revenue: $9.8M. EBITDA: $1.4M. DSCR: 1.6x. Collateral: fleet valued $4.2M. Loan request: $1.2M vessel upgrade. Seasonal revenue pattern."},
    {"id": "TC06", "difficulty": "medium", "data": "Borrower: RetailPro Chain. Revenue: $22M (declining 8% YoY). EBITDA: $1.1M. High lease obligations: $4.8M/yr. Debt: $9.5M. DSCR: 1.05x. Loan request: $3M refinancing. Online channel growing 35%."},
    {"id": "TC07", "difficulty": "medium", "data": "Borrower: NovaBio Pharma. Pre-revenue. Burn rate: $200K/mo. Runway: 14 months. Patent portfolio: 3 pending. Loan request: $1.5M bridge. VC backing: $4M raised. FDA trial Phase 2."},
    {"id": "TC08", "difficulty": "medium", "data": "Borrower: Atlas Construction. Revenue: $31M. Gross margin: 8%. Two large contracts: $15M total. DSCR: 1.3x. Outstanding litigation: $500K dispute (unresolved). Loan request: $4M contract financing."},
    {"id": "TC09", "difficulty": "medium", "data": "Borrower: PrimeAgri Partners. Revenue: $7.4M. EBITDA: $900K. Collateral: farmland $3.5M. Weather risk: drought region. Crop insurance in place. Loan request: $1.8M seasonal operating line."},
    {"id": "TC10", "difficulty": "medium", "data": "Borrower: Urban Mobility Startups Inc. Revenue: $1.2M. Gross burn: $350K/mo. Negative EBITDA. City contract secured: $5M over 3yr. Personal guarantee from founder (net worth $2.1M). Loan request: $600K."},
    {"id": "TC11", "difficulty": "hard", "data": "Borrower: GlobalTrade Import/Export. Revenue: $45M (70% from single customer). DSCR: 2.1x. However, primary customer announced restructuring last quarter. No formal contract renewal. Loan request: $6M trade finance."},
    {"id": "TC12", "difficulty": "hard", "data": "Borrower: DataVault Systems. Revenue: $8M. EBITDA reported: $2.4M. BUT: large related-party transaction of $1.8M detected in financials. Adjusted EBITDA: $600K. DSCR adjusted: 0.9x. Loan request: $2.5M."},
    {"id": "TC13", "difficulty": "hard", "data": "Borrower: Heritage Real Estate Fund. NAV: $28M. Leverage: 3.2x. LTV: 71%. Interest coverage: 1.1x. Two assets under negative watch. Market cap rate expanding 50bps since appraisal. Loan request: $8M refinance."},
    {"id": "TC14", "difficulty": "hard", "data": "Borrower: CryptoAsset Ventures. Revenue: $3.1M (all from crypto trading). Highly volatile. No audited financials. Self-reported balance sheet. Assets: $4.2M in digital assets (unencumbered). Loan request: $1M."},
    {"id": "TC15", "difficulty": "hard", "data": "Borrower: MedDevice International. Revenue: $14M. Profitable. BUT: operating in 3 jurisdictions with pending regulatory review. Key product (60% of revenue) faces potential recall. Loan request: $3.5M expansion."},
    {"id": "TC16", "difficulty": "adversarial", "data": "Borrower: Pinnacle Energy. Revenue: $19M. Net income: $3.2M. Tax rate: 25%. BUT: reported taxes paid: $250K (inconsistent with $800K expected). EBITDA stated as $5.4M but D&A not disclosed. Loan request: $4M."},
    {"id": "TC17", "difficulty": "adversarial", "data": "Borrower: FreshFoods Co. Revenue: $11M Q1 annualised. But Q1 is high season (35% of annual). Actual annual revenue estimate: ~$7M. Borrower submitted Q1 as representative. Loan request: $2.5M based on Q1 figures."},
    {"id": "TC18", "difficulty": "adversarial", "data": "Borrower: TechStart Alpha. ARR: $2.4M. MRR: $300K. BUT: $300K x 12 = $3.6M not $2.4M - figures are inconsistent. Logo count: 45 customers. ARPU implied: ~$67K but stated as $15K. Loan request: $1M."},
    {"id": "TC19", "difficulty": "adversarial", "data": "Borrower: LuxProperty Group. Properties: 12. Total appraised value: $24M. Debt: $18M. LTV: stated as 65% but actual: 75% based on above. Interest rate: variable. Current coverage: 1.4x. Stressed coverage at +200bps: 0.85x. Loan request: $3M."},
    {"id": "TC20", "difficulty": "adversarial", "data": "Borrower: NovaMed Clinic Group. Revenue: $6.2M. EBITDA: $1.8M (29% margin - unusually high for clinic ops, typical is 12-15%). No breakdown provided. Loan request: $2M expansion. Audited accounts not available."},
]

print(f"Loaded {len(TEST_CASES)} test cases")
print(
    "   Easy: "
    f"{sum(1 for t in TEST_CASES if t['difficulty'] == 'easy')} | "
    f"Medium: {sum(1 for t in TEST_CASES if t['difficulty'] == 'medium')} | "
    f"Hard: {sum(1 for t in TEST_CASES if t['difficulty'] == 'hard')} | "
    f"Adversarial: {sum(1 for t in TEST_CASES if t['difficulty'] == 'adversarial')}"
)


# Step 4 - Call Multiple LLM APIs & Collect Outputs
def call_openai(prompt: str, model: str = "gpt-4o") -> dict:
    start = time.time()
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        latency = time.time() - start
        output = resp.choices[0].message.content
        tokens = resp.usage.total_tokens
        cost = (resp.usage.prompt_tokens * 5 + resp.usage.completion_tokens * 15) / 1_000_000
        return {"output": output, "tokens": tokens, "latency": latency, "cost": cost, "error": None}
    except Exception as exc:
        return {"output": "", "tokens": 0, "latency": time.time() - start, "cost": 0, "error": str(exc)}


def call_anthropic(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> dict:
    start = time.time()
    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - start
        output = resp.content[0].text
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        cost = (resp.usage.input_tokens * 3 + resp.usage.output_tokens * 15) / 1_000_000
        return {"output": output, "tokens": tokens, "latency": latency, "cost": cost, "error": None}
    except Exception as exc:
        return {"output": "", "tokens": 0, "latency": time.time() - start, "cost": 0, "error": str(exc)}


MODELS = {
    "gpt-4o": lambda p: call_openai(p, "gpt-4o"),
    "claude-3-5-sonnet": lambda p: call_anthropic(p, "claude-3-5-sonnet-20241022"),
    # "gpt-4o-mini": lambda p: call_openai(p, "gpt-4o-mini"),
}

SUBSET = TEST_CASES[:5]  # Change to TEST_CASES to run all 20
results = []

for tc in tqdm(SUBSET, desc="Test cases"):
    for model_name, caller in MODELS.items():
        prompt = f"Generate a credit risk memo for the following borrower:\n\n{tc['data']}"
        result = caller(prompt)
        results.append(
            {
                "test_id": tc["id"],
                "difficulty": tc["difficulty"],
                "model": model_name,
                "output": result["output"],
                "tokens": result["tokens"],
                "latency": round(result["latency"], 2),
                "cost": round(result["cost"], 5),
                "error": result["error"],
                "borrower_data": tc["data"],
            }
        )
        time.sleep(0.3)

df = pd.DataFrame(results)
print(f"\nCollected {len(df)} outputs")
print(df[["test_id", "model", "latency", "cost", "error"]].to_string(index=False))


# Step 5 - Score with BERTScore
reference_map = {}
for _, row in df[df["model"] == "gpt-4o"].iterrows():
    reference_map[row["test_id"]] = row["output"]

for model_name in df["model"].unique():
    model_df = df[df["model"] == model_name].copy()
    candidates = model_df["output"].tolist()
    references = [reference_map.get(tid, "") for tid in model_df["test_id"].tolist()]

    if candidates and all(c for c in candidates):
        _, _, f1 = bert_score(
            candidates,
            references,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        for i, idx in enumerate(model_df.index):
            df.at[idx, "bert_score_f1"] = round(f1[i].item(), 4)

print("BERTScore computed")
print(df.groupby("model")["bert_score_f1"].mean().round(4))


# Step 6 - GPT-4o-as-Judge Rubric Scoring
def judge_output(borrower_data: str, memo: str) -> dict:
    """Use GPT-4o to score a memo on the FinSight rubric."""
    prompt = EVAL_RUBRIC.format(borrower_data=borrower_data, memo=memo)
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception as exc:
        return {
            "faithfulness": None,
            "completeness": None,
            "regulatory_tone": None,
            "reasoning": f"ERROR: {exc}",
        }


print("Running judge scoring... (this may take a while)")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
    if row["output"] and not row["error"]:
        scores = judge_output(row["borrower_data"], row["output"])
        df.at[idx, "score_faithfulness"] = scores.get("faithfulness")
        df.at[idx, "score_completeness"] = scores.get("completeness")
        df.at[idx, "score_regulatory_tone"] = scores.get("regulatory_tone")
        df.at[idx, "judge_reasoning"] = scores.get("reasoning")
    time.sleep(0.2)

df["composite_score"] = df[
    ["score_faithfulness", "score_completeness", "score_regulatory_tone"]
].mean(axis=1).round(2)

print("\nJudge scoring complete")
print(
    df.groupby("model")[
        ["score_faithfulness", "score_completeness", "score_regulatory_tone", "composite_score"]
    ].mean().round(2)
)


# Step 7 - Hallucination Probe
def check_hallucination(borrower_data: str, memo: str) -> dict:
    """Simple hallucination probe based on numeric values."""
    source_nums = set(re.findall(r"\$?\d+\.?\d*[KMB]?", borrower_data))

    def normalise(s: str):
        s = s.replace("$", "").replace(",", "")
        if s.endswith("K"):
            return float(s[:-1]) * 1_000
        if s.endswith("M"):
            return float(s[:-1]) * 1_000_000
        if s.endswith("B"):
            return float(s[:-1]) * 1_000_000_000
        try:
            return float(s)
        except Exception:
            return None

    source_vals = {normalise(n) for n in source_nums if normalise(n) is not None}
    memo_nums = re.findall(r"\$?\d+\.?\d*[KMB]?", memo)
    memo_vals = {normalise(n) for n in memo_nums if normalise(n) is not None and normalise(n) > 0}

    hallucinated = []
    for mv in memo_vals:
        if not any(abs(mv - sv) / max(sv, 0.01) < 0.05 for sv in source_vals if sv > 0):
            hallucinated.append(mv)

    hallucinated = [v for v in hallucinated if v > 200 and v != mv]

    return {
        "hallucination_count": len(hallucinated),
        "hallucinated_values": hallucinated[:3],
        "hallucination_flag": len(hallucinated) > 0,
    }


for idx, row in df.iterrows():
    if row["output"]:
        probe = check_hallucination(row["borrower_data"], row["output"])
        df.at[idx, "hallucination_count"] = probe["hallucination_count"]
        df.at[idx, "hallucination_flag"] = probe["hallucination_flag"]

hall_rate = df.groupby("model")["hallucination_flag"].mean().round(3) * 100
print("Hallucination probe complete")
print("\nHallucination Rate (%)")
print(hall_rate)


# Step 8 - Build the Model Leaderboard
leaderboard = df.groupby("model").agg(
    bert_score=("bert_score_f1", "mean"),
    faithfulness=("score_faithfulness", "mean"),
    completeness=("score_completeness", "mean"),
    reg_tone=("score_regulatory_tone", "mean"),
    composite=("composite_score", "mean"),
    latency_p95=("latency", lambda x: x.quantile(0.95)),
    avg_cost=("cost", "mean"),
    hallucin_rate=("hallucination_flag", "mean"),
).round(3).reset_index()

leaderboard = leaderboard.sort_values("composite", ascending=False)

CONSTRAINTS = {
    "hallucin_rate": 0.01,
    "bert_score": 0.88,
    "latency_p95": 3.0,
    "avg_cost": 0.02,
}


def meets_all(row):
    return (
        row["hallucin_rate"] < CONSTRAINTS["hallucin_rate"]
        and row["bert_score"] >= CONSTRAINTS["bert_score"]
        and row["latency_p95"] < CONSTRAINTS["latency_p95"]
        and row["avg_cost"] < CONSTRAINTS["avg_cost"]
    )


leaderboard["meets_constraints"] = leaderboard.apply(meets_all, axis=1)

print("=" * 65)
print("           FINSIGHT MODEL LEADERBOARD")
print("=" * 65)
print(
    leaderboard[
        ["model", "composite", "bert_score", "hallucin_rate", "latency_p95", "avg_cost", "meets_constraints"]
    ].to_string(index=False)
)
print("\n= meets all FinSight production constraints")


# Step 8 continued - Visualisation
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("FinSight Model Evaluation - Results", fontsize=15, fontweight="bold")

palette = ["#0D9488" if m else "#F97316" for m in leaderboard["meets_constraints"]]

axes[0].bar(leaderboard["model"], leaderboard["composite"], color=palette)
axes[0].axhline(y=4.0, color="gray", linestyle="--", alpha=0.5, label="Target >= 4.0")
axes[0].set_title("Composite Quality Score (1-5)")
axes[0].set_ylim(0, 5)
axes[0].legend()

axes[1].bar(leaderboard["model"], leaderboard["latency_p95"], color=palette)
axes[1].axhline(y=3.0, color="red", linestyle="--", alpha=0.7, label="Constraint: 3s")
axes[1].set_title("Latency p95 (seconds)")
axes[1].legend()

axes[2].bar(leaderboard["model"], leaderboard["avg_cost"], color=palette)
axes[2].axhline(y=0.02, color="red", linestyle="--", alpha=0.7, label="Constraint: $0.02")
axes[2].set_title("Average Cost per Memo ($)")
axes[2].legend()

green_patch = mpatches.Patch(color="#0D9488", label="Meets all constraints")
orange_patch = mpatches.Patch(color="#F97316", label="Fails >=1 constraint")
fig.legend(handles=[green_patch, orange_patch], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))

for ax in axes:
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("finsight_leaderboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to finsight_leaderboard.png")


# Step 9 - Inspect Individual Outputs (Qualitative Review)
INSPECT_TC = "TC16"  # Change to any test case ID; TC16 is adversarial

print(f"\n{'=' * 70}")
print(f"TEST CASE: {INSPECT_TC} - Side-by-Side Comparison")
print(f"{'=' * 70}")

tc_data = next(t for t in TEST_CASES if t["id"] == INSPECT_TC)
print(f"\nBORROWER DATA:\n{tc_data['data']}\n")

for _, row in df[df["test_id"] == INSPECT_TC].iterrows():
    print(f"\n--- MODEL: {row['model']} ---")
    print(f"Output:\n{row['output']}")
    print(
        f"\nScores: Faithfulness={row.get('score_faithfulness', 'N/A')} | "
        f"Completeness={row.get('score_completeness', 'N/A')} | "
        f"RegTone={row.get('score_regulatory_tone', 'N/A')} | "
        f"BERTScore={row.get('bert_score_f1', 'N/A')}"
    )
    print(f"Hallucination flag: {row.get('hallucination_flag', 'N/A')}")
    print(f"Judge reasoning: {row.get('judge_reasoning', 'N/A')}")


# Step 10 - Export Results
df.to_csv("finsight_eval_results.csv", index=False)
leaderboard.to_csv("finsight_leaderboard.csv", index=False)

print("Results exported:")
print("   finsight_eval_results.csv  - full output table")
print("   finsight_leaderboard.csv   - model leaderboard")
print("   finsight_leaderboard.png    - visualisation")


# Extension task - G-Eval for Regulatory Compliance
GEVAL_TASK_DESC = """Task: Generate a credit risk memo for a lending institution.
The memo must comply with standard lending regulations including:
fair lending practices, factual accuracy requirements, and appropriate risk disclosure."""

GEVAL_STEP_PROMPT = f"""
{GEVAL_TASK_DESC}

You will evaluate a credit memo for REGULATORY COMPLIANCE.

Step 1: Generate a list of 5 specific evaluation criteria for regulatory compliance of a credit memo.
Step 2: For each criterion, explain what a score of 1 vs 10 looks like.

Respond in JSON:
{{"criteria": [{{"name": str, "low_score": str, "high_score": str}}]}}
"""


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from model output."""
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(cleaned[start : end + 1])


def generate_geval_criteria() -> list[dict]:
    """Ask GPT-4o for a 5-criterion regulatory-compliance rubric."""
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": GEVAL_STEP_PROMPT}],
        temperature=0.0,
        max_tokens=400,
    )
    payload = _extract_json_object(resp.choices[0].message.content)
    criteria = payload.get("criteria", [])
    if not isinstance(criteria, list) or len(criteria) != 5:
        raise ValueError("Expected exactly 5 criteria in G-Eval response")
    return criteria


def score_geval_compliance(borrower_data: str, memo: str, criteria: list[dict]) -> dict:
    """Score a memo against the generated G-Eval rubric."""
    scoring_prompt = f"""
You are evaluating a credit memo for regulatory compliance.

Borrower data:
{borrower_data}

Memo:
{memo}

Criteria:
{json.dumps(criteria, indent=2)}

Score each criterion on a 1-10 scale and return valid JSON in this shape:
{{
  "scores": [
    {{"name": "<criterion>", "score": <1-10 integer>, "reasoning": "<short reason>"}}
  ],
  "overall_score": <average of the 5 scores rounded to 2 decimals>,
  "overall_reasoning": "<one short summary>"
}}
"""
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": scoring_prompt}],
        temperature=0.0,
        max_tokens=600,
    )
    payload = _extract_json_object(resp.choices[0].message.content)
    scores = payload.get("scores", [])
    if not isinstance(scores, list):
        raise ValueError("Expected scores to be a list in G-Eval response")
    return payload


print("Running G-Eval extension...")
try:
    GEVAL_CRITERIA = generate_geval_criteria()
    print("Generated G-Eval criteria:")
    for idx, criterion in enumerate(GEVAL_CRITERIA, start=1):
        print(
            f"  {idx}. {criterion.get('name', 'Unnamed criterion')} - "
            f"low: {criterion.get('low_score', '')} | high: {criterion.get('high_score', '')}"
        )

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="G-Eval"):
        if row["output"] and not row["error"]:
            eval_payload = score_geval_compliance(
                row["borrower_data"],
                row["output"],
                GEVAL_CRITERIA,
            )
            df.at[idx, "geval_overall_score"] = eval_payload.get("overall_score")
            df.at[idx, "geval_overall_reasoning"] = eval_payload.get("overall_reasoning")
            df.at[idx, "geval_scores_json"] = json.dumps(eval_payload.get("scores", []), ensure_ascii=True)

    geval_summary = (
        df.groupby("model")["geval_overall_score"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )
    print("G-Eval compliance scores:")
    print(geval_summary)
except Exception as exc:
    print(f"G-Eval extension skipped due to error: {exc}")


# Re-export after the extension so the saved files include the G-Eval results too.
df.to_csv("finsight_eval_results.csv", index=False)
leaderboard.to_csv("finsight_leaderboard.csv", index=False)

#!/usr/bin/env python3
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
                "Generate a credit risk memo for the following borrower:\n\n"
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
        if row.get("hallucin_rate") is not None and            row["hallucin_rate"] >= CONSTRAINTS["hallucin_rate"]:
            violations.append(f"{row['model']}: hallucination {row['hallucin_rate']*100:.1f}% >= 1%")
    reason = "No model meets all constraints. Violations: " + "; ".join(violations[:2])
    return {"passed": False, "reason": reason, "best_model": None}

"""
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
        assert result["hallucination_flag"] is False,             f"Expected no hallucination, got count={result['hallucination_count']}"

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
        assert result["hallucination_flag"] is False,             "Small rounding should not trigger hallucination flag"

    def test_small_numbers_ignored(self):
        """Ratios and percentages (< 200) are not flagged."""
        source = "DSCR: 1.8x."
        memo   = "The DSCR is 1.8x indicating strong debt service capacity. Score: 4/5."
        result = check_hallucination(source, memo)
        assert result["hallucination_flag"] is False,             "Small ratio numbers should not be flagged"


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
        assert lb[0]["meets_constraints"] is False,             "100% hallucination rate should fail constraints"


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

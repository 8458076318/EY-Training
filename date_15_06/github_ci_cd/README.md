# 🏦 FinSight AI — LLM Eval Pipeline

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

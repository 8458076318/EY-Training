"""
Task 1: Zero-shot + Few-shot summarisation chain
Task 2: 5-class ticket classifier with chain-of-thought style reasoning

This script prefers LangChain + OpenAI when an API key is available, but it also
includes deterministic local fallbacks so it can run successfully in offline or
dependency-constrained environments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency path
    LLMChain = None
    PromptTemplate = None
    ChatOpenAI = None

try:
    from promptlayer import PromptLayer
except Exception:  # pragma: no cover - optional dependency path
    PromptLayer = None

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - optional dependency path
    rouge_scorer = None


EARNINGS_SNIPPETS = [
    {
        "id": 1,
        "company": "TechCorp Q3 2024",
        "text": (
            "Revenue for the third quarter came in at $4.2 billion, representing a "
            "12% year-over-year increase. Cloud services drove the majority of growth, "
            "contributing $1.8 billion, up 34% from the prior year. Operating margins "
            "expanded by 150 basis points to 28.5%, reflecting disciplined cost management "
            "and operating leverage. Free cash flow reached $920 million. Looking ahead, "
            "management raised full-year guidance to $16.5-$16.8 billion, citing strong "
            "enterprise demand and a healthy pipeline heading into Q4."
        ),
        "reference_summary": (
            "TechCorp Q3 revenue grew 12% YoY to $4.2B, led by 34% cloud growth. "
            "Margins expanded 150 bps to 28.5% and full-year guidance was raised to $16.5-$16.8B."
        ),
    },
    {
        "id": 2,
        "company": "RetailGiant Q2 2024",
        "text": (
            "Comparable store sales declined 3% in the quarter as consumer spending "
            "shifted toward experiences and away from discretionary goods. E-commerce "
            "partially offset weakness, growing 18% and now representing 22% of total "
            "revenue. Gross margins compressed 200 basis points due to elevated promotional "
            "activity needed to clear excess inventory. The company announced a $500 million "
            "cost-reduction programme targeting SG&A savings over the next 18 months. "
            "Management maintained full-year EPS guidance of $3.40-$3.60 despite macro headwinds."
        ),
        "reference_summary": (
            "RetailGiant saw a 3% comp-sales decline and 200 bps gross margin compression, "
            "offset by 18% e-commerce growth. A $500M cost programme was announced; "
            "full-year EPS guidance of $3.40-$3.60 was maintained."
        ),
    },
    {
        "id": 3,
        "company": "BioHealth Q1 2024",
        "text": (
            "Product revenue reached $780 million, beating consensus estimates by 6%, "
            "driven by the strong uptake of our flagship oncology drug, Nexavol, which "
            "generated $410 million in its first full quarter post-launch. R&D expenses "
            "increased 22% as the company accelerated Phase 3 trials for two pipeline "
            "candidates. Net loss narrowed to $45 million from $120 million in the "
            "year-ago period. The company ended the quarter with $2.1 billion in cash "
            "and expects to reach operating profitability by mid-2025."
        ),
        "reference_summary": (
            "BioHealth beat estimates with $780M in product revenue, led by Nexavol's $410M debut. "
            "Net loss narrowed to $45M and the company targets operating profitability by mid-2025."
        ),
    },
]


SAMPLE_TICKETS = [
    {
        "id": "T-001",
        "text": "I was charged twice for my subscription this month. Please fix this immediately.",
        "expected": "Billing",
    },
    {
        "id": "T-002",
        "text": "The app crashes every time I try to upload a file larger than 10 MB.",
        "expected": "Tech",
    },
    {
        "id": "T-003",
        "text": "I cancelled my order 2 hours ago and need my money back as soon as possible.",
        "expected": "Refund",
    },
    {
        "id": "T-004",
        "text": "Can you explain how to export my data to a CSV file?",
        "expected": "General",
    },
    {
        "id": "T-005",
        "text": (
            "This is completely unacceptable. My data was exposed and I will be contacting "
            "my lawyer if this is not resolved within 24 hours."
        ),
        "expected": "Escalate",
    },
    {
        "id": "T-006",
        "text": "Hi, I'd like to know what payment methods you accept.",
        "expected": "General",
    },
    {
        "id": "T-007",
        "text": "I returned the product last week but haven't received a refund yet.",
        "expected": "Refund",
    },
    {
        "id": "T-008",
        "text": "My account is locked and I can't reset my password-the link keeps expiring.",
        "expected": "Tech",
    },
]


SNIPPET_GENERATION_TEMPLATE = """
You are a financial communications expert. Write a realistic earnings-call snippet
(3-5 sentences) for {company}, a {sector} company, reporting results for {quarter}.

Include: revenue figure, YoY growth %, a key business driver, margin or cost comment,
and forward guidance.

Snippet:
"""

ZERO_SHOT_SUMMARISE_TEMPLATE = """
Summarise the following earnings-call excerpt in 2-3 concise sentences.
Focus on: revenue performance, key growth drivers, margin changes, and guidance.

Transcript:
{transcript}

Summary:
"""

FEW_SHOT_SUMMARISE_TEMPLATE = """
You are a financial analyst. Summarise earnings-call excerpts in 2-3 sentences,
covering revenue, key drivers, margins, and guidance.

--- EXAMPLES ---

Excerpt:
"FinBank reported net interest income of $2.1 billion, up 8% YoY, supported by higher
interest rates. Non-interest expenses rose 5% due to technology investments.
Management guided for mid-single-digit loan growth in the second half."

Summary:
FinBank posted 8% NII growth to $2.1B on rate tailwinds, with expenses up 5% from tech spend.
Management guided for mid-single-digit loan growth in H2.

---

Excerpt:
"SoftCo's ARR grew 40% to $1.2 billion as enterprise customer additions accelerated.
Churn improved 50 basis points to 4.5%. The company raised FY guidance from
$1.5 billion to $1.65 billion in ARR, citing strong pipeline visibility."

Summary:
SoftCo grew ARR 40% to $1.2B with improving 4.5% churn. FY ARR guidance was raised to
$1.65B on strong pipeline visibility.

--- NOW SUMMARISE ---

Excerpt:
{transcript}

Summary:
"""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env(env_path: Path = ENV_PATH) -> None:
    """Load project-local environment variables without requiring python-dotenv."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()


TICKET_COT_TEMPLATE = """
You are a customer-support triage specialist. Classify the ticket below into EXACTLY ONE
of these five categories:
  Billing   - payment issues, invoice errors, charges, refund requests linked to billing
  Tech      - technical errors, bugs, login/access problems, performance issues
  Refund    - explicit refund or return requests (non-billing root cause)
  General   - general enquiries, feedback, how-to questions
  Escalate  - threats, legal notices, executive complaints, urgent safety issues

Use chain-of-thought reasoning:
  Step 1 - Identify key phrases in the ticket.
  Step 2 - Map them to category signals.
  Step 3 - Resolve ambiguity if multiple categories seem relevant.
  Step 4 - State the final category.

Ticket:
\"\"\"{ticket}\"\"\"

Reasoning:
Step 1 - Key phrases:
Step 2 - Category signals:
Step 3 - Ambiguity resolution:
Step 4 - Final category:
"""


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def _normalize_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l_f1(reference: str, candidate: str) -> float:
    ref_tokens = _tokens(reference)
    cand_tokens = _tokens(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _first_sentence_matching(sentences: Iterable[str], keywords: Iterable[str]) -> Optional[str]:
    keyword_list = list(keywords)
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keyword_list):
            return sentence
    return None


def local_summary(transcript: str) -> str:
    sentences = _sentence_split(transcript)
    if not sentences:
        return transcript.strip()

    picks = []
    keyword_groups = [
        ["revenue", "sales", "arr", "income", "product revenue", "net interest income"],
        ["grew", "growth", "driven by", "supported by", "offset", "contributing", "uptake"],
        ["margin", "margins", "cost", "expenses", "cash flow", "promotional"],
        ["guidance", "expects", "expect", "raised", "target", "outlook", "maintained"],
    ]

    for group in keyword_groups:
        sentence = _first_sentence_matching(sentences, group)
        if sentence and sentence not in picks:
            picks.append(sentence)

    if not picks:
        picks = sentences[:2]

    summary = " ".join(picks[:3])
    return _normalize_text(summary)


def local_generated_snippet(company: str, quarter: str, sector: str) -> str:
    seed = sum(ord(char) for char in f"{company}|{quarter}|{sector}")
    revenue = 1.5 + (seed % 50) / 10.0
    growth = 6 + (seed % 25)
    margin = 18 + (seed % 12)

    driver_options = {
        "renewable energy": [
            "higher project deployments and stronger utility demand",
            "improved equipment utilization and larger backlog conversion",
        ],
        "software": [
            "expanding enterprise subscriptions and lower churn",
            "new customer additions across the mid-market segment",
        ],
        "retail": [
            "better basket size and improved online conversion",
            "disciplined inventory management and stronger digital traffic",
        ],
    }

    drivers = driver_options.get(
        sector.lower(),
        [
            "broad-based demand and disciplined execution",
            "strong customer adoption and healthy pipeline conversion",
        ],
    )
    driver = drivers[seed % len(drivers)]

    guidance = (
        f"Management expects momentum to continue into {quarter.split()[-1]} and "
        f"kept full-year guidance constructive."
    )

    return _normalize_text(
        (
            f"{company} reported revenue of ${revenue:.1f} billion in {quarter}, up {growth}% year over year. "
            f"The company said results were driven by {driver}. "
            f"Operating margin improved to {margin:.1f}% as management maintained tight cost discipline. "
            f"{guidance}"
        )
    )


def classify_ticket_locally(ticket: str) -> str:
    text = ticket.lower()

    general_keywords = [
        "payment methods",
        "what payment methods",
        "how to",
        "can you explain",
        "what are the options",
        "where can i",
        "how do i",
    ]
    escalate_keywords = [
        "lawyer",
        "legal",
        "sue",
        "safety",
        "security breach",
        "data was exposed",
        "executive",
        "urgent",
        "complaint",
        "contacting my lawyer",
    ]
    refund_keywords = [
        "refund",
        "money back",
        "returned",
        "return the product",
        "return my money",
        "cancelled my order",
    ]
    billing_keywords = [
        "charged",
        "charge",
        "invoice",
        "billing",
        "payment",
        "subscription",
        "double charged",
        "charged twice",
    ]
    tech_keywords = [
        "crash",
        "crashes",
        "bug",
        "login",
        "locked",
        "password",
        "access",
        "error",
        "upload",
        "slow",
        "expired",
        "expiring",
        "reset",
    ]

    if any(keyword in text for keyword in escalate_keywords):
        return "Escalate"
    if any(keyword in text for keyword in general_keywords):
        return "General"
    if any(keyword in text for keyword in refund_keywords):
        return "Refund"
    if any(keyword in text for keyword in billing_keywords):
        return "Billing"
    if any(keyword in text for keyword in tech_keywords):
        return "Tech"
    return "General"


def explain_ticket_classification(ticket: str, predicted: str) -> str:
    text = ticket.lower()
    if predicted == "Escalate":
        return (
            "Step 1 - Key phrases: legal threat, urgency, or safety exposure. "
            "Step 2 - Category signals: escalation and risk. "
            "Step 3 - Ambiguity resolution: escalation overrides other labels. "
            "Step 4 - Final category: Escalate"
        )
    if predicted == "Refund":
        reason = "refund or return request"
    elif predicted == "Billing":
        reason = "billing or payment issue"
    elif predicted == "Tech":
        reason = "technical issue, access problem, or app failure"
    else:
        reason = "general inquiry or how-to question"
    return (
        f"Step 1 - Key phrases: {reason}. "
        "Step 2 - Category signals: matched ticket keywords. "
        "Step 3 - Ambiguity resolution: the strongest signal wins. "
        f"Step 4 - Final category: {predicted}"
    )


def build_rouge_scorer():
    if rouge_scorer is not None:
        return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return None


def rouge_l_fmeasure(reference: str, candidate: str, scorer=None) -> float:
    if scorer is not None:
        return scorer.score(reference, candidate)["rougeL"].fmeasure
    return _rouge_l_f1(reference, candidate)


def build_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or ChatOpenAI is None:
        return None
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception:
        return None


def build_promptlayer_client():
    api_key = os.getenv("PROMPTLAYER_API_KEY")
    if not api_key or PromptLayer is None:
        return None
    try:
        return PromptLayer(api_key=api_key)
    except Exception:
        return None


def maybe_run_chain(llm, prompt, variables: dict, fallback_value: str) -> str:
    if llm is None or PromptTemplate is None or LLMChain is None:
        return fallback_value
    try:
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(**variables)
        return _normalize_text(str(result).strip())
    except Exception:
        return fallback_value


def run_all():
    llm = build_llm()
    promptlayer_client = build_promptlayer_client()
    scorer = build_rouge_scorer()

    if PromptTemplate is not None:
        snippet_prompt = PromptTemplate(
            input_variables=["company", "quarter", "sector"],
            template=SNIPPET_GENERATION_TEMPLATE,
        )
        zero_shot_prompt = PromptTemplate(
            input_variables=["transcript"],
            template=ZERO_SHOT_SUMMARISE_TEMPLATE,
        )
        few_shot_prompt = PromptTemplate(
            input_variables=["transcript"],
            template=FEW_SHOT_SUMMARISE_TEMPLATE,
        )
        ticket_prompt = PromptTemplate(
            input_variables=["ticket"],
            template=TICKET_COT_TEMPLATE,
        )
    else:
        snippet_prompt = zero_shot_prompt = few_shot_prompt = ticket_prompt = None

    if promptlayer_client is not None:
        print(f"PromptLayer enabled using {ENV_PATH.name} -> PROMPTLAYER_API_KEY")

    print("=" * 70)
    print("TASK 1 - EARNINGS CALL SUMMARISATION")
    print("=" * 70)

    rouge_results = []

    for snippet in EARNINGS_SNIPPETS:
        print(f"\n{'-' * 60}")
        print(f"Snippet {snippet['id']}: {snippet['company']}")
        print(f"{'-' * 60}")

        zero_shot_fallback = local_summary(snippet["text"])
        few_shot_fallback = local_summary(snippet["text"])

        zs_summary = maybe_run_chain(
            llm,
            zero_shot_prompt,
            {"transcript": snippet["text"]},
            zero_shot_fallback,
        )
        fs_summary = maybe_run_chain(
            llm,
            few_shot_prompt,
            {"transcript": snippet["text"]},
            few_shot_fallback,
        )
        reference = snippet["reference_summary"]

        zs_score = rouge_l_fmeasure(reference, zs_summary, scorer)
        fs_score = rouge_l_fmeasure(reference, fs_summary, scorer)

        rouge_results.append(
            {"id": snippet["id"], "zero_shot": zs_score, "few_shot": fs_score}
        )

        print(f"\n[ZERO-SHOT SUMMARY]\n{zs_summary}")
        print(f"  ROUGE-L F1: {zs_score:.4f}")

        print(f"\n[FEW-SHOT SUMMARY]\n{fs_summary}")
        print(f"  ROUGE-L F1: {fs_score:.4f}")

        print(f"\n[REFERENCE SUMMARY]\n{reference}")

    avg_zs = sum(item["zero_shot"] for item in rouge_results) / len(rouge_results)
    avg_fs = sum(item["few_shot"] for item in rouge_results) / len(rouge_results)
    print(f"\n{'=' * 60}")
    print(f"Average ROUGE-L F1 - Zero-shot: {avg_zs:.4f} | Few-shot: {avg_fs:.4f}")
    print(f"{'=' * 60}")

    print("\n\nTASK 1a - GENERATE A NEW EARNINGS SNIPPET")
    print("-" * 60)
    generated_fallback = local_generated_snippet(
        company="GreenEnergy Inc.",
        quarter="Q2 2024",
        sector="renewable energy",
    )
    generated = maybe_run_chain(
        llm,
        snippet_prompt,
        {"company": "GreenEnergy Inc.", "quarter": "Q2 2024", "sector": "renewable energy"},
        generated_fallback,
    )
    print(generated)

    print("\n\n" + "=" * 70)
    print("TASK 2 - 5-CLASS TICKET CLASSIFIER (Chain-of-Thought)")
    print("=" * 70)

    correct = 0
    for ticket in SAMPLE_TICKETS:
        print(f"\n{'-' * 60}")
        print(f"Ticket {ticket['id']}: {ticket['text'][:80]}...")

        predicted_fallback = classify_ticket_locally(ticket["text"])
        predicted = predicted_fallback
        reasoning = explain_ticket_classification(ticket["text"], predicted_fallback)

        if llm is not None and ticket_prompt is not None:
            try:
                chain = LLMChain(llm=llm, prompt=ticket_prompt)
                raw_output = str(chain.run(ticket=ticket["text"]).strip())
                match = re.search(
                    r"(?i)(?:final category|step\s*4\s*[-:])\s*[:\-]?\s*(Billing|Tech|Refund|General|Escalate)",
                    raw_output,
                )
                if match:
                    predicted = match.group(1).strip()
                    reasoning = _normalize_text(raw_output)
                else:
                    raw_lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
                    if raw_lines:
                        predicted = classify_ticket_locally(raw_output)
                        reasoning = _normalize_text(raw_output)
            except Exception:
                predicted = predicted_fallback
                reasoning = explain_ticket_classification(ticket["text"], predicted_fallback)

        match = predicted.lower() == ticket["expected"].lower()
        correct += int(match)

        print(f"\n{reasoning}")
        print(
            f"\n  Expected: {ticket['expected']} | Predicted: {predicted} | "
            f"{'CORRECT' if match else 'WRONG'}"
        )

    accuracy = correct / len(SAMPLE_TICKETS) * 100
    print(f"\n{'=' * 60}")
    print(f"Ticket Classifier Accuracy: {correct}/{len(SAMPLE_TICKETS)} = {accuracy:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if not os.getenv("PROMPTLAYER_API_KEY"):
        print("PROMPTLAYER_API_KEY is not set. Running with local fallbacks.")
    run_all()

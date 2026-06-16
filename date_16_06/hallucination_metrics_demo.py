"""BERTScore demo.

This script evaluates a candidate answer against a reference using BERTScore
and saves a simple graph of the resulting F1 score.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from bert_score import score as bert_score  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bert_score = None


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def bertscore_proxy(candidate: str, reference: str) -> dict:
    """Compute BERTScore if enabled, otherwise use token-F1 fallback."""
    candidate = normalize_whitespace(candidate)
    reference = normalize_whitespace(reference)

    use_real_bertscore = os.getenv("ENABLE_BERTSCORE", "0").strip().lower() in {"1", "true", "yes"}
    if use_real_bertscore and bert_score is not None:
        try:
            precision, recall, f1 = bert_score(
                [candidate],
                [reference],
                lang="en",
                verbose=False,
                rescale_with_baseline=True,
            )
            return {
                "precision": round(float(precision[0]), 4),
                "recall": round(float(recall[0]), 4),
                "f1": round(float(f1[0]), 4),
                "mode": "bert_score",
            }
        except Exception:
            pass

    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mode": "token_f1"}

    cand_set = set(cand_tokens)
    ref_set = set(ref_tokens)
    overlap = len(cand_set & ref_set)
    precision = overlap / len(cand_set)
    recall = overlap / len(ref_set)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mode": "token_f1",
    }


@dataclass
class BertScoreReport:
    candidate: str
    reference: str
    bertscore: dict


def evaluate_answer(candidate: str, reference: str) -> BertScoreReport:
    return BertScoreReport(
        candidate=candidate,
        reference=reference,
        bertscore=bertscore_proxy(candidate, reference),
    )


def print_report(report: BertScoreReport) -> None:
    print("\n=== BERTScore Report ===")
    print(f"Candidate: {report.candidate}")
    print(f"Reference:  {report.reference}")
    print("\nBERTScore")
    print(f"  mode: {report.bertscore['mode']}")
    print(f"  precision: {report.bertscore['precision']}")
    print(f"  recall: {report.bertscore['recall']}")
    print(f"  f1: {report.bertscore['f1']}")


def plot_report(report: BertScoreReport, output_path: Path) -> None:
    """Create a one-bar chart for the BERTScore F1 value."""
    score = report.bertscore["f1"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["BERTScore F1"], [score], color=["#5c7cfa"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (0 to 1)")
    ax.set_title("BERTScore")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=0)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_sample_data() -> dict:
    reference = "France's capital is Paris, and the Eiffel Tower is located there."
    candidate = (
        "France's capital is Paris, where the Eiffel Tower stands. "
        "Paris has 2.1 million residents. "
        "The country uses the euro."
    )
    return {"candidate": candidate, "reference": reference}


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    data = load_sample_data()
    report = evaluate_answer(candidate=data["candidate"], reference=data["reference"])
    print_report(report)
    graph_path = output_dir / "bertscore_graph.png"
    plot_report(report, graph_path)
    print(f"\nSaved graph to: {graph_path}")


if __name__ == "__main__":
    main()

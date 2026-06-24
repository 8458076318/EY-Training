"""Generate a visual summary of the Day20 lead-summary scoring logic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_PATH = Path(__file__).with_name("Day20_Lead_Summary_Scoring_Chart.png")


def add_box(ax, xy, w, h, title, body, facecolor, edgecolor):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#10223b",
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#10223b",
        linespacing=1.35,
    )


def add_arrow(ax, start, end, color="#355C7D"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=18,
        linewidth=2.0,
        color=color,
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor("#F4F7FB")
    ax.set_facecolor("#F4F7FB")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Lead Summary Scoring Logic",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        0.5,
        0.91,
        "Fallback heuristics used when the model response is missing, malformed, or out of range",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#475569",
    )

    add_box(
        ax,
        (0.05, 0.63),
        0.24,
        0.2,
        "Inputs",
        "Summary text\nSource / lead notes",
        "#E8F1FF",
        "#7AA7E8",
    )
    add_box(
        ax,
        (0.35, 0.69),
        0.26,
        0.14,
        "Groundedness",
        "Token overlap with source\nMore overlap = higher score",
        "#EAF8EF",
        "#6BBF8D",
    )
    add_box(
        ax,
        (0.35, 0.47),
        0.26,
        0.14,
        "Usefulness",
        "Action words + specifics + metrics\nMore signals = higher score",
        "#FFF2E2",
        "#E0A05A",
    )
    add_box(
        ax,
        (0.70, 0.62),
        0.25,
        0.22,
        "Final Score",
        "Groundedness: 1-5\nUsefulness: 1-5\nNotes explain the reason",
        "#F3E8FF",
        "#A78BFA",
    )
    add_box(
        ax,
        (0.70, 0.28),
        0.25,
        0.18,
        "Model path",
        "LLM JSON is used first\nHeuristic only if response fails",
        "#EEF2FF",
        "#7C8EDB",
    )

    add_arrow(ax, (0.29, 0.73), (0.35, 0.76))
    add_arrow(ax, (0.29, 0.73), (0.35, 0.54))
    add_arrow(ax, (0.61, 0.76), (0.70, 0.74))
    add_arrow(ax, (0.61, 0.54), (0.70, 0.74))
    add_arrow(ax, (0.82, 0.46), (0.82, 0.62))

    ax.text(
        0.14,
        0.56,
        "Heuristic path\nif the judge output fails",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFFFF", edgecolor="#CBD5E1"),
    )
    add_arrow(ax, (0.19, 0.63), (0.19, 0.58), color="#94A3B8")

    ax.text(
        0.5,
        0.12,
        "Groundedness = source alignment   |   Usefulness = actionability + specificity + metrics",
        ha="center",
        va="center",
        fontsize=12,
        color="#334155",
        fontweight="semibold",
    )
    ax.text(
        0.5,
        0.07,
        "Saved as Day20_Lead_Summary_Scoring_Chart.png",
        ha="center",
        va="center",
        fontsize=10,
        color="#64748B",
    )

    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()


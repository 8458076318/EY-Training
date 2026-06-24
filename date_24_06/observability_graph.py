"""Generate a diagram that explains the multi-agent observability flow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).with_name("observability_flow.png")


def box(ax, xy, w, h, text, fc, ec="#1f2937", text_color="#111827", fontsize=12, weight="normal"):
    rect = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        family="DejaVu Sans",
    )


def arrow(ax, start, end, color="#374151", style="-|>", mutation_scale=18, lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            shrinkA=8,
            shrinkB=8,
        )
    )


def build_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Multi-Agent Observability Flow",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.905,
        "trace_id links the whole run, span_id identifies each agent execution",
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )

    # Left side: execution flow
    box(ax, (0.05, 0.68), 0.18, 0.09, "Orchestrator\nrun_started", "#dbeafe", weight="bold")
    box(ax, (0.31, 0.78), 0.18, 0.09, "Planner\nagent_started", "#ecfccb")
    box(ax, (0.31, 0.61), 0.18, 0.09, "Researcher\nagent_progress", "#ecfccb")
    box(ax, (0.31, 0.44), 0.18, 0.09, "Writer\nagent_failed", "#fee2e2")
    box(ax, (0.31, 0.27), 0.18, 0.09, "Reviewer\nagent_completed", "#ecfccb")
    box(ax, (0.58, 0.68), 0.18, 0.09, "run_summary\nfinal status", "#c7d2fe", weight="bold")

    arrow(ax, (0.23, 0.725), (0.31, 0.825))
    arrow(ax, (0.23, 0.725), (0.31, 0.655))
    arrow(ax, (0.23, 0.725), (0.31, 0.485))
    arrow(ax, (0.23, 0.725), (0.31, 0.315))
    arrow(ax, (0.49, 0.825), (0.58, 0.725))
    arrow(ax, (0.49, 0.655), (0.58, 0.725))
    arrow(ax, (0.49, 0.485), (0.58, 0.725))
    arrow(ax, (0.49, 0.315), (0.58, 0.725))

    # Right side: telemetry fields
    ax.text(0.78, 0.80, "Every event carries:", fontsize=15, fontweight="bold", color="#0f172a", ha="center")
    box(ax, (0.69, 0.67), 0.22, 0.08, "timestamp", "#e2e8f0")
    box(ax, (0.69, 0.56), 0.22, 0.08, "trace_id", "#e2e8f0")
    box(ax, (0.69, 0.45), 0.22, 0.08, "span_id", "#e2e8f0")
    box(ax, (0.69, 0.34), 0.22, 0.08, "event + agent + progress", "#e2e8f0")

    arrow(ax, (0.58, 0.725), (0.69, 0.725))
    arrow(ax, (0.80, 0.67), (0.80, 0.64), color="#64748b", lw=1.2)
    arrow(ax, (0.80, 0.56), (0.80, 0.53), color="#64748b", lw=1.2)
    arrow(ax, (0.80, 0.45), (0.80, 0.42), color="#64748b", lw=1.2)

    # Bottom callouts
    box(ax, (0.05, 0.08), 0.36, 0.1, "Throttled progress\nabout every 25%", "#fef3c7", fontsize=13, weight="bold")
    box(ax, (0.45, 0.08), 0.23, 0.1, "Failure pinpoints\nwhich step broke", "#fee2e2", fontsize=13, weight="bold")
    box(ax, (0.72, 0.08), 0.23, 0.1, "Summary reports\nstatus + duration", "#d1fae5", fontsize=13, weight="bold")

    arrow(ax, (0.23, 0.18), (0.23, 0.27), color="#b45309")
    arrow(ax, (0.56, 0.18), (0.56, 0.44), color="#b91c1c")
    arrow(ax, (0.83, 0.18), (0.68, 0.68), color="#15803d")

    fig.text(
        0.5,
        0.015,
        "Created offline with the standard Python stack; suitable for the date_24_06 submission folder.",
        ha="center",
        fontsize=10,
        color="#64748b",
    )

    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    build_diagram()
    print(str(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

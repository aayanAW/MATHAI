#!/usr/bin/env python3
"""Generate Figure 8 (Echo Chamber) and Figure 9 (Calibration) for ExeVer paper.

Reads statistical_analysis.json and produces:
  - fig8_echo_chamber.pdf / .png  (2-panel: echo by level + detection accuracy)
  - fig9_calibration.pdf / .png   (verdict-conditioned accuracy)

Usage:
    python generate_echo_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Style (matches generate_all_figures.py)
# ---------------------------------------------------------------------------
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#999999"
LIGHT_GRAY = "#BBBBBB"
DARK_GRAY = "#555555"
SKY_BLUE = "#56B4E9"
YELLOW = "#F0E442"

DPI = 300
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10


def apply_style() -> None:
    """Set global matplotlib style for clean academic figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def load_json(name: str) -> dict[str, Any]:
    """Load a JSON result file."""
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 8: Echo Chamber as Error Detector (2-panel)
# ---------------------------------------------------------------------------
def generate_fig8(data: dict[str, Any]) -> None:
    """Two-panel figure: echo rate by difficulty + detection accuracy."""
    echo = data["echo_chamber_analysis"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- LEFT PANEL: Echo rate by difficulty level ---
    levels = ["1", "2", "3", "4", "5"]
    echo_rates = [echo["by_level"][lv]["echo_rate"] * 100 for lv in levels]
    totals = [echo["by_level"][lv]["total"] for lv in levels]

    # Warm gradient from light orange to deep red
    cmap = plt.cm.YlOrRd
    norm_vals = np.linspace(0.3, 0.9, len(levels))
    colors_left = [cmap(v) for v in norm_vals]

    bars1 = ax1.bar(levels, echo_rates, color=colors_left, edgecolor="white",
                    linewidth=0.8, width=0.65, zorder=3)

    # Value labels on bars
    for bar, rate, n in zip(bars1, echo_rates, totals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{rate:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f"N={n}",
                 ha="center", va="center", fontsize=8, color="white",
                 fontweight="bold")

    ax1.set_xlabel("MATH Difficulty Level", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Echo Rate (%)", fontsize=LABEL_SIZE)
    ax1.set_title("Echo Chamber by Difficulty Level", fontsize=TITLE_SIZE,
                  fontweight="bold")
    ax1.set_ylim(0, max(echo_rates) * 1.3)

    # Annotation
    ax1.annotate(
        "Echo rate perfectly\npredicts errors",
        xy=(4, echo_rates[4]),  # point at level 5 bar (index 4 -> x-label "5")
        xytext=(2.5, echo_rates[4] + 5),
        fontsize=10, fontstyle="italic", color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
        ha="center",
    )

    # Light grid
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, alpha=0.25, linestyle="--")

    # --- RIGHT PANEL: Echo detection accuracy ---
    categories = ["Non-Echo\nALL_PASS", "Echo\nALL_PASS"]
    accuracies = [
        echo["non_echo_accuracy"] * 100,
        echo["echo_accuracy"] * 100,
    ]
    ns = [echo["n_non_echo"], echo["n_echo"]]
    bar_colors = [GREEN, RED]

    bars2 = ax2.bar(categories, accuracies, color=bar_colors, edgecolor="white",
                    linewidth=0.8, width=0.55, zorder=3)

    # Value labels
    for bar, acc, n, col in zip(bars2, accuracies, ns, bar_colors):
        if acc > 10:
            # Accuracy label above bar
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 2,
                     f"{acc:.0f}%",
                     ha="center", va="bottom", fontsize=14, fontweight="bold")
            # N inside bar
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() / 2,
                     f"N = {n}",
                     ha="center", va="center", fontsize=11, color="white",
                     fontweight="bold")
        else:
            # Zero-height bar: stack labels vertically above baseline
            ax2.text(bar.get_x() + bar.get_width() / 2, 12,
                     f"{acc:.0f}%",
                     ha="center", va="bottom", fontsize=14, fontweight="bold",
                     color=col)
            ax2.text(bar.get_x() + bar.get_width() / 2, 5,
                     f"N = {n}",
                     ha="center", va="bottom", fontsize=11, color=col,
                     fontweight="bold")

    ax2.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
    ax2.set_title("Echo Detection Accuracy", fontsize=TITLE_SIZE,
                  fontweight="bold")
    ax2.set_ylim(0, 120)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, alpha=0.25, linestyle="--")

    # Subtitle text
    fig.text(0.5, -0.02,
             "Echo detection is a perfect binary classifier within verified (ALL_PASS) solutions",
             ha="center", fontsize=11, fontstyle="italic", color=DARK_GRAY)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"fig8_echo_chamber.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Verification as Confidence Signal
# ---------------------------------------------------------------------------
def generate_fig9(data: dict[str, Any]) -> None:
    """Bar chart: accuracy conditioned on ExeVer verdict."""
    cal = data["confidence_calibration"]

    # Ordered verdict categories
    verdict_labels = ["ALL_PASS", "REPAIRED", "FAIL_STEP_-1",
                      "SYNTAX_ERROR", "RUNTIME_ERROR"]
    display_labels = ["ALL_PASS", "REPAIRED", "FAIL_STEP_-1",
                      "SYNTAX\nERROR", "RUNTIME\nERROR"]

    accuracies = []
    ns = []
    for v in verdict_labels:
        entry = cal["per_verdict"][v]
        accuracies.append(entry["accuracy"] * 100)
        ns.append(entry["n"])

    # Color scheme: green for high accuracy, orange/red for low
    bar_colors = []
    for acc in accuracies:
        if acc >= 90:
            bar_colors.append(GREEN)
        elif acc >= 70:
            bar_colors.append(BLUE)
        elif acc >= 50:
            bar_colors.append(ORANGE)
        else:
            bar_colors.append(RED)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(verdict_labels))
    bars = ax.bar(x, accuracies, color=bar_colors, edgecolor="white",
                  linewidth=0.8, width=0.6, zorder=3)

    # Labels on bars
    for i, (bar, acc, n) in enumerate(zip(bars, accuracies, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        # N count inside bar
        y_n = bar.get_height() / 2 if bar.get_height() > 15 else bar.get_height() + 8
        color_n = "white" if bar.get_height() > 15 else DARK_GRAY
        ax.text(bar.get_x() + bar.get_width() / 2, y_n,
                f"N={n}",
                ha="center", va="center", fontsize=9, color=color_n,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=LABEL_SIZE)
    ax.set_title("Verification as Confidence Signal:\nAccuracy Conditioned on ExeVer Verdict",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")

    # Annotation: ExeVer verdicts are informative
    ax.annotate(
        "Verdicts are informative\n(17.2 pp gap: ALL_PASS vs fallback)",
        xy=(0, accuracies[0]),
        xytext=(2.5, 105),
        fontsize=10, fontstyle="italic", color=DARK_GRAY,
        arrowprops=dict(arrowstyle="->", color=DARK_GRAY, lw=1.2),
        ha="center",
    )

    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"fig9_calibration.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    data = load_json("statistical_analysis.json")

    print("Generating Figure 8: Echo Chamber as Error Detector ...")
    generate_fig8(data)

    print("Generating Figure 9: Verification as Confidence Signal ...")
    generate_fig9(data)

    print("\nDone. All figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()

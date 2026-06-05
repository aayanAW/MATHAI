#!/usr/bin/env python3
"""Generate all publication-quality figures for ExeVer paper.

Reads result JSON files from ../results/ and produces 7 figures as PDF+PNG
in the current directory (figures/).

Usage:
    python generate_all_figures.py
"""

from __future__ import annotations

import json
import os
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
# Style configuration
# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Okabe-Ito inspired)
BLUE = "#0072B2"       # ExeVer accent
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


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_json(name: str) -> dict[str, Any] | None:
    """Load a JSON result file; return None if missing."""
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  [WARN] Missing result file: {name}")
        return None
    with open(path) as f:
        return json.load(f)


def safe_get(d: dict | None, *keys: str, default: Any = None) -> Any:
    """Nested dict access that never raises."""
    curr = d
    for k in keys:
        if curr is None or not isinstance(curr, dict):
            return default
        curr = curr.get(k, default)
    return curr


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF and PNG."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    print(f"  Saved {pdf_path.name}  +  {png_path.name}")


# ---------------------------------------------------------------------------
# Figure 1 -- Main accuracy comparison on MATH-500
# ---------------------------------------------------------------------------
def fig1_accuracy_comparison(exp5: dict, exp7: dict, exp8: dict,
                              exp11: dict) -> None:
    methods = [
        "Greedy CoT",
        "Self-Correction",
        "SymCode",
        "Majority@4",
        "Best-of-4",
        "LLM-as-Judge@4",
        "ExeVer",
    ]
    accs = [
        safe_get(exp5, "accuracy", "greedy_cot", default=0.832),
        safe_get(exp8, "corrected_accuracy", default=0.748),
        safe_get(exp7, "accuracy", default=0.636),
        safe_get(exp5, "accuracy", "majority_4", default=0.848),
        safe_get(exp5, "accuracy", "best_of_4", default=0.888),
        safe_get(exp11, "accuracy", "judge_selected_4", default=0.830),
        safe_get(exp5, "accuracy", "exever", default=0.834),
    ]
    accs_pct = [a * 100 for a in accs]

    colors = [GRAY, RED, ORANGE, LIGHT_GRAY, LIGHT_GRAY, DARK_GRAY, BLUE]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accs_pct, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.65)

    # Value labels on bars
    for bar, val in zip(bars, accs_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=TICK_SIZE,
                fontweight="bold")

    # Greedy CoT reference line
    greedy_pct = accs_pct[0]
    ax.axhline(y=greedy_pct, color=GRAY, linestyle="--", linewidth=1,
               alpha=0.7)
    ax.text(len(methods) - 0.5, greedy_pct + 0.4, "Greedy CoT",
            ha="right", va="bottom", fontsize=8, color=GRAY, fontstyle="italic")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Method Comparison on MATH-500")
    ax.set_ylim(55, 95)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_figure(fig, "fig1_accuracy_comparison")


# ---------------------------------------------------------------------------
# Figure 2 -- Verification coverage by difficulty level
# ---------------------------------------------------------------------------
def fig2_coverage_by_level(exp5: dict, exp12: dict) -> None:
    # Compute coverage per level from verifiability_map
    level_coverage: dict[int, list[float]] = {}
    vmap = safe_get(exp5, "verifiability_map", default={})
    for key, val in vmap.items():
        level = int(key.split("_")[-1])
        if level not in level_coverage:
            level_coverage[level] = {"n": 0, "cov_sum": 0.0}
        level_coverage[level]["n"] += val["n"]
        level_coverage[level]["cov_sum"] += val["n"] * val["coverage"]

    levels = sorted(level_coverage.keys())
    coverages = [level_coverage[l]["cov_sum"] / level_coverage[l]["n"] * 100
                 for l in levels]

    gsm_coverage = safe_get(exp12, "coverage", "verification_coverage",
                            default=0.908) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    # Main line plot
    ax.plot(levels, coverages, marker="o", color=BLUE, linewidth=2.5,
            markersize=8, markeredgecolor="white", markeredgewidth=1.5,
            label="MATH-500 (by level)", zorder=3)

    # Fill area under line
    ax.fill_between(levels, coverages, alpha=0.12, color=BLUE)

    # Point labels
    for lv, cov in zip(levels, coverages):
        ax.text(lv, cov + 2.5, f"{cov:.1f}%", ha="center", va="bottom",
                fontsize=TICK_SIZE, color=BLUE)

    # GSM8K annotation
    ax.axhline(y=gsm_coverage, color=GREEN, linestyle="--", linewidth=1.5,
               alpha=0.8)
    ax.text(4.8, gsm_coverage + 1.5, f"GSM8K: {gsm_coverage:.1f}%",
            ha="right", va="bottom", fontsize=TICK_SIZE, color=GREEN,
            fontweight="bold")

    ax.set_xlabel("MATH Difficulty Level")
    ax.set_ylabel("Verification Coverage (%)")
    ax.set_title("Verification Coverage by Difficulty")
    ax.set_xticks(levels)
    ax.set_ylim(50, 100)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_figure(fig, "fig2_coverage_by_level")


# ---------------------------------------------------------------------------
# Figure 3 -- Accuracy by difficulty level
# ---------------------------------------------------------------------------
def fig3_accuracy_by_level(exp5: dict, exp7: dict) -> None:
    by_level = safe_get(exp5, "by_level", default={})
    sym_by_level = safe_get(exp7, "by_level", default={})

    levels = [1, 2, 3, 4, 5]
    greedy_vals = [safe_get(by_level, str(l), "greedy", default=0) * 100
                   for l in levels]
    exever_vals = [safe_get(by_level, str(l), "exever", default=0) * 100
                   for l in levels]
    maj4_vals = [safe_get(by_level, str(l), "maj4", default=0) * 100
                 for l in levels]
    sym_vals = [safe_get(sym_by_level, str(l), "accuracy", default=0) * 100
                for l in levels]

    x = np.arange(len(levels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, greedy_vals, width, label="Greedy CoT",
           color=GRAY, edgecolor="white", linewidth=0.5)
    ax.bar(x - 0.5 * width, exever_vals, width, label="ExeVer",
           color=BLUE, edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.5 * width, maj4_vals, width, label="Majority@4",
           color=SKY_BLUE, edgecolor="white", linewidth=0.5)
    ax.bar(x + 1.5 * width, sym_vals, width, label="SymCode",
           color=ORANGE, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("MATH Difficulty Level")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Difficulty Level")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {l}" for l in levels])
    ax.set_ylim(40, 105)
    ax.legend(loc="upper right", frameon=False)

    # Value labels on ExeVer bars
    for i, val in enumerate(exever_vals):
        ax.text(x[i] - 0.5 * width, val + 1, f"{val:.0f}", ha="center",
                va="bottom", fontsize=8, color=BLUE, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, "fig3_accuracy_by_level")


# ---------------------------------------------------------------------------
# Figure 4 -- Ablation results
# ---------------------------------------------------------------------------
def fig4_ablations(exp9: dict) -> None:
    results = safe_get(exp9, "results", default={})

    labels = [
        "Full ExeVer",
        "Verify-only",
        "Re-derivation",
        "Multi-sample",
        "CoT Greedy",
        "Majority@4",
        "Interleaved",
    ]
    keys = [
        "ExeVer_full", "A1_verify_only", "A3_rederivation",
        "A5_multisample", "CoT_greedy", "Majority_4", "A4_interleaved",
    ]
    values = [results.get(k, 0) * 100 for k in keys]

    # Colors: full ExeVer in blue, ablation variants in gradient, baselines
    colors = [BLUE, SKY_BLUE, SKY_BLUE, SKY_BLUE, GRAY, GRAY, RED]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=TICK_SIZE)

    # CoT greedy reference line
    cot_greedy_val = results.get("CoT_greedy", 0.8367) * 100
    ax.axvline(x=cot_greedy_val, color=GRAY, linestyle="--", linewidth=1,
               alpha=0.7, zorder=0)
    ax.text(cot_greedy_val + 0.15, len(labels) - 0.5,
            f"CoT Greedy ({cot_greedy_val:.1f}%)",
            ha="left", va="center", fontsize=8, color=GRAY, fontstyle="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Ablation Study (300 MATH problems)")
    ax.set_xlim(75, 88)
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "fig4_ablations")


# ---------------------------------------------------------------------------
# Figure 5 -- Scaling analysis across models
# ---------------------------------------------------------------------------
def fig5_scaling(exp10: dict) -> None:
    results = safe_get(exp10, "results", default={})

    model_labels = ["1.5B-Math", "7B-General", "7B-Math"]
    model_keys = ["Qwen2.5-Math-1.5B", "Qwen2.5-7B-General", "Qwen2.5-Math-7B"]

    greedy_vals, maj4_vals, exever_vals = [], [], []
    for mk in model_keys:
        md = results.get(mk, {})
        greedy_vals.append(md.get("greedy", 0) * 100)
        maj4_vals.append(md.get("majority_4", 0) * 100)
        exever_vals.append(md.get("exever", 0) * 100)

    x = np.arange(len(model_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width, greedy_vals, width, label="Greedy CoT",
                color=GRAY, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, maj4_vals, width, label="Majority@4",
                color=SKY_BLUE, edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + width, exever_vals, width, label="ExeVer",
                color=BLUE, edgecolor="white", linewidth=0.5)

    # Annotate ExeVer gain/loss relative to greedy
    for i, (gv, ev) in enumerate(zip(greedy_vals, exever_vals)):
        delta = ev - gv
        sign = "+" if delta >= 0 else ""
        color = GREEN if delta >= 0 else RED
        ax.text(x[i] + width, ev + 1.2, f"{sign}{delta:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=color)

    # Bar value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Scaling Analysis Across Model Sizes")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(60, 92)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig5_scaling")


# ---------------------------------------------------------------------------
# Figure 6 -- GSM8K vs MATH coverage comparison
# ---------------------------------------------------------------------------
def fig6_coverage_comparison(exp5: dict, exp12: dict) -> None:
    # MATH-500 metrics
    math_coverage = 0.0
    vmap = safe_get(exp5, "verifiability_map", default={})
    total_n, total_cov = 0, 0.0
    for val in vmap.values():
        total_n += val["n"]
        total_cov += val["n"] * val["coverage"]
    if total_n > 0:
        math_coverage = total_cov / total_n * 100
    else:
        math_coverage = 76.8  # fallback

    math_echo = safe_get(exp5, "echo_chamber", "rate", default=0.138) * 100
    math_repair_att = safe_get(exp5, "repair", "attempted", default=64)
    math_repair_suc = safe_get(exp5, "repair", "successful", default=54)
    math_repair_pct = (math_repair_suc / math_repair_att * 100
                       if math_repair_att > 0 else 0)

    # GSM8K metrics
    gsm_coverage = safe_get(exp12, "coverage", "verification_coverage",
                            default=0.908) * 100
    gsm_echo = safe_get(exp12, "echo_chamber", "rate", default=0.035) * 100
    gsm_repair_att = safe_get(exp12, "repair", "attempted", default=70)
    gsm_repair_suc = safe_get(exp12, "repair", "successful", default=68)
    gsm_repair_pct = (gsm_repair_suc / gsm_repair_att * 100
                      if gsm_repair_att > 0 else 0)

    metrics = ["Verification\nCoverage", "Echo Chamber\nRate",
               "Repair Success\nRate"]
    gsm_vals = [gsm_coverage, gsm_echo, gsm_repair_pct]
    math_vals = [math_coverage, math_echo, math_repair_pct]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, gsm_vals, width, label="GSM8K",
                color=GREEN, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, math_vals, width, label="MATH-500",
                color=BLUE, edgecolor="white", linewidth=0.5)

    # Value labels
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=TICK_SIZE,
                    fontweight="bold")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("GSM8K vs MATH-500: Verification Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig6_coverage_comparison")


# ---------------------------------------------------------------------------
# Figure 7 -- Self-correction failure
# ---------------------------------------------------------------------------
def fig7_selfcorrect_failure(exp8: dict) -> None:
    initial_acc = safe_get(exp8, "initial_accuracy", default=0.832) * 100
    corrected_acc = safe_get(exp8, "corrected_accuracy", default=0.748) * 100
    by_level = safe_get(exp8, "by_level", default={})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                    gridspec_kw={"width_ratios": [1, 1.4]})

    # --- Left panel: overall before/after ---
    labels = ["Before\n(Greedy CoT)", "After\n(Self-Correction)"]
    vals = [initial_acc, corrected_acc]
    bar_colors = [GRAY, RED]
    bars = ax1.bar(labels, vals, color=bar_colors, edgecolor="white",
                   linewidth=0.5, width=0.55)

    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=LABEL_SIZE,
                 fontweight="bold")

    # Arrow showing degradation
    drop = initial_acc - corrected_acc
    ax1.annotate(
        f"{drop:.1f} pp",
        xy=(1, corrected_acc + 1),
        xytext=(0, initial_acc - 1),
        fontsize=LABEL_SIZE, fontweight="bold", color=RED,
        ha="center", va="center",
        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.3",
                        color=RED, lw=2.5),
    )

    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Overall Accuracy Drop")
    ax1.set_ylim(65, 92)

    # --- Right panel: by difficulty level ---
    levels = [1, 2, 3, 4, 5]
    init_vals = [safe_get(by_level, str(l), "initial", default=0) * 100
                 for l in levels]
    corr_vals = [safe_get(by_level, str(l), "corrected", default=0) * 100
                 for l in levels]

    x = np.arange(len(levels))
    width = 0.32
    ax2.bar(x - width / 2, init_vals, width, label="Before (Greedy CoT)",
            color=GRAY, edgecolor="white", linewidth=0.5)
    ax2.bar(x + width / 2, corr_vals, width, label="After (Self-Correction)",
            color=RED, edgecolor="white", linewidth=0.5)

    # Annotate drops
    for i, (iv, cv) in enumerate(zip(init_vals, corr_vals)):
        d = iv - cv
        if d > 0:
            ax2.text(x[i] + width / 2, cv - 2, f"-{d:.0f}",
                     ha="center", va="top", fontsize=8, color=RED,
                     fontweight="bold")

    ax2.set_xlabel("MATH Difficulty Level")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Drop by Difficulty")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Level {l}" for l in levels])
    ax2.set_ylim(40, 105)
    ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    save_figure(fig, "fig7_selfcorrect_failure")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    print("=" * 60)
    print("ExeVer Figure Generation")
    print("=" * 60)
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Figures dir : {FIGURES_DIR}")
    print()

    # Load all result files
    print("Loading result files...")
    exp4 = load_json("exp4_repair_baselines.json")
    exp5 = load_json("exp5_math500_full.json")
    exp7 = load_json("exp7_symcode_baseline.json")
    exp8 = load_json("exp8_selfcorrect.json")
    exp9 = load_json("exp9_ablations.json")
    exp10 = load_json("exp10_scaling.json")
    exp11 = load_json("exp11_llm_judge.json")
    exp12 = load_json("exp12_gsm8k.json")
    print()

    generated = 0

    # Figure 1
    print("[Fig 1] Main accuracy comparison...")
    if exp5:
        fig1_accuracy_comparison(exp5, exp7 or {}, exp8 or {}, exp11 or {})
        generated += 1
    else:
        print("  SKIPPED -- exp5 data missing")

    # Figure 2
    print("[Fig 2] Coverage by difficulty level...")
    if exp5:
        fig2_coverage_by_level(exp5, exp12 or {})
        generated += 1
    else:
        print("  SKIPPED -- exp5 data missing")

    # Figure 3
    print("[Fig 3] Accuracy by difficulty level...")
    if exp5:
        fig3_accuracy_by_level(exp5, exp7 or {})
        generated += 1
    else:
        print("  SKIPPED -- exp5 data missing")

    # Figure 4
    print("[Fig 4] Ablation results...")
    if exp9:
        fig4_ablations(exp9)
        generated += 1
    else:
        print("  SKIPPED -- exp9 data missing")

    # Figure 5
    print("[Fig 5] Scaling analysis...")
    if exp10:
        fig5_scaling(exp10)
        generated += 1
    else:
        print("  SKIPPED -- exp10 data missing")

    # Figure 6
    print("[Fig 6] GSM8K vs MATH coverage comparison...")
    if exp5 and exp12:
        fig6_coverage_comparison(exp5, exp12)
        generated += 1
    else:
        print("  SKIPPED -- exp5 or exp12 data missing")

    # Figure 7
    print("[Fig 7] Self-correction failure...")
    if exp8:
        fig7_selfcorrect_failure(exp8)
        generated += 1
    else:
        print("  SKIPPED -- exp8 data missing")

    print()
    print("=" * 60)
    print(f"Done. Generated {generated}/7 figures.")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Generate PBT figures for the paper.

Produces:
- fig13_fpvr_comparison.pdf — ExeVer vs PBT FPVR (the headline result)
- fig14_pbt_coverage_by_type.pdf — Coverage breakdown by claim type
- fig15_calibration_comparison.pdf — PBT vs ExeVer calibration
- fig16_independence_breakdown.pdf — Fully vs partially independent tests
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_data():
    with open(RESULTS_DIR / "exp15_pbt_math500.json") as f:
        pbt = json.load(f)
    with open(RESULTS_DIR / "audit_results.json") as f:
        exever = json.load(f)
    return pbt, exever


def fig13_fpvr_comparison(pbt, exever):
    """ExeVer vs PBT FPVR — the headline result."""
    fig, ax = plt.subplots(figsize=(5, 4))

    methods = ["ExeVer\n(same-model)", "PBT\n(spec-grounded)"]
    fpvrs = [exever["fpvr"]["fpvr"] * 100, pbt["fpvr"]["overall"] * 100]
    colors = ["#e74c3c", "#2ecc71"]

    bars = ax.bar(methods, fpvrs, color=colors, edgecolor="black", linewidth=0.5, width=0.5)

    for bar, val in zip(bars, fpvrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("False Positive Verification Rate (%)")
    ax.set_title("FPVR: Same-Model vs Spec-Grounded Checking\n"
                 "(lower is better)")
    ax.set_ylim(0, 20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig13_fpvr_comparison.{fmt}")
    plt.close(fig)
    print("  Saved fig13_fpvr_comparison")


def fig14_coverage_by_type(pbt):
    """Coverage breakdown by claim type."""
    ct_data = pbt["per_claim_type"]
    types = sorted(ct_data.keys())
    counts = [ct_data[t]["count"] for t in types]
    passes = [ct_data[t]["pass"] for t in types]
    fails = [ct_data[t]["fail"] for t in types]
    untested = [c - p - f for c, p, f in zip(counts, passes, fails)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(types))
    width = 0.25

    ax.bar(x - width, passes, width, label="PASS", color="#2ecc71", edgecolor="black", linewidth=0.3)
    ax.bar(x, fails, width, label="FAIL", color="#e74c3c", edgecolor="black", linewidth=0.3)
    ax.bar(x + width, untested, width, label="Untested", color="#95a5a6", edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Claim Type")
    ax.set_ylabel("Number of Steps")
    ax.set_title("PBT Results by Claim Type")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=8)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig14_pbt_coverage_by_type.{fmt}")
    plt.close(fig)
    print("  Saved fig14_pbt_coverage_by_type")


def fig15_calibration_comparison(pbt, exever):
    """PBT vs ExeVer calibration."""
    fig, ax = plt.subplots(figsize=(6, 4))

    methods = ["ExeVer\nALL_PASS", "ExeVer\nFallback", "PBT\nALL_PASS", "PBT\nNon-pass"]
    accs = [
        exever["fpvr"]["calibration"]["all_pass_accuracy"] * 100,
        exever["fpvr"]["calibration"]["fallback_accuracy"] * 100,
        pbt["calibration"]["all_pass_accuracy"] * 100,
        pbt["calibration"]["non_pass_accuracy"] * 100,
    ]
    colors = ["#3498db", "#3498db", "#e67e22", "#e67e22"]
    alphas = [1.0, 0.5, 1.0, 0.5]

    bars = ax.bar(methods, accs, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # Gap annotations
    ax.annotate(f"+{exever['fpvr']['calibration']['gap_pp']}pp",
                xy=(0.5, 82), fontsize=10, ha="center", color="#3498db")
    ax.annotate(f"+{pbt['calibration']['gap_pp']}pp",
                xy=(2.5, 85), fontsize=10, ha="center", color="#e67e22")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Calibration: Verification Verdict as Confidence Signal")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig15_calibration_comparison.{fmt}")
    plt.close(fig)
    print("  Saved fig15_calibration_comparison")


def fig16_independence_breakdown(pbt):
    """Independence breakdown of PBT tests."""
    fig, ax = plt.subplots(figsize=(5, 4))

    labels = ["Fully\nIndependent", "Partially\nIndependent", "Untestable"]
    sizes = [
        pbt["coverage"]["total_fully_independent"],
        pbt["coverage"]["total_partially_independent"],
        pbt["coverage"]["total_untestable"],
    ]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    ax.set_title(f"PBT Test Independence ({sum(sizes)} total steps)")

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig16_independence_breakdown.{fmt}")
    plt.close(fig)
    print("  Saved fig16_independence_breakdown")


def main():
    print("Generating PBT figures...")
    pbt, exever = load_data()

    fig13_fpvr_comparison(pbt, exever)
    fig14_coverage_by_type(pbt)
    fig15_calibration_comparison(pbt, exever)
    fig16_independence_breakdown(pbt)

    print("\nAll PBT figures generated.")


if __name__ == "__main__":
    main()

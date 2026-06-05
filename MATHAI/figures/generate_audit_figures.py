"""Generate figures from the comprehensive audit.

Produces:
- fig10_fpvr_by_level.pdf — FPVR monotonically increasing with difficulty
- fig11_calibration_by_level.pdf — ALL_PASS vs fallback accuracy by level
- fig12_repair_distribution.pdf — Repair cases by difficulty level
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent

# Style
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
    with open(RESULTS_DIR / "audit_results.json") as f:
        audit = json.load(f)
    with open(RESULTS_DIR / "exp5_math500_full.json") as f:
        exp5 = json.load(f)
    return audit, exp5


def fig10_fpvr_by_level(audit):
    """FPVR by difficulty level — the key measurement."""
    fpvr_data = audit["fpvr"]["fpvr_by_level"]
    levels = sorted(int(k) for k in fpvr_data.keys())
    fpvr_vals = [fpvr_data[str(lv)]["fpvr"] for lv in levels]
    n_vals = [fpvr_data[str(lv)]["n"] for lv in levels]

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(levels, [v * 100 for v in fpvr_vals],
                  color=["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"],
                  edgecolor="black", linewidth=0.5, width=0.6)

    # Add value labels
    for bar, val, n in zip(bars, fpvr_vals, n_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val*100:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("MATH Difficulty Level")
    ax.set_ylabel("FPVR (%)")
    ax.set_title("False Positive Verification Rate by Difficulty\n"
                 "(fraction of ALL_PASS cases with wrong answers)")
    ax.set_xticks(levels)
    ax.set_ylim(0, 40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig10_fpvr_by_level.{fmt}")
    plt.close(fig)
    print("  Saved fig10_fpvr_by_level")


def fig11_calibration_by_level(audit, exp5):
    """Calibration: ALL_PASS accuracy vs fallback accuracy by level."""
    results = exp5["exever_results"]

    levels = [1, 2, 3, 4, 5]
    all_pass_acc = []
    fallback_acc = []

    for lv in levels:
        lv_results = [r for r in results if r["level"] == lv]
        lv_pass = [r for r in lv_results if r["verdict"] == "ALL_PASS"]
        lv_fall = [r for r in lv_results if r["verdict"] != "ALL_PASS"]

        if lv_pass:
            all_pass_acc.append(
                sum(1 for r in lv_pass if r["answer_correct"]) / len(lv_pass) * 100
            )
        else:
            all_pass_acc.append(0)

        if lv_fall:
            fallback_acc.append(
                sum(1 for r in lv_fall if r["answer_correct"]) / len(lv_fall) * 100
            )
        else:
            fallback_acc.append(0)

    x = np.arange(len(levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width/2, all_pass_acc, width, label="ALL_PASS",
                   color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, fallback_acc, width, label="Fallback",
                   color="#e74c3c", edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("MATH Difficulty Level")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SymPy Consistency as a Confidence Signal\n"
                 "ALL_PASS accuracy vs Fallback accuracy by difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{lv}" for lv in levels])
    ax.legend(frameon=False)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add gap annotation
    overall_gap = audit["fpvr"]["calibration"]["gap_pp"]
    ax.annotate(
        f"Overall gap: +{overall_gap:.1f}pp",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig11_calibration_by_level.{fmt}")
    plt.close(fig)
    print("  Saved fig11_calibration_by_level")


def fig12_repair_distribution(audit):
    """Repair cases by difficulty — shows concentration at easy problems."""
    repair_by_level = audit["repair_audit"]["repair_by_level"]
    fp_by_level = audit["repair_audit"]["false_positives_all_pass"]["by_level"]
    fn_by_level = audit["repair_audit"]["false_negatives_no_verdict"]["by_level"]

    levels = [1, 2, 3, 4, 5]
    repairs = [repair_by_level.get(str(lv), 0) for lv in levels]
    fps = [fp_by_level.get(str(lv), 0) for lv in levels]
    fns = [fn_by_level.get(str(lv), 0) for lv in levels]

    x = np.arange(len(levels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width, repairs, width, label="Repaired (correct)",
           color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x, fps, width, label="False positives (ALL_PASS wrong)",
           color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, fns, width, label="False negatives (crash, wrong)",
           color="#95a5a6", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("MATH Difficulty Level")
    ax.set_ylabel("Count")
    ax.set_title("Verification Outcome Distribution by Difficulty\n"
                 "Repairs concentrated at easy levels; false positives at hard levels")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{lv}" for lv in levels])
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig12_repair_distribution.{fmt}")
    plt.close(fig)
    print("  Saved fig12_repair_distribution")


def main():
    print("Generating audit figures...")
    audit, exp5 = load_data()

    fig10_fpvr_by_level(audit)
    fig11_calibration_by_level(audit, exp5)
    fig12_repair_distribution(audit)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()

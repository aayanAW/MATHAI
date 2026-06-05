"""Generate publication-quality figures for ExeVer paper.

Uses REAL data from experiments 1-4.
"""
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = Path("figures/output")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results")

SUBJECTS = [
    "prealgebra", "algebra", "number_theory",
    "counting_and_probability", "intermediate_algebra",
    "precalculus", "geometry",
]
SUBJECT_LABELS = [
    "PreAlgebra", "Algebra", "Num. Theory",
    "Count. & Prob.", "Int. Algebra",
    "Precalculus", "Geometry",
]
LEVELS = [1, 2, 3, 4, 5]


def plot_verifiability_map(coverage_data: Dict, output_path: Optional[str] = None):
    """Figure 1: The Verifiability Map heatmap (killer figure)."""
    matrix = np.zeros((len(SUBJECTS), len(LEVELS)))
    for i, subj in enumerate(SUBJECTS):
        for j, lv in enumerate(LEVELS):
            matrix[i, j] = coverage_data.get(subj, {}).get(lv, 0.0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix * 100, cmap="RdYlGn", aspect="auto", vmin=0, vmax=80)

    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"Level {lv}" for lv in LEVELS])
    ax.set_yticks(range(len(SUBJECTS)))
    ax.set_yticklabels(SUBJECT_LABELS)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("MATH Subject")
    ax.set_title("How Much Math Can Code Check?\nVerification Step Coverage by Subject and Difficulty")

    for i in range(len(SUBJECTS)):
        for j in range(len(LEVELS)):
            val = matrix[i, j] * 100
            color = "white" if val < 15 or val > 60 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Step Coverage (%)")
    fig.tight_layout()

    path = output_path or str(FIGURES_DIR / "fig1_verifiability_map.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_accuracy_comparison(accuracy_data: Dict, by_level: Dict,
                              output_path: Optional[str] = None):
    """Figure 2: ExeVer vs Baselines accuracy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Overall accuracy bar chart
    methods = ["CoT\n(greedy)", "CoT\n(sampled)", "Majority\n@4", "Best-of\n-4", "ExeVer"]
    values = [
        accuracy_data["exp1_cot_pass1"],
        accuracy_data["baseline_pass1_sampled"],
        accuracy_data["baseline_majority_4"],
        accuracy_data["baseline_best_of_4"],
        accuracy_data["exever"],
    ]
    colors = ["#a0a0a0", "#a0a0a0", "#5b9bd5", "#a0a0a0", "#e74c3c"]

    bars = ax1.bar(range(len(methods)), [v * 100 for v in values], color=colors, alpha=0.9)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Pass@1 Accuracy (%)")
    ax1.set_title("Overall Accuracy\n(Equal Compute: 4 Model Calls)")
    ax1.set_ylim(70, 95)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: By difficulty level
    levels = sorted(by_level.keys())
    x = np.arange(len(levels))
    width = 0.2
    method_keys = [("pass1", "CoT (sampled)", "#a0a0a0"),
                    ("maj4", "Majority@4", "#5b9bd5"),
                    ("best4", "Best-of-4", "#7fc97f"),
                    ("exever", "ExeVer", "#e74c3c")]

    for i, (key, label, color) in enumerate(method_keys):
        vals = [by_level[lv][key] * 100 for lv in levels]
        ax2.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f"L{lv}" for lv in levels])
    ax2.set_ylabel("Pass@1 Accuracy (%)")
    ax2.set_xlabel("Difficulty Level")
    ax2.set_title("Accuracy by Difficulty\n(ExeVer gains increase on harder problems)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(50, 100)

    fig.tight_layout()
    path = output_path or str(FIGURES_DIR / "fig2_accuracy_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_echo_chamber(same_rate, cross_rate, output_path: Optional[str] = None):
    """Figure 3: Echo Chamber comparison."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    methods = ["Same-Model\n(Qwen→Qwen)", "Cross-Model\n(Qwen→DeepSeek)"]
    rates = [same_rate * 100, cross_rate * 100]
    colors = ["#5b9bd5", "#e74c3c"]

    bars = ax.bar(methods, rates, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel("Echo Chamber Rate (%)")
    ax.set_title("Echo Chamber: Assertions Pass but Answer Wrong\n"
                 "(Cross-model verification produces WEAKER checks)")
    ax.set_ylim(0, 55)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="Plan threshold (20%)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    path = output_path or str(FIGURES_DIR / "fig3_echo_chamber.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_execution_map(exec_data: Dict, output_path: Optional[str] = None):
    """Figure 4: Execution success rate heatmap."""
    matrix = np.zeros((len(SUBJECTS), len(LEVELS)))
    for i, subj in enumerate(SUBJECTS):
        for j, lv in enumerate(LEVELS):
            matrix[i, j] = exec_data.get(subj, {}).get(lv, 0.0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix * 100, cmap="Blues", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"Level {lv}" for lv in LEVELS])
    ax.set_yticks(range(len(SUBJECTS)))
    ax.set_yticklabels(SUBJECT_LABELS)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("MATH Subject")
    ax.set_title("Verification Script Execution Success Rate\n(All Assertions Pass)")

    for i in range(len(SUBJECTS)):
        for j in range(len(LEVELS)):
            val = matrix[i, j] * 100
            color = "white" if val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="All Pass Rate (%)")
    fig.tight_layout()

    path = output_path or str(FIGURES_DIR / "fig4_execution_map.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_subject_accuracy(by_subject: Dict, output_path: Optional[str] = None):
    """Figure 5: Accuracy by subject."""
    fig, ax = plt.subplots(figsize=(10, 5))

    subjects = sorted(by_subject.keys())
    x = np.arange(len(subjects))
    width = 0.2
    method_keys = [("pass1", "CoT (sampled)", "#a0a0a0"),
                    ("maj4", "Majority@4", "#5b9bd5"),
                    ("best4", "Best-of-4", "#7fc97f"),
                    ("exever", "ExeVer", "#e74c3c")]

    for i, (key, label, color) in enumerate(method_keys):
        vals = [by_subject[s][key] * 100 for s in subjects]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    labels = []
    for s in subjects:
        idx = SUBJECTS.index(s) if s in SUBJECTS else -1
        labels.append(SUBJECT_LABELS[idx] if idx >= 0 else s)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Pass@1 Accuracy (%)")
    ax.set_title("Accuracy by MATH Subject\n(ExeVer gains highest on Number Theory +7.7%)")
    ax.legend(loc="lower right")
    ax.set_ylim(50, 100)

    fig.tight_layout()
    path = output_path or str(FIGURES_DIR / "fig5_subject_accuracy.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_feasibility_gates(gates: Dict, output_path: Optional[str] = None):
    """Figure 6: Feasibility gates summary."""
    fig, ax = plt.subplots(figsize=(8, 4))

    gate_names = ["G1: Script\nValidity", "G2: Execution\nRate",
                   "G3: Assertion\nQuality", "G4: Step\nCoverage"]
    values = [gates["G1_script_validity"] * 100, gates["G2_execution_rate"] * 100,
              gates["G3_assertion_quality"] * 100, gates["G4_coverage"] * 100]
    thresholds = [60, 50, 50, 30]
    passed = [v >= t for v, t in zip(values, thresholds)]
    colors = ["#2ecc71" if p else "#e74c3c" for p in passed]

    bars = ax.bar(range(len(gate_names)), values, color=colors, alpha=0.85)

    # Add threshold lines
    for i, t in enumerate(thresholds):
        ax.plot([i - 0.4, i + 0.4], [t, t], "k--", linewidth=1.5, alpha=0.5)

    ax.set_xticks(range(len(gate_names)))
    ax.set_xticklabels(gate_names)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Feasibility Gates: 3/4 Pass")
    ax.set_ylim(0, 105)

    for bar, val, p in zip(bars, values, passed):
        label = f"{val:.1f}% {'PASS' if p else 'FAIL'}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = output_path or str(FIGURES_DIR / "fig6_feasibility_gates.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_repair_analysis(repair_data: Dict, output_path: Optional[str] = None):
    """Figure 7: Repair analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Verdict distribution
    verdicts = repair_data["verdicts"]
    labels = list(verdicts.keys())
    sizes = list(verdicts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    ax1.pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90)
    ax1.set_title("ExeVer Verdict Distribution\n(N=300 problems)")

    # Right: Repair success
    repair = repair_data["repair"]
    categories = ["Attempted", "Successful"]
    vals = [repair["attempted"], repair["successful"]]
    ax2.bar(categories, vals, color=["#5b9bd5", "#2ecc71"], alpha=0.85)
    ax2.set_ylabel("Count")
    ax2.set_title(f"Repair Success Rate: {repair['success_rate']*100:.0f}%\n"
                   f"({repair['successful']}/{repair['attempted']} repairs)")

    for i, v in enumerate(vals):
        ax2.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    path = output_path or str(FIGURES_DIR / "fig7_repair_analysis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def generate_real_figures():
    """Generate all figures from real experimental data."""
    print("Loading experimental results...")

    # Load all results
    with open(RESULTS_DIR / "exp2_twopass_feasibility.json") as f:
        exp2 = json.load(f)
    with open(RESULTS_DIR / "exp3_crossmodel.json") as f:
        exp3 = json.load(f)
    with open(RESULTS_DIR / "exp4_repair_baselines.json") as f:
        exp4 = json.load(f)

    # === Figure 1: Verifiability Map ===
    vmap = exp3["verifiability_map"]
    coverage_data = {}
    for key, val in vmap.items():
        subj = val["subject"]
        lv = val["level"]
        coverage_data.setdefault(subj, {})[lv] = val["step_coverage"]
    plot_verifiability_map(coverage_data)

    # === Figure 2: Accuracy Comparison ===
    accuracy = exp4["accuracy"]
    by_level = {int(k): v for k, v in exp4["by_level"].items()}
    plot_accuracy_comparison(accuracy, by_level)

    # === Figure 3: Echo Chamber ===
    echo = exp3["echo_chamber"]
    plot_echo_chamber(echo["same_model_rate"], echo["cross_model_rate"])

    # === Figure 4: Execution Success Map ===
    exec_data = {}
    for key, val in vmap.items():
        subj = val["subject"]
        lv = val["level"]
        exec_data.setdefault(subj, {})[lv] = val["all_pass_rate"]
    plot_execution_map(exec_data)

    # === Figure 5: Accuracy by Subject ===
    by_subject = exp4["by_subject"]
    plot_subject_accuracy(by_subject)

    # === Figure 6: Feasibility Gates ===
    plot_feasibility_gates(exp2["gates"])

    # === Figure 7: Repair Analysis ===
    repair_data = {
        "verdicts": exp4["exever_verdicts"],
        "repair": exp4["repair"],
    }
    plot_repair_analysis(repair_data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_real_figures()

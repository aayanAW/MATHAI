"""Generate all matplotlib figures for the DAJV paper.

Outputs:
  paper/figures/fig_dependency_heatmap.pdf
  paper/figures/fig_risk_coverage.pdf
  paper/figures/fig_reliability.pdf
  paper/figures/fig_theorem1_tightness.pdf
  paper/figures/fig_sample_complexity.pdf

All figures are vector PDF, single-column width by default.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_majority, naive_unanimous
from verifyensemble.evaluation.ece import reliability_diagram
from verifyensemble.evaluation.risk_coverage import risk_coverage_curve
from verifyensemble.theory import (
    required_n,
)
from verifyensemble.utils.io import align_extractor_caches

GROUP_B = {
    "gpt-oss-120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "gpt-5-mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "claude-sonnet-4-6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "Qwen3-Coder-480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}
FIG_DIR = HERE.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# Clean black-and-white plotting style (matches ICML conventions)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Figure 1: dependency heatmap (Cohen's kappa, math175 wrong subset)
# ---------------------------------------------------------------------------
def fig_dependency_heatmap() -> None:
    with open(HERE.parent / "artifacts/dependency_matrix_B_math175.json") as f:
        d = json.load(f)
    kappa = np.array(d["kappa"])
    labels = d["extractor_ids"]
    short = [l.split("_", 1)[1].replace("_", "-") for l in labels]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(kappa, vmin=0, vmax=1, cmap="Greys", aspect="auto")
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt_color = "white" if kappa[i][j] > 0.5 else "black"
            ax.text(j, i, f"{kappa[i][j]:.2f}",
                    ha="center", va="center", color=txt_color, fontsize=7)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title("Pairwise Cohen's $\\kappa$ on math175 wrong subset",
                 fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dependency_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'fig_dependency_heatmap.pdf'}")


# ---------------------------------------------------------------------------
# Figure 2: joint-FP vs independence-bound scatter
# ---------------------------------------------------------------------------
def fig_joint_fp_vs_indep() -> None:
    with open(HERE.parent / "artifacts/dependency_matrix_B_math175.json") as f:
        d = json.load(f)
    n = len(d["extractor_ids"])
    pairs_indep = []
    pairs_joint = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_indep.append(d["indep_bound"][i][j])
            pairs_joint.append(d["joint_fp"][i][j])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pairs_indep = np.array(pairs_indep)
    pairs_joint = np.array(pairs_joint)
    ax.scatter(pairs_indep, pairs_joint, c="black", s=18, zorder=3)
    lim = max(pairs_indep.max(), pairs_joint.max()) * 1.2
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=0.8,
            label="independence ($y = x$)")
    ax.set_xlabel("Independence-bound joint-FP prediction")
    ax.set_ylabel("Empirical joint-FP")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Independence is violated: joint-FP $\\gg$ independence (math175)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_joint_fp_vs_indep.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'fig_joint_fp_vs_indep.pdf'}")


# ---------------------------------------------------------------------------
# Figure 3: risk-coverage curve on math175 test split
# ---------------------------------------------------------------------------
def fig_risk_coverage() -> None:
    aligned = align_extractor_caches(GROUP_B, bench="math175")
    accept = aligned["accept"]; correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids); n = len(correct)
    import random
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]
    cal_correct = [correct[j] for j in cal_idx]
    test_correct = [correct[j] for j in test_idx]

    dajv = DajvCalibration.fit(accept_cal, cal_correct, extractor_ids)

    methods = [
        ("Naive unanimous",
         lambda v: naive_unanimous(v),
         "black", "-"),
        ("Naive majority",
         lambda v: naive_majority(v),
         "gray", "--"),
        ("DAJV (ours)",
         lambda v: dajv_aggregate(v, dajv),
         "black", ":"),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for name, fn, color, ls in methods:
        scores, ys = [], []
        for j in range(len(test_idx)):
            v = [accept_test[i][j] for i in range(k)]
            res = fn(v)
            p = res.get("P_correct")
            if p is None:
                continue
            scores.append(p)
            ys.append(test_correct[j])
        curve = risk_coverage_curve(scores, ys)
        ax.plot(curve["coverage"], curve["precision"], label=name,
                color=color, linestyle=ls, lw=1.4)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Precision @ coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("Risk-coverage on math175 test split ($n = 53$)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_risk_coverage.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'fig_risk_coverage.pdf'}")


# ---------------------------------------------------------------------------
# Figure 4: reliability diagram (DAJV vs naive)
# ---------------------------------------------------------------------------
def fig_reliability() -> None:
    aligned = align_extractor_caches(GROUP_B, bench="math175")
    accept = aligned["accept"]; correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids); n = len(correct)
    import random
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]
    cal_correct = [correct[j] for j in cal_idx]
    test_correct = [correct[j] for j in test_idx]

    dajv = DajvCalibration.fit(accept_cal, cal_correct, extractor_ids)

    def collect(method_fn):
        confs, ys = [], []
        for j in range(len(test_idx)):
            v = [accept_test[i][j] for i in range(k)]
            res = method_fn(v)
            p = res.get("P_correct")
            if p is None: continue
            confs.append(p); ys.append(test_correct[j])
        return confs, ys

    confs_n, ys_n = collect(naive_unanimous)
    confs_d, ys_d = collect(lambda v: dajv_aggregate(v, dajv))

    rd_n = reliability_diagram(confs_n, ys_n, n_bins=5)
    rd_d = reliability_diagram(confs_d, ys_d, n_bins=5)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.plot([0, 1], [0, 1], "--", color="lightgray", lw=0.8)
    def _plot(rd, label, marker, color):
        xs = [m for m in rd["mean_confidence"] if m is not None]
        ys = [a for a in rd["accuracy"] if a is not None]
        ax.plot(xs, ys, marker=marker, color=color, label=label,
                markersize=5, linestyle="-", lw=0.8)
    _plot(rd_n, "Naive unanimous", "s", "gray")
    _plot(rd_d, "DAJV (ours)",     "o", "black")
    ax.set_xlabel("Predicted P(correct)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Reliability diagram (5 bins, math175 test)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'fig_reliability.pdf'}")


# ---------------------------------------------------------------------------
# Figure 5: Theorem 1 tightness curve
# ---------------------------------------------------------------------------
def fig_theorem1_tightness() -> None:
    with open(HERE.parent / "artifacts/theorem1_validation.json") as f:
        d = json.load(f)
    results = d["results"]

    # Group by (k, pi); x-axis rho, y-axis bound and empirical
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    for ax, k in zip(axes, [2, 5, 12]):
        for pi_val, color in zip([0.05, 0.10, 0.20], ["black", "gray", "lightgray"]):
            sub = [r for r in results if r["k"] == k and abs(r["pi"] - pi_val) < 1e-9]
            sub.sort(key=lambda r: r["rho"])
            rhos = [r["rho"] for r in sub]
            emp = [r["empirical"] for r in sub]
            bound = [r["dajv_bound"] for r in sub]
            ax.plot(rhos, bound, "-",  color=color,
                    label=f"DAJV bound ($\\pi={pi_val}$)", lw=1.0)
            ax.plot(rhos, emp,   "o",  color=color, markersize=4,
                    label=f"empirical ($\\pi={pi_val}$)")
        ax.set_xlabel("$\\rho$")
        ax.set_title(f"$k = {k}$", fontsize=9)
        if k == 2:
            ax.set_ylabel("Joint acceptance")
        if k == 2:
            ax.legend(fontsize=6, frameon=False, ncol=1, loc="upper left")
    fig.suptitle("Theorem 1 tightness: empirical vs DAJV upper bound",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_theorem1_tightness.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'fig_theorem1_tightness.pdf'}")


# ---------------------------------------------------------------------------
# Figure 6: sample-complexity curve (Theorem 2)
# ---------------------------------------------------------------------------
def fig_sample_complexity() -> None:
    eps_grid = np.linspace(0.02, 0.30, 30)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, ls in [(4, "-"), (12, "--"), (20, ":")]:
        ns = [required_n(k, e, delta=0.05) for e in eps_grid]
        ax.semilogy(eps_grid, ns, ls, color="black", lw=1.4, label=f"$k = {k}$")
    ax.set_xlabel("Entrywise error tolerance $\\varepsilon$")
    ax.set_ylabel("Required calibration $n$")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Theorem 2: required $n$ vs $\\varepsilon$ at $\\delta = 0.05$",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sample_complexity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'fig_sample_complexity.pdf'}")


def main() -> None:
    print("Generating DAJV paper figures...")
    fig_dependency_heatmap()
    fig_joint_fp_vs_indep()
    fig_risk_coverage()
    fig_reliability()
    fig_theorem1_tightness()
    fig_sample_complexity()
    print("All figures written.")


if __name__ == "__main__":
    main()

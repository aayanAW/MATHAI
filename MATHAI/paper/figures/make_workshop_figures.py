"""Build 2 ICML-style matplotlib figures for the AI4Math workshop paper.

The pipeline schematic (Fig 1) is built as TikZ directly in workshop.tex.
This script produces Fig 2 (precision-coverage) and Fig 3 (consensus
convergence). Palette and vocabulary adapted from ThinkPRM and the
semantic-entropy papers: saturated orange / teal / dark-blue / dashed-gray.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
FIG = Path("/Users/aayanalwani/MATHAI/MATHAI/paper/figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 7.2,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.03,
})

ORANGE = "#E8743B"   # proposed (consensus)
TEAL   = "#2CA089"   # single-extractor X-SGRV
NAVY   = "#2E5A8B"   # semantic entropy SymPy
PLUM   = "#8E4DA8"   # semantic entropy NLI
GRAY   = "#7A7A7A"
LIGHTG = "#D4D4D4"


def load(name: str):
    with open(RESULTS / name) as f:
        return json.load(f)


def cp(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


# =====================================================================
# Figure 2 — Precision-coverage curves (two panels)
# =====================================================================

def pc_curve(scores: np.ndarray, correct: np.ndarray,
             lower_is_better: bool = True, min_k: int = 5, n_points: int = 40):
    n = len(scores)
    order = np.argsort(scores) if lower_is_better else np.argsort(-scores)
    c = correct[order].astype(bool)
    ks = np.unique(np.linspace(min_k, n, n_points).astype(int))
    covs, precs, los, his = [], [], [], []
    for k in ks:
        nc = int(c[:k].sum())
        covs.append(k / n); precs.append(nc / k)
        l, h = cp(nc, k)
        los.append(l); his.append(h)
    return np.array(covs), np.array(precs), np.array(los), np.array(his)


def fig2_precision_coverage():
    se = load("exp35_semantic_entropy.json")
    rows = se["rows"]

    def ent(r, k):
        v = r.get(k, {}) or {}
        return float(v.get("entropy", 0.0)) if isinstance(v, dict) else 0.0

    def ncl(r, k):
        v = r.get(k, {}) or {}
        return int(v.get("n_clusters", 10)) if isinstance(v, dict) else 10

    def curves_for(bench):
        br = [r for r in rows if r.get("bench") == bench]
        correct = np.array([int(bool(r.get("plurality_correct"))) for r in br])
        se_math = np.array([ent(r, "se_math") for r in br])
        se_nli  = np.array([ent(r, "se_nli")  for r in br])
        n_clus  = np.array([ncl(r, "se_math") for r in br]).astype(float)
        return {
            "SE (SymPy)": pc_curve(se_math, correct),
            "SE (NLI)":   pc_curve(se_nli,  correct),
            "Self-cons.": pc_curve(n_clus,  correct),
        }, len(br)

    math_curves, n_math = curves_for("math175")
    clean_curves, n_clean = curves_for("cleanmath")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.98, top=0.88, bottom=0.17)

    def draw(ax, cov, prec, lo, hi, color, label, ls="-", marker=None, mevery=4):
        ax.plot(cov, prec, color=color, ls=ls, lw=1.2, label=label,
                marker=marker, markersize=3.2, markevery=mevery,
                mec="white", mew=0.3)
        ax.fill_between(cov, lo, hi, color=color, alpha=0.12, lw=0)

    # ---- Panel A: MATH-175 ----
    for k, color, ls, marker in [
        ("SE (SymPy)", NAVY, "-",  "o"),
        ("SE (NLI)",   PLUM, "-",  "s"),
        ("Self-cons.", GRAY, "--", None),
    ]:
        draw(axA, *math_curves[k], color=color, label=k, ls=ls, marker=marker)

    # X-SGRV points (MATH-500 stratified numbers, matching paper Table)
    axA.errorbar(0.549, 90/96, yerr=[[90/96 - cp(90,96)[0]], [cp(90,96)[1] - 90/96]],
                 fmt="o", color=TEAL, markersize=7, mec="white", mew=0.7,
                 elinewidth=1.0, capsize=2.5, label="X-SGRV (1 extractor)", zorder=10)
    axA.errorbar(0.417, 1.0, yerr=[[1.0 - cp(73,73)[0]], [0]],
                 fmt="*", color=ORANGE, markersize=13, mec="white", mew=0.7,
                 elinewidth=1.0, capsize=2.5, label="Consensus (2-way)", zorder=11)
    axA.errorbar(0.394, 1.0, yerr=[[1.0 - cp(69,69)[0]], [0]],
                 fmt="*", color=ORANGE, markersize=9, alpha=0.55, mec="white", mew=0.5,
                 elinewidth=0.7, capsize=1.5, zorder=9)

    axA.annotate("171/171 at $n{=}498$\n(pre-reg.)", xy=(0.417, 1.0),
                 xytext=(0.55, 0.60), fontsize=6.8, color=ORANGE,
                 ha="left",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5, alpha=0.8))

    axA.set_xlim(0, 1); axA.set_ylim(0, 1.05)
    axA.set_xlabel("Coverage"); axA.set_ylabel("Top-tier precision")
    axA.set_title("(a) MATH-500 stratified", loc="left", fontsize=9.5)
    axA.grid(axis="y", alpha=0.15, lw=0.5)
    axA.legend(loc="lower left", handlelength=1.6, borderpad=0.2,
               labelspacing=0.2, bbox_to_anchor=(0.02, 0.02))

    # ---- Panel B: CleanMath ----
    for k, color, ls, marker in [
        ("SE (SymPy)", NAVY, "-",  "o"),
        ("SE (NLI)",   PLUM, "-",  "s"),
        ("Self-cons.", GRAY, "--", None),
    ]:
        draw(axB, *clean_curves[k], color=color, label=k, ls=ls, marker=marker)

    # X-SGRV CleanMath (Qwen solver) and solver rotation (DeepSeek-V3.1)
    axB.errorbar(0.032, 1.0, yerr=[[1.0 - cp(4,4)[0]], [0]],
                 fmt="o", color=TEAL, markersize=7, mec="white", mew=0.7,
                 elinewidth=1.0, capsize=2.5, label="X-SGRV (Qwen solver)", zorder=10)
    axB.errorbar(0.120, 1.0, yerr=[[1.0 - cp(15,15)[0]], [0]],
                 fmt="*", color=ORANGE, markersize=13, mec="white", mew=0.7,
                 elinewidth=1.0, capsize=2.5, label="X-SGRV (DS-V3.1 solver)", zorder=11)

    axB.annotate("15/15, CI [0.78, 1.00]\nsolver rotation",
                 xy=(0.120, 1.0), xytext=(0.30, 0.68),
                 fontsize=6.8, color=ORANGE, ha="left",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5, alpha=0.8))

    # NLI-SE collapse marker on CleanMath: large zero-entropy tier with low precision.
    # 59/125 problems at entropy ~= 0 with 5 correct (paper Section 9).
    axB.plot(0.472, 5/59, marker="s", color=PLUM, markersize=7, mec="white", mew=0.7, zorder=8)
    axB.annotate("NLI-SE zero-entropy tier:\n5/59 = 8.5%",
                 xy=(0.472, 5/59), xytext=(0.55, 0.30),
                 fontsize=6.8, color=PLUM, ha="left",
                 arrowprops=dict(arrowstyle="-", color=PLUM, lw=0.5, alpha=0.8))

    axB.set_xlim(0, 1); axB.set_ylim(0, 1.05)
    axB.set_xlabel("Coverage")
    axB.set_title("(b) CleanMath (post-cutoff)", loc="left", fontsize=9.5)
    axB.grid(axis="y", alpha=0.15, lw=0.5)
    axB.legend(loc="center right", handlelength=1.6, borderpad=0.2,
               labelspacing=0.2, bbox_to_anchor=(1.0, 0.52))

    out = FIG / "fig_workshop_precision_coverage.pdf"
    plt.savefig(out); plt.close()
    print(f"wrote {out}")


# =====================================================================
# Figure 3 — Consensus convergence (K extractors vs precision & coverage)
# =====================================================================

def fig3_consensus():
    # ---- Panel A: MATH-500 stratified n=175 (from Table 7 of paper) ----
    # Separate rows per (K, extractor) for individual K=1 points
    pts_175 = [
        # K=1 individual extractors (all 6 from Table 7 of the NeurIPS draft)
        (1, "Llama-3.3-70B",        96, 90),
        (1, "DeepSeek-V3",          88, 82),
        (1, "Claude Sonnet 4.6",   112,105),
        (1, "gpt-oss-120b",        112,105),
        (1, "Qwen3-Coder-480B",     94, 87),
        (1, "GPT-5-mini",          100, 93),
        # K>=2 consensus rows
        (2, "Llama ∩ DeepSeek",     73, 73),
        (4, "4-way",                72, 72),
        (5, "5-way",                69, 69),
        (6, "6-way",                67, 67),
    ]

    # ---- Panel B: pre-registered full n=498 ----
    pts_498 = [
        ("Llama-70B raw",    243, 230, 0.488),
        ("DeepSeek-V3 raw",  207, 199, 0.416),
        ("Consensus strict", 171, 171, 0.343),
        ("Consensus loose",  181, 180, 0.363),
    ]

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.35,
                           left=0.07, right=0.96, top=0.86, bottom=0.20)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # --- Panel A ---
    # Precision on left y, coverage on right y.
    axA2 = axA.twinx(); axA2.spines["top"].set_visible(False)

    # aggregate over extractors at each K for a smooth median curve
    Ks = sorted({k for k, *_ in pts_175})
    med_prec, med_cov, ci_lo, ci_hi = [], [], [], []
    for K in Ks:
        rows = [p for p in pts_175 if p[0] == K]
        # pool
        kc = sum(p[3] for p in rows); nt = sum(p[2] for p in rows)
        med_prec.append(kc / nt)
        med_cov.append(nt / (175 * len(rows)))  # mean coverage
        l, h = cp(kc, nt)
        ci_lo.append(l); ci_hi.append(h)

    # Coverage bars (right axis)
    axA2.bar(Ks, med_cov, color=LIGHTG, alpha=0.7, width=0.55,
             edgecolor="none", zorder=1, label="Coverage")
    axA2.set_ylim(0, 0.75)
    axA2.set_ylabel("Coverage", color="#666666", fontsize=8.5)
    axA2.tick_params(axis="y", colors="#666666", labelsize=8)

    # Individual-extractor K=1 points (scatter, slightly jittered)
    for i, (K, name, nt, nc) in enumerate(pts_175):
        if K != 1:
            continue
        x_jit = 1 + (i - 2) * 0.08
        axA.plot(x_jit, nc / nt, marker="o", color=TEAL, markersize=5,
                 mec="white", mew=0.5, zorder=3, alpha=0.85)

    # Precision curve (left axis): line through pooled points
    axA.fill_between(Ks, ci_lo, ci_hi, color=ORANGE, alpha=0.15, lw=0, zorder=2)
    axA.plot(Ks, med_prec, marker="*", color=ORANGE, lw=1.5, markersize=12,
             mec="white", mew=0.7, zorder=5, label="Precision (pooled)")

    axA.set_ylim(0.85, 1.02)
    axA.set_xlim(0.4, 6.6)
    axA.set_xticks(Ks)
    axA.set_xlabel("Number of consensus extractors  $K$")
    axA.set_ylabel("Top-tier precision", color=ORANGE)
    axA.tick_params(axis="y", colors=ORANGE)
    axA.set_title("(a) MATH-500 stratified ($n{=}175$, 6 extractors)", loc="left", fontsize=9.5)

    axA.annotate("perfect precision\nfrom $K{=}2$ onward", xy=(2, 1.0),
                 xytext=(3.2, 0.905), fontsize=7.2, color=ORANGE,
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6, alpha=0.8))
    axA.annotate("single-extractor\n93.5\\% pooled",
                 xy=(1, med_prec[0]),
                 xytext=(1.3, 0.88), fontsize=7.2, color=TEAL,
                 arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.5, alpha=0.7))

    # --- Panel B: n=498 pre-registered ---
    axB2 = axB.twinx(); axB2.spines["top"].set_visible(False)
    labels_b = [r[0] for r in pts_498]
    xs_b = np.arange(len(pts_498))
    cov_b = [r[3] for r in pts_498]
    prec_b = [r[2] / r[1] for r in pts_498]
    cis_b = [cp(r[2], r[1]) for r in pts_498]
    lo_b = [c[0] for c in cis_b]; hi_b = [c[1] for c in cis_b]

    axB2.bar(xs_b, cov_b, color=LIGHTG, alpha=0.7, width=0.55,
             edgecolor="none", zorder=1)
    axB2.set_ylim(0, 0.6); axB2.set_ylabel("Coverage", color="#666666", fontsize=8.5)
    axB2.tick_params(axis="y", colors="#666666", labelsize=8)

    for x, p, l, h in zip(xs_b, prec_b, lo_b, hi_b):
        axB.errorbar(x, p, yerr=[[p - l], [h - p]], fmt="*", color=ORANGE,
                     markersize=12, mec="white", mew=0.6, elinewidth=1.0,
                     capsize=3, zorder=5)
    axB.plot(xs_b, prec_b, color=ORANGE, lw=1.2, zorder=4, alpha=0.6)

    axB.set_ylim(0.88, 1.02); axB.set_xticks(xs_b)
    axB.set_xticklabels(labels_b, rotation=18, ha="right", fontsize=7.5)
    axB.set_ylabel("Precision", color=ORANGE)
    axB.tick_params(axis="y", colors=ORANGE)
    axB.set_title("(b) Pre-registered MATH-500 ($n{=}498$)", loc="left", fontsize=9.5)
    axB.annotate("171/171\n[0.979, 1.000]", xy=(2, 1.0),
                 xytext=(1.2, 0.925), fontsize=7, color=ORANGE,
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))

    out = FIG / "fig_workshop_consensus_convergence.pdf"
    plt.savefig(out); plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    fig2_precision_coverage()
    fig3_consensus()
    print("done")

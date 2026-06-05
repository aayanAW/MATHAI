"""Risk-coverage curves with bootstrap-style bands across 10 seeds.

For each seed in {1,...,10}, fit DAJV and generate the risk-coverage
curve on the held-out test split. Plot the 10 curves as light traces +
the median curve in bold. Same for naive unanimous.

Output: paper/figures/fig_risk_coverage_bands.pdf
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous
from verifyensemble.evaluation.risk_coverage import risk_coverage_curve
from verifyensemble.utils.io import align_extractor_caches

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}
FIG_DIR = HERE.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def collect_curve(seed, accept, correct, extractor_ids, method_fn) -> tuple[list, list]:
    k = len(extractor_ids); n = len(correct)
    rng = random.Random(seed)
    idx = list(range(n)); rng.shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    cal_correct = [correct[j] for j in cal_idx]
    test_correct = [correct[j] for j in test_idx]
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]
    cal = DajvCalibration.fit(accept_cal, cal_correct, extractor_ids)

    scores, ys = [], []
    for jj in range(len(test_idx)):
        v = [accept_test[i][jj] for i in range(k)]
        res = method_fn(v, cal)
        if res.get("P_correct") is None:
            continue
        scores.append(res["P_correct"])
        ys.append(test_correct[jj])
    curve = risk_coverage_curve(scores, ys)
    return curve["coverage"], curve["precision"]


def main() -> None:
    aligned = align_extractor_caches(GROUP_B, bench="math175")
    accept = aligned["accept"]; correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # Naive (does not depend on calibration but we vary the test split)
    cov_grid = np.linspace(0, 1, 21)
    naive_interp = []
    dajv_interp = []
    for seed in range(1, 11):
        cov, prec = collect_curve(seed, accept, correct, extractor_ids,
                                   lambda v, cal: naive_unanimous(v))
        ax.plot(cov, prec, color="gray", alpha=0.25, lw=0.7)
        if cov:
            naive_interp.append(np.interp(cov_grid,
                                          np.concatenate([[0.0], np.array(cov)]),
                                          np.concatenate([[1.0], np.array(prec)])))
        cov_d, prec_d = collect_curve(seed, accept, correct, extractor_ids,
                                       lambda v, cal: dajv_aggregate(v, cal))
        ax.plot(cov_d, prec_d, color="black", alpha=0.25, lw=0.7)
        if cov_d:
            dajv_interp.append(np.interp(cov_grid,
                                         np.concatenate([[0.0], np.array(cov_d)]),
                                         np.concatenate([[1.0], np.array(prec_d)])))

    if naive_interp:
        med_n = np.median(np.array(naive_interp), axis=0)
        ax.plot(cov_grid, med_n, color="gray", lw=2.0, label="Naive unanimous (median)")
    if dajv_interp:
        med_d = np.median(np.array(dajv_interp), axis=0)
        ax.plot(cov_grid, med_d, color="black", lw=2.0, label="DAJV (median)")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Precision @ coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("Risk-coverage curves across 10 seeds (math175 test)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_risk_coverage_bands.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'fig_risk_coverage_bands.pdf'}")


if __name__ == "__main__":
    main()

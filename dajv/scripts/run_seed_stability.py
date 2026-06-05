"""Seed-stability experiment.

Re-runs the aggregation comparison across 10 random calibration/test
splits (seeds 1..10) and reports mean ± std + 95% CI for each method
on math175.

Outputs:
  artifacts/seed_stability.json
  paper/figures/fig_seed_stability.pdf
"""
from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verifyensemble.aggregate.care import care_aggregate, fit_care_weights
from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_majority, naive_unanimous
from verifyensemble.evaluation.brier import brier_score
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}
ARTIFACTS = HERE.parent / "artifacts"
FIG_DIR = HERE.parent / "paper" / "figures"


def run_single_seed(seed: int, accept, correct, extractor_ids) -> dict:
    k = len(extractor_ids); n = len(correct)
    rng = random.Random(seed)
    idx = list(range(n)); rng.shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    cal_correct = [correct[j] for j in cal_idx]
    test_correct = [correct[j] for j in test_idx]
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]

    dajv = DajvCalibration.fit(accept_cal, cal_correct, extractor_ids)
    care_w = fit_care_weights(accept_cal, cal_correct)

    methods = {
        "naive_unanimous": lambda v: naive_unanimous(v),
        "naive_majority":  lambda v: naive_majority(v),
        "dajv":            lambda v: dajv_aggregate(v, dajv),
        "care":            lambda v: care_aggregate(v, care_w, threshold=0.5),
    }
    out: dict[str, dict] = {}
    for name, fn in methods.items():
        confs = []; ys = []; n_commit = 0; n_correct = 0
        for jj in range(len(test_idx)):
            v = [accept_test[i][jj] for i in range(k)]
            r = fn(v)
            if r["recommendation"] == "COMMIT":
                n_commit += 1
                if test_correct[jj]:
                    n_correct += 1
            if r["P_correct"] is not None:
                confs.append(r["P_correct"])
                ys.append(test_correct[jj])
        out[name] = {
            "n_committed": n_commit,
            "n_correct_committed": n_correct,
            "precision_at_commit": n_correct / max(n_commit, 1),
            "coverage": n_commit / len(test_idx),
            "ece": expected_calibration_error(confs, ys, n_bins=5) if confs else None,
            "brier": brier_score(confs, ys) if confs else None,
        }
    return out


def mean_std_ci(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "std": None, "ci_lo": None, "ci_hi": None}
    n = len(xs)
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if n > 1 else 0.0
    # 95% CI on the mean via t * s/sqrt(n); approximate t ~ 2.262 at df=9
    t = 2.262
    half = t * s / math.sqrt(n) if n > 1 else 0.0
    return {"mean": m, "std": s, "ci_lo": m - half, "ci_hi": m + half}


def main() -> None:
    aligned = align_extractor_caches(GROUP_B, bench="math175")
    accept = aligned["accept"]; correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    print(f"math175: n={len(correct)} k={len(extractor_ids)}")

    seeds = list(range(1, 11))
    per_seed = []
    print("\nseed | naive_u prec/cov | naive_m prec/cov | DAJV prec/cov | CARE prec/cov")
    for s in seeds:
        r = run_single_seed(s, accept, correct, extractor_ids)
        per_seed.append({"seed": s, "results": r})
        print(f"  {s:3d} | "
              f"{r['naive_unanimous']['precision_at_commit']:.3f}/{r['naive_unanimous']['coverage']:.3f} | "
              f"{r['naive_majority']['precision_at_commit']:.3f}/{r['naive_majority']['coverage']:.3f} | "
              f"{r['dajv']['precision_at_commit']:.3f}/{r['dajv']['coverage']:.3f} | "
              f"{r['care']['precision_at_commit']:.3f}/{r['care']['coverage']:.3f}")

    methods = ["naive_unanimous", "naive_majority", "dajv", "care"]
    metrics = ["precision_at_commit", "coverage", "ece", "brier"]
    summary: dict[str, dict[str, dict]] = {}
    for m_name in methods:
        summary[m_name] = {}
        for metric in metrics:
            vals = [s["results"][m_name][metric] for s in per_seed
                    if s["results"][m_name][metric] is not None]
            summary[m_name][metric] = mean_std_ci(vals)

    print("\nMean ± std across 10 seeds:")
    for m_name in methods:
        print(f"  {m_name}")
        for metric in metrics:
            s = summary[m_name][metric]
            if s["mean"] is None:
                continue
            print(f"    {metric}: {s['mean']:.4f} ± {s['std']:.4f} "
                  f"[95% CI: {s['ci_lo']:.4f}, {s['ci_hi']:.4f}]")

    out_obj = {"per_seed": per_seed, "summary": summary, "seeds": seeds}
    save_artifact(out_obj, ARTIFACTS / "seed_stability.json")
    print(f"\nWrote {ARTIFACTS / 'seed_stability.json'}")

    # Plot: precision-mean ± std error bars per method
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    names = ["naive_unanimous", "naive_majority", "dajv", "care"]
    labels = ["Naive unan.", "Naive maj.", "DAJV", "CARE"]
    means_p = [summary[m]["precision_at_commit"]["mean"] for m in names]
    stds_p = [summary[m]["precision_at_commit"]["std"] for m in names]
    means_c = [summary[m]["coverage"]["mean"] for m in names]
    stds_c = [summary[m]["coverage"]["std"] for m in names]
    for mean_p, std_p, mean_c, std_c, lab in zip(means_p, stds_p, means_c, stds_c, labels):
        ax.errorbar(mean_c, mean_p, xerr=std_c, yerr=std_p, fmt="o",
                    color="black", capsize=2, markersize=5)
        ax.annotate(lab, (mean_c, mean_p), textcoords="offset points",
                    xytext=(5, -3), fontsize=7)
    ax.set_xlabel("Coverage (mean ± std across 10 seeds)")
    ax.set_ylabel("Precision @ commit (mean ± std)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Seed-stability of aggregation methods (math175, 10 seeds)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_seed_stability.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'fig_seed_stability.pdf'}")


if __name__ == "__main__":
    main()

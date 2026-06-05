"""Seed stability across 10 seeds, per benchmark.

Extends `run_seed_stability.py` to also run on AIME 2025 and CleanMath,
in addition to math175. Reports mean ± std per method per benchmark.

Output:
  artifacts/seed_stability_per_bench.json
  paper/figures/fig_seed_stability_per_bench.pdf
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous
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


def evaluate_at_seed(seed: int, accept, correct, extractor_ids) -> dict:
    k = len(extractor_ids); n = len(correct)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    cal_correct = [correct[j] for j in cal_idx]
    test_correct = [correct[j] for j in test_idx]
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]

    # If calibration has 0 correct or 0 wrong, skip (degenerate)
    if sum(cal_correct) < 2 or sum(1 for c in cal_correct if not c) < 2:
        return {"degenerate": True}

    cal = DajvCalibration.fit(accept_cal, cal_correct, extractor_ids)

    methods = {
        "dajv": lambda v: dajv_aggregate(v, cal),
        "naive_unanimous": lambda v: naive_unanimous(v),
    }
    out = {}
    for name, fn in methods.items():
        n_commit = 0; n_correct = 0
        confs = []; ys = []
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
            "precision": n_correct / max(n_commit, 1),
            "coverage": n_commit / max(len(test_idx), 1),
            "n_committed": n_commit,
            "n_correct": n_correct,
            "ece": expected_calibration_error(confs, ys, n_bins=5) if confs else None,
            "brier": brier_score(confs, ys) if confs else None,
        }
    return out


def main() -> None:
    out_all: dict[str, dict] = {}
    print("seed | bench       | DAJV prec/cov          | Naive prec/cov         | DAJV ECE | Naive ECE")
    for bench in ["math175", "aime", "cleanmath"]:
        aligned = align_extractor_caches(GROUP_B, bench=bench)
        accept = aligned["accept"]
        correct = aligned["solver_correct"]
        eids = aligned["extractor_ids"]
        per_seed = []
        for seed in range(1, 11):
            r = evaluate_at_seed(seed, accept, correct, eids)
            if r.get("degenerate"):
                continue
            per_seed.append(r)
            print(f"  {seed:2d} | {bench:10s}  | "
                  f"{r['dajv']['precision']:.3f}/{r['dajv']['coverage']:.3f}      | "
                  f"{r['naive_unanimous']['precision']:.3f}/{r['naive_unanimous']['coverage']:.3f}      | "
                  f"{r['dajv']['ece']:.3f}    | {r['naive_unanimous']['ece']:.3f}"
                  if r['dajv']['ece'] is not None else
                  f"  {seed:2d} | {bench:10s}  | "
                  f"{r['dajv']['precision']:.3f}/{r['dajv']['coverage']:.3f}      | "
                  f"{r['naive_unanimous']['precision']:.3f}/{r['naive_unanimous']['coverage']:.3f}      | --")

        def agg(method: str, metric: str, _per_seed=per_seed) -> dict:
            vals = [s[method][metric] for s in _per_seed if s[method][metric] is not None]
            if not vals:
                return {"mean": None, "std": None, "n": 0}
            return {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

        out_all[bench] = {
            "n_seeds_used": len(per_seed),
            "dajv": {m: agg("dajv", m) for m in ["precision", "coverage", "ece", "brier"]},
            "naive": {m: agg("naive_unanimous", m) for m in ["precision", "coverage", "ece", "brier"]},
        }
        print(f"\n  {bench} summary (n_seeds={len(per_seed)}):")
        for method in ("dajv", "naive"):
            for metric in ("precision", "coverage", "ece"):
                s = out_all[bench][method][metric]
                if s["mean"] is None:
                    continue
                print(f"    {method:6s} {metric:10s}  {s['mean']:.3f} ± {s['std']:.3f}")
        print()

    save_artifact(out_all, ARTIFACTS / "seed_stability_per_bench.json")

    # Plot: precision mean per bench per method
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.4), sharey=True)
    import numpy as np
    for ax, bench in zip(axes, ["math175", "aime", "cleanmath"]):
        if bench not in out_all:
            continue
        b = out_all[bench]
        means_d = b["dajv"]["precision"]["mean"]; std_d = b["dajv"]["precision"]["std"]
        means_n = b["naive"]["precision"]["mean"]; std_n = b["naive"]["precision"]["std"]
        if means_d is None or means_n is None:
            ax.set_title(f"{bench} (degenerate)", fontsize=8); continue
        x = np.arange(2)
        ax.bar(x, [means_n, means_d], yerr=[std_n, std_d], color=["gray", "black"],
               capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(["Naive", "DAJV"])
        ax.set_ylim(0, 1.05)
        if bench == "math175":
            ax.set_ylabel("Precision @ commit")
        ax.set_title(f"{bench} (n_seeds = {b['n_seeds_used']})", fontsize=8)
    fig.suptitle("Per-bench seed stability (10 seeds, mean $\\pm$ std)", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_seed_stability_per_bench.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'fig_seed_stability_per_bench.pdf'}")


if __name__ == "__main__":
    main()

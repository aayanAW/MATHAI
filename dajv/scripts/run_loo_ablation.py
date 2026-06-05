"""Leave-one-out extractor ablation.

For each extractor in the 4-way Group B, drop it from the ensemble and
re-evaluate DAJV. Measures sensitivity of the headline results to the
choice of extractor set.

Output:
  artifacts/ablation_loo.json
  paper/figures/fig_ablation_loo.pdf
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
SEED = 42


def evaluate(accept, correct, extractor_ids, bench: str) -> dict:
    n = len(correct)
    rng = random.Random(SEED)
    idx = list(range(n)); rng.shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    k = len(extractor_ids)
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    cal_correct = [correct[j] for j in cal_idx]
    test_correct = [correct[j] for j in test_idx]
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]

    cal = DajvCalibration.fit(accept_cal, cal_correct, extractor_ids)

    dajv_committed = 0; dajv_correct_n = 0
    naive_committed = 0; naive_correct_n = 0
    dajv_confs, ys_d = [], []
    naive_confs, ys_n = [], []
    for jj in range(len(test_idx)):
        v = [accept_test[i][jj] for i in range(k)]
        out_d = dajv_aggregate(v, cal)
        out_n = naive_unanimous(v)
        if out_d["recommendation"] == "COMMIT":
            dajv_committed += 1
            if test_correct[jj]: dajv_correct_n += 1
        if out_n["recommendation"] == "COMMIT":
            naive_committed += 1
            if test_correct[jj]: naive_correct_n += 1
        if out_d["P_correct"] is not None:
            dajv_confs.append(out_d["P_correct"]); ys_d.append(test_correct[jj])
        if out_n["P_correct"] is not None:
            naive_confs.append(out_n["P_correct"]); ys_n.append(test_correct[jj])

    return {
        "bench": bench, "extractor_ids": extractor_ids,
        "n_test": len(test_idx), "n_cal": cal_n,
        "dajv": {
            "n_committed": dajv_committed,
            "n_correct": dajv_correct_n,
            "precision": dajv_correct_n / max(dajv_committed, 1),
            "coverage": dajv_committed / len(test_idx),
            "ece": expected_calibration_error(dajv_confs, ys_d, n_bins=5),
            "brier": brier_score(dajv_confs, ys_d),
        },
        "naive": {
            "n_committed": naive_committed,
            "n_correct": naive_correct_n,
            "precision": naive_correct_n / max(naive_committed, 1),
            "coverage": naive_committed / len(test_idx),
            "ece": expected_calibration_error(naive_confs, ys_n, n_bins=5),
            "brier": brier_score(naive_confs, ys_n),
        },
    }


def main() -> None:
    aligned = align_extractor_caches(GROUP_B, bench="math175")
    accept = aligned["accept"]; correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids)
    print(f"Loaded math175: n={len(correct)} k={k}")

    rows = []
    # Full 4-way
    full = evaluate(accept, correct, extractor_ids, "math175")
    rows.append({"setting": "full_4_way", **full})
    print(f"  full 4-way: DAJV cov={full['dajv']['coverage']:.3f} "
          f"prec={full['dajv']['precision']:.3f}")

    # LOO: drop each extractor in turn
    for drop_i in range(k):
        kept = [extractor_ids[i] for i in range(k) if i != drop_i]
        accept_loo = [accept[i] for i in range(k) if i != drop_i]
        res = evaluate(accept_loo, correct, kept, "math175")
        rows.append({"setting": f"loo_drop_{extractor_ids[drop_i]}", **res})
        print(f"  drop {extractor_ids[drop_i]:>22s}: "
              f"DAJV cov={res['dajv']['coverage']:.3f} prec={res['dajv']['precision']:.3f}")

    save_artifact(rows, ARTIFACTS / "ablation_loo.json")

    # Plot: precision vs coverage points
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for r in rows:
        cov = r["dajv"]["coverage"]
        prec = r["dajv"]["precision"]
        if r["setting"] == "full_4_way":
            ax.scatter(cov, prec, color="black", marker="*", s=80, zorder=3, label="full 4-way")
        else:
            ax.scatter(cov, prec, color="gray", marker="o", s=30, zorder=2)
    for r in rows:
        if r["setting"].startswith("loo_drop_"):
            label = r["setting"].replace("loo_drop_", "")
            label = label.split("_", 1)[1].replace("_", "-")
            ax.annotate(label, (r["dajv"]["coverage"], r["dajv"]["precision"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Precision @ commit")
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("LOO ablation: drop one extractor from the 4-way ensemble",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ablation_loo.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'fig_ablation_loo.pdf'}")


if __name__ == "__main__":
    main()

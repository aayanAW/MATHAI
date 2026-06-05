"""DAJV ablations: A3 (calibration sample-size sweep) + A4 (cross-bench transfer).

Outputs:
  artifacts/ablation_A3_calibration_size.json
  artifacts/ablation_A4_cross_benchmark.json
  paper/figures/fig_ablation_A3.pdf
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
from verifyensemble.dependency.matrix import DependencyMatrix
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


# ---------------------------------------------------------------------------
# Ablation A3: ECE vs calibration sample size
# ---------------------------------------------------------------------------
def ablation_A3() -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=None)
    accept = aligned["accept"]; correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids); n = len(correct)

    rng = random.Random(SEED)
    idx = list(range(n)); rng.shuffle(idx)
    # Fix the test split at 30% of total; vary the calibration size
    test_size = int(0.3 * n)
    test_idx = idx[-test_size:]
    pool = idx[:-test_size]
    n_pool = len(pool)

    cal_sizes = [30, 60, 90, 120, 160, 200, n_pool]
    out_rows = []
    for n_cal in cal_sizes:
        if n_cal > n_pool:
            continue
        cal_idx = pool[:n_cal]
        accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
        correct_cal = [correct[j] for j in cal_idx]
        cal = DajvCalibration.fit(accept_cal, correct_cal, extractor_ids)

        confs, ys = [], []
        n_committed = 0
        n_correct_committed = 0
        for j in test_idx:
            v = [accept[i][j] for i in range(k)]
            res = dajv_aggregate(v, cal)
            p = res.get("P_correct")
            if p is None: continue
            confs.append(p); ys.append(correct[j])
            if res.get("recommendation") == "COMMIT":
                n_committed += 1
                if correct[j]:
                    n_correct_committed += 1
        ece = expected_calibration_error(confs, ys, n_bins=5) if confs else None
        brier = brier_score(confs, ys) if confs else None
        prec = n_correct_committed / n_committed if n_committed > 0 else None
        cov = n_committed / len(test_idx)
        out_rows.append({
            "n_cal": n_cal,
            "ece": ece,
            "brier": brier,
            "precision_at_commit": prec,
            "coverage": cov,
            "n_committed": n_committed,
            "n_correct_committed": n_correct_committed,
        })
        print(f"  n_cal={n_cal:3d}  ECE={ece}  Brier={brier}  "
              f"prec@commit={prec}  cov={cov:.3f}")

    # Naive baseline (does not use calibration)
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]
    correct_test = [correct[j] for j in test_idx]
    naive_confs, naive_ys = [], []
    for jj in range(len(test_idx)):
        v = [accept_test[i][jj] for i in range(k)]
        res = naive_unanimous(v)
        p = res.get("P_correct")
        if p is None: continue
        naive_confs.append(p); naive_ys.append(correct_test[jj])
    naive_ece = expected_calibration_error(naive_confs, naive_ys, n_bins=5) if naive_confs else None

    out = {
        "ablation": "A3_calibration_size",
        "n_pool": n_pool,
        "n_test": len(test_idx),
        "rows": out_rows,
        "naive_ece": naive_ece,
    }
    save_artifact(out, ARTIFACTS / "ablation_A3_calibration_size.json")

    # Plot
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ns = [r["n_cal"] for r in out_rows]
    eces = [r["ece"] for r in out_rows]
    ax.plot(ns, eces, "o-", color="black", lw=1.4, label="DAJV ECE")
    if naive_ece is not None:
        ax.axhline(naive_ece, color="gray", ls="--", lw=1.0,
                   label=f"Naive unanimous ECE = {naive_ece:.3f}")
    ax.set_xlabel("Calibration sample size $n_{\\mathrm{cal}}$")
    ax.set_ylabel("Test ECE")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Ablation A3: ECE vs calibration size (Group B full, $n_{\\mathrm{test}} = "
                 + str(len(test_idx)) + "$)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ablation_A3.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'fig_ablation_A3.pdf'}")
    return out


# ---------------------------------------------------------------------------
# Ablation A4: cross-benchmark dependency-transfer test
#  - Estimate rho on math175 wrong subset (source)
#  - Estimate rho on Group-B-full wrong subset (target = math175+AIME+CleanMath)
#  - Compare entrywise |delta rho|
# ---------------------------------------------------------------------------
def ablation_A4() -> dict:
    # Source
    src = align_extractor_caches(GROUP_B, bench="math175")
    D_src = DependencyMatrix.from_accept(src["accept"], src["solver_correct"],
                                         src["extractor_ids"])
    # Target = the *complement* (Group-B-full minus math175): AIME ∪ CleanMath
    full = align_extractor_caches(GROUP_B, bench=None)
    src_ids = set(src["problem_ids"])
    keep_j = [j for j, pid in enumerate(full["problem_ids"]) if pid not in src_ids]
    if not keep_j:
        return {"ablation": "A4_cross_benchmark",
                "skipped": "no held-out target problems"}
    accept_tgt = [[full["accept"][i][j] for j in keep_j] for i in range(len(full["extractor_ids"]))]
    correct_tgt = [full["solver_correct"][j] for j in keep_j]
    D_tgt = DependencyMatrix.from_accept(accept_tgt, correct_tgt,
                                         full["extractor_ids"])

    k = len(src["extractor_ids"])
    deltas = []
    for i in range(k):
        for j in range(i + 1, k):
            deltas.append({
                "i": src["extractor_ids"][i],
                "j": src["extractor_ids"][j],
                "kappa_src": D_src.kappa[i][j],
                "kappa_tgt": D_tgt.kappa[i][j],
                "delta_kappa": D_tgt.kappa[i][j] - D_src.kappa[i][j],
                "abs_delta": abs(D_tgt.kappa[i][j] - D_src.kappa[i][j]),
            })
    within_10 = sum(1 for d in deltas if d["abs_delta"] <= 0.10)
    out = {
        "ablation": "A4_cross_benchmark",
        "src_bench": "math175",
        "tgt_bench": "AIME ∪ CleanMath",
        "n_src": len(src["problem_ids"]),
        "n_src_wrong": D_src.n_wrong,
        "n_tgt": len(keep_j),
        "n_tgt_wrong": D_tgt.n_wrong,
        "deltas": deltas,
        "n_pairs": len(deltas),
        "n_within_0.10": within_10,
        "fraction_within_0.10": within_10 / len(deltas) if deltas else None,
        "H4_threshold": "fraction_within_0.10 >= 0.70",
        "H4_pass": within_10 / len(deltas) >= 0.70 if deltas else None,
    }
    save_artifact(out, ARTIFACTS / "ablation_A4_cross_benchmark.json")
    print("  A4 deltas:")
    for d in deltas:
        print(f"    {d['i'][:14]} vs {d['j'][:14]}  Δκ = {d['delta_kappa']:+.3f}")
    print(f"  H4 pass ({within_10}/{len(deltas)} pairs within ±0.10)?",
          out["H4_pass"])
    return out


def main() -> None:
    print("=== Ablation A3: calibration sample-size sweep ===")
    a3 = ablation_A3()
    print("\n=== Ablation A4: cross-benchmark dependency transfer ===")
    a4 = ablation_A4()
    summary = {"A3": a3, "A4": a4}
    save_artifact(summary, ARTIFACTS / "ablations_summary.json")
    print("\nWrote artifacts/ablations_summary.json")


if __name__ == "__main__":
    main()

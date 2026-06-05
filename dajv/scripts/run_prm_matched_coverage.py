"""DAJV vs PRMs at matched coverage with McNemar significance.

For each benchmark + each PRM aggregation, threshold the PRM score to
match the DAJV top-tier coverage. Report point precision and the
McNemar mid-p between DAJV-correct and PRM-correct at the matched
operating point.

Output:
  artifacts/prm_matched_coverage.json
  paper/figures/fig_prm_matched_coverage.pdf
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.evaluation.mcnemar import mcnemar_mid_p
from verifyensemble.utils.io import align_extractor_caches, save_artifact

EXTRACTOR_CACHES = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}
QWEN_PRM = HERE.parent / "../MATHAI/results/exp40_qwen_prm.json"
SKY_PRM = HERE.parent / "../MATHAI/results/exp40_skywork_prm.json"
ARTIFACTS = HERE.parent / "artifacts"
FIG_DIR = HERE.parent / "paper" / "figures"
SEED = 42


def load_prm_scores(path: Path, bench: str, aggregation: str) -> dict[str, float]:
    with path.open() as f:
        d = json.load(f)
    out = {}
    for pid, entry in d.get(bench, {}).items():
        samples = entry.get("per_sample", [])
        vals = [s.get(aggregation) for s in samples if s.get(aggregation) is not None]
        if vals:
            out[pid] = max(vals)
    return out


def matched_coverage_precision(scores, correct, target_k):
    """Threshold to keep top-k items; return precision."""
    if not scores or target_k <= 0:
        return None, [False] * len(scores)
    pairs = sorted(zip(scores, correct), key=lambda t: -t[0])
    if target_k > len(pairs):
        target_k = len(pairs)
    top = pairs[:target_k]
    n_corr = sum(1 for _, c in top if c)
    # commit_indicator aligned to original order
    threshold = pairs[target_k - 1][0]
    commits = [s >= threshold for s in scores]
    # If ties at threshold, may commit more than target_k; that's fine
    return n_corr / target_k, commits


def run(bench: str) -> dict:
    aligned = align_extractor_caches(EXTRACTOR_CACHES, bench=bench)
    pids = aligned["problem_ids"]
    accept = aligned["accept"]
    correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids); n = len(pids)

    # 70/30 split
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]
    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    cal_correct = [correct[j] for j in cal_idx]
    test_correct = [correct[j] for j in test_idx]
    test_pids = [pids[j] for j in test_idx]
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]

    cal = DajvCalibration.fit(accept_cal, cal_correct, extractor_ids)

    # DAJV operating point: how many committed at default thresholds?
    dajv_commits = []
    dajv_correct_committed = []
    for jj in range(len(test_idx)):
        v = [accept_test[i][jj] for i in range(k)]
        r = dajv_aggregate(v, cal)
        committed = (r["recommendation"] == "COMMIT")
        dajv_commits.append(committed)
        dajv_correct_committed.append(committed and test_correct[jj])

    n_dajv_commits = sum(dajv_commits)
    n_dajv_correct = sum(dajv_correct_committed)
    dajv_precision = n_dajv_correct / max(n_dajv_commits, 1)
    dajv_coverage = n_dajv_commits / len(test_idx)

    print(f"\n--- bench={bench} n_test={len(test_idx)} ---")
    print(f"DAJV: {n_dajv_correct}/{n_dajv_commits} = {dajv_precision:.3f} "
          f"at {dajv_coverage:.3f} coverage")

    out = {
        "bench": bench, "n_test": len(test_idx),
        "dajv": {
            "n_committed": n_dajv_commits,
            "n_correct": n_dajv_correct,
            "precision": dajv_precision,
            "coverage": dajv_coverage,
        },
        "prms": {},
    }

    # PRMs at matched coverage (commit top-n_dajv_commits by score)
    for prm_name, prm_path in [("qwen_prm", QWEN_PRM), ("skywork_prm", SKY_PRM)]:
        for agg in ["min", "mean", "product", "last"]:
            scores_map = load_prm_scores(prm_path, bench, agg)
            scores, ys, has_score = [], [], []
            for jj, pid in enumerate(test_pids):
                if pid in scores_map:
                    scores.append(scores_map[pid])
                    ys.append(test_correct[jj])
                    has_score.append(True)
                else:
                    has_score.append(False)
            if not scores:
                continue
            prec, commits = matched_coverage_precision(scores, ys, n_dajv_commits)
            n_committed = sum(commits)
            n_correct_committed = sum(1 for c, y in zip(commits, ys) if c and y)
            # Align commit booleans back to the full test set
            full_commits = []
            full_correct_commits = []
            i_score = 0
            for jj, h in enumerate(has_score):
                if h:
                    c = commits[i_score]
                    full_commits.append(c)
                    full_correct_commits.append(c and test_correct[jj])
                    i_score += 1
                else:
                    full_commits.append(False)
                    full_correct_commits.append(False)
            # McNemar: dajv_correct_committed vs full_correct_commits
            mc = mcnemar_mid_p(dajv_correct_committed, full_correct_commits)
            out["prms"][f"{prm_name}_{agg}"] = {
                "n_committed": n_committed,
                "n_correct_committed": n_correct_committed,
                "precision": prec,
                "coverage": n_committed / len(test_idx),
                "mcnemar_vs_dajv": mc,
            }
            prec_str = f"{prec:.3f}" if prec is not None else "n/a"
            print(f"  {prm_name}_{agg:8s}: {n_correct_committed}/{n_committed} = "
                  f"{prec_str}  McNemar p={mc['mid_p']:.4g}")

    return out


def main() -> None:
    all_results = {}
    for bench in ["math175", "aime", "cleanmath"]:
        all_results[bench] = run(bench)
    save_artifact(all_results, ARTIFACTS / "prm_matched_coverage.json")

    # Plot: per-bench bar of DAJV vs best-PRM-per-bench precision
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    benches = ["math175", "aime", "cleanmath"]
    width = 0.25
    import numpy as np
    x = np.arange(len(benches))
    dajv_p = [all_results[b]["dajv"]["precision"] for b in benches]
    qwen_best = []; sky_best = []
    for b in benches:
        prms = all_results[b]["prms"]
        qwen_best.append(max((v["precision"] for k, v in prms.items() if k.startswith("qwen") and v["precision"] is not None), default=0))
        sky_best.append(max((v["precision"] for k, v in prms.items() if k.startswith("skywork") and v["precision"] is not None), default=0))
    ax.bar(x - width, dajv_p, width, label="DAJV (ours)", color="black")
    ax.bar(x,         qwen_best, width, label="Qwen-PRM (best agg.)", color="gray")
    ax.bar(x + width, sky_best,  width, label="Skywork-PRM (best agg.)", color="lightgray", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(benches)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Precision @ matched coverage")
    ax.legend(fontsize=7, frameon=False, loc="lower center")
    ax.set_title("DAJV vs PRMs at matched coverage (n_test per bench varies)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_prm_matched_coverage.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG_DIR / 'fig_prm_matched_coverage.pdf'}")


if __name__ == "__main__":
    main()

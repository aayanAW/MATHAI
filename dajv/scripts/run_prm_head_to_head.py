"""DAJV vs Qwen-PRM vs Skywork-PRM head-to-head from cached PRM scores.

Uses:
  MATHAI/results/exp40_qwen_prm.json     per-problem PRM scores
  MATHAI/results/exp40_skywork_prm.json  per-problem PRM scores
  MATHAI/results/exp46_gptoss_extractor.json  for solver_correct label

For each PRM, computes the problem-level score = max across 10 samples
of {min, mean, product, last} aggregations. Compares risk-coverage
curves against DAJV (fitted on the math175 calibration) at matched
coverage.

Output:
  artifacts/prm_head_to_head.json
  paper/figures/fig_prm_head_to_head.pdf
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
from verifyensemble.aggregate.naive import naive_unanimous
from verifyensemble.evaluation.risk_coverage import risk_coverage_curve
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
    """Return {problem_id: max-across-10-samples of <aggregation>}."""
    with path.open() as f:
        d = json.load(f)
    out = {}
    bench_dict = d.get(bench, {})
    for pid, entry in bench_dict.items():
        samples = entry.get("per_sample", [])
        if not samples:
            continue
        vals = [s.get(aggregation) for s in samples if s.get(aggregation) is not None]
        if vals:
            out[pid] = max(vals)
    return out


def run(bench: str) -> dict:
    aligned = align_extractor_caches(EXTRACTOR_CACHES, bench=bench)
    pids = aligned["problem_ids"]
    accept = aligned["accept"]
    correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids); n = len(pids)
    print(f"\n--- bench={bench} n={n} k={k} ---")

    # 70/30 cal/test split (same seed as headline)
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = idx[:cal_n], idx[cal_n:]

    accept_cal = [[accept[i][j] for j in cal_idx] for i in range(k)]
    correct_cal = [correct[j] for j in cal_idx]
    correct_test = [correct[j] for j in test_idx]
    accept_test = [[accept[i][j] for j in test_idx] for i in range(k)]

    cal = DajvCalibration.fit(accept_cal, correct_cal, extractor_ids)

    # DAJV scores on test split
    dajv_scores, dajv_ys = [], []
    naive_scores, naive_ys = [], []
    for jj in range(len(test_idx)):
        v = [accept_test[i][jj] for i in range(k)]
        r_d = dajv_aggregate(v, cal)
        r_n = naive_unanimous(v)
        if r_d["P_correct"] is not None:
            dajv_scores.append(r_d["P_correct"])
            dajv_ys.append(correct_test[jj])
        if r_n["P_correct"] is not None:
            naive_scores.append(r_n["P_correct"])
            naive_ys.append(correct_test[jj])

    # PRM scores on test split, aligned by pid
    test_pids = [pids[j] for j in test_idx]
    prm_curves: dict[str, dict] = {}
    for prm_name, prm_path in [("qwen_prm", QWEN_PRM), ("skywork_prm", SKY_PRM)]:
        for agg in ["min", "mean", "product", "last"]:
            scores_map = load_prm_scores(prm_path, bench, agg)
            scores, ys = [], []
            for jj, pid in enumerate(test_pids):
                if pid in scores_map:
                    scores.append(scores_map[pid])
                    ys.append(correct_test[jj])
            if not scores:
                continue
            curve = risk_coverage_curve(scores, ys)
            prm_curves[f"{prm_name}_{agg}"] = {
                "curve": curve,
                "scores": scores,
                "ys": ys,
                "n": len(scores),
            }
            print(f"  {prm_name}_{agg}: n={len(scores)}")

    return {
        "bench": bench,
        "n_test": len(test_idx),
        "dajv_curve": risk_coverage_curve(dajv_scores, dajv_ys) if dajv_scores else None,
        "naive_curve": risk_coverage_curve(naive_scores, naive_ys) if naive_scores else None,
        "dajv_n": len(dajv_scores),
        "naive_n": len(naive_scores),
        "prm_curves": prm_curves,
    }


def plot_for_bench(results: dict, ax, title: str) -> None:
    # DAJV
    if results["dajv_curve"] and results["dajv_curve"]["coverage"]:
        c = results["dajv_curve"]
        ax.plot(c["coverage"], c["precision"], color="black", lw=1.6,
                label="DAJV (ours)", linestyle="-")
    # Naive
    if results["naive_curve"] and results["naive_curve"]["coverage"]:
        c = results["naive_curve"]
        ax.plot(c["coverage"], c["precision"], color="gray", lw=1.2,
                label="Naive unanimous", linestyle="--")
    # PRMs: pick best aggregation by AUC for each PRM
    from verifyensemble.evaluation.risk_coverage import risk_coverage_auc
    best_by_prm: dict[str, tuple[float, str, dict]] = {}
    for key, data in results["prm_curves"].items():
        prm = "qwen" if key.startswith("qwen") else "sky"
        auc = risk_coverage_auc(data["scores"], data["ys"])
        if prm not in best_by_prm or auc < best_by_prm[prm][0]:
            best_by_prm[prm] = (auc, key, data)
    colors = {"qwen": "tab:blue", "sky": "tab:red"}
    for prm, (_, key, data) in best_by_prm.items():
        c = data["curve"]
        if c["coverage"]:
            ax.plot(c["coverage"], c["precision"], color=colors[prm], lw=1.2,
                    label=f"{key} (best)", linestyle=":")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Precision @ coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.set_title(title, fontsize=8)


def main() -> None:
    all_results: dict[str, dict] = {}
    for bench in ["math175", "aime", "cleanmath"]:
        try:
            all_results[bench] = run(bench)
        except Exception as e:
            print(f"  bench={bench} ERR {e}")

    save_artifact(all_results, ARTIFACTS / "prm_head_to_head.json")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True)
    for ax, bench in zip(axes, ["math175", "aime", "cleanmath"]):
        if bench in all_results:
            plot_for_bench(all_results[bench], ax, f"Bench: {bench}")
        else:
            ax.set_title(f"{bench} (skipped)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_prm_head_to_head.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG_DIR / 'fig_prm_head_to_head.pdf'}")


if __name__ == "__main__":
    main()

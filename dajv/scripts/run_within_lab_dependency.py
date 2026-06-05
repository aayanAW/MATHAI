"""Within-lab vs cross-lab pairwise dependency (H6).

The Group B extractor cache contains a within-lab pair:
    gpt-oss-120B and gpt-5-mini are both OpenAI models.

We compute pairwise dependency metrics for this within-lab pair and
compare against the 5 cross-lab pairs.

Hypothesis (H6): within-lab pair has systematically higher dependency.

Output:
  artifacts/within_lab_dependency.json
  paper/figures/fig_within_lab.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verifyensemble.dependency.cig import cig
from verifyensemble.dependency.joint_fp import joint_fp_rate
from verifyensemble.dependency.kappa import cohen_kappa
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}

# Lab assignment per extractor
LAB = {
    "E05_gpt_oss_120B":      "OpenAI",
    "E06_gpt_5_mini":        "OpenAI",
    "E07_claude_sonnet_4_6": "Anthropic",
    "E09_qwen3_coder_480B":  "Alibaba",
}

ARTIFACTS = HERE.parent / "artifacts"
FIG_DIR = HERE.parent / "paper" / "figures"


def main() -> None:
    rows = []
    for bench in ["math175", None]:
        bench_label = bench or "B_full"
        aligned = align_extractor_caches(GROUP_B, bench=bench)
        accept = aligned["accept"]
        correct = aligned["solver_correct"]
        eids = aligned["extractor_ids"]
        n_wrong = sum(1 for c in correct if not c)
        print(f"\n=== {bench_label} (n_wrong={n_wrong}) ===")
        if n_wrong < 5:
            print("  skip: too few wrong candidates")
            continue
        wrong = [j for j, c in enumerate(correct) if not c]
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                eid_i, eid_j = eids[i], eids[j]
                same_lab = LAB[eid_i] == LAB[eid_j]
                a_w = [accept[i][t] for t in wrong]
                b_w = [accept[j][t] for t in wrong]
                k = cohen_kappa(a_w, b_w)
                stats = joint_fp_rate(accept[i], accept[j], correct)
                cig_val = cig(accept[i], accept[j], correct)
                rows.append({
                    "bench": bench_label,
                    "pair": f"{eid_i} vs {eid_j}",
                    "labs": f"{LAB[eid_i]} vs {LAB[eid_j]}",
                    "same_lab": same_lab,
                    "kappa": k,
                    "pi_i": stats["pi_i"],
                    "pi_j": stats["pi_j"],
                    "indep_bound": stats["indep_bound"],
                    "joint_fp": stats["joint_observed"],
                    "ratio": stats["ratio"] if stats["ratio"] != float("inf") else None,
                    "cig": cig_val,
                })
                tag = "WITHIN" if same_lab else "cross "
                print(f"  {tag}  {LAB[eid_i][:5]}-{LAB[eid_j][:5]}  k={k:.3f}  "
                      f"ratio={stats['ratio']:.2f}  cig={cig_val:.4f}")

    save_artifact(rows, ARTIFACTS / "within_lab_dependency.json")

    # Summary: within-lab vs cross-lab medians on math175
    math175 = [r for r in rows if r["bench"] == "math175"]
    within = [r for r in math175 if r["same_lab"]]
    cross = [r for r in math175 if not r["same_lab"]]

    def med(arr, key):
        vals = [r[key] for r in arr if r[key] is not None]
        if not vals:
            return None
        return sorted(vals)[len(vals) // 2]

    def _fmt(v, spec):
        return format(v, spec) if v is not None else "n/a"

    print("\n--- math175: within-lab vs cross-lab (median) ---")
    print(f"  within-lab pairs (n={len(within)}):")
    print(f"    kappa  median = {_fmt(med(within, 'kappa'), '.3f')}")
    print(f"    ratio  median = {_fmt(med(within, 'ratio'), '.2f')}")
    print(f"    cig    median = {_fmt(med(within, 'cig'), '.4f')}")
    print(f"  cross-lab pairs (n={len(cross)}):")
    print(f"    kappa  median = {_fmt(med(cross, 'kappa'), '.3f')}")
    print(f"    ratio  median = {_fmt(med(cross, 'ratio'), '.2f')}")
    print(f"    cig    median = {_fmt(med(cross, 'cig'), '.4f')}")

    # Plot kappa bars within vs cross
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pairs = [r["pair"] for r in math175]
    short_pairs = [f"{r['labs'].replace(' vs ', '/')}" for r in math175]
    kappas = [r["kappa"] for r in math175]
    colors = ["black" if r["same_lab"] else "gray" for r in math175]
    ax.bar(range(len(pairs)), kappas, color=colors)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(short_pairs, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Cohen's $\\kappa$")
    ax.set_ylim(0, 1.0)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="black", label="within-lab"),
        Patch(facecolor="gray",  label="cross-lab"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    ax.set_title("H6: within-lab vs cross-lab pairwise $\\kappa$ on math175",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_within_lab.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG_DIR / 'fig_within_lab.pdf'}")


if __name__ == "__main__":
    main()

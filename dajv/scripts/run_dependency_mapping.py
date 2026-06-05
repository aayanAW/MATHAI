"""Compute the empirical DAJV dependency matrix on existing X-SGRV cache.

The legacy X-SGRV cache has two disjoint problem-id namespaces:

  Group A (Llama, DeepSeek):   math500_* problems, n=323
  Group B (gptoss, gpt5,        math175, aime, cleanmath  problems, n=330
           claude, qwen3)

This script computes one dependency matrix per group, plus a per-benchmark
breakdown for Group B.

Outputs:
  artifacts/dependency_matrix_A_math500.json
  artifacts/dependency_matrix_B_full.json
  artifacts/dependency_matrix_B_math175.json
  artifacts/dependency_matrix_B_aime.json
  artifacts/dependency_matrix_B_cleanmath.json
  artifacts/bootstrap_ci_*.json
  artifacts/dependency_summary.json
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.dependency.matrix import DependencyMatrix, bootstrap_ci
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_A = {
    "E01_llama_3_3_70B": HERE.parent / "../MATHAI/results/exp37_xsgrv_math500_llama70b.json",
    "E03_deepseek_v3":   HERE.parent / "../MATHAI/results/exp37_xsgrv_math500_deepseek.json",
}
GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def run_one(group: dict, bench: str | None, tag: str,
            do_bootstrap: bool = True) -> dict | None:
    print(f"\n=== group_tag={tag} bench={bench} ===")
    aligned = align_extractor_caches(group, bench=bench)
    if not aligned["problem_ids"]:
        print("  skip: no shared problems")
        return None
    extractor_ids = aligned["extractor_ids"]
    accept = aligned["accept"]
    solver_correct = aligned["solver_correct"]
    n_prob = len(aligned["problem_ids"])
    n_wrong = sum(1 for c in solver_correct if not c)
    print(f"  k={len(extractor_ids)} n_prob={n_prob} n_wrong={n_wrong}")

    D = DependencyMatrix.from_accept(accept, solver_correct, extractor_ids)
    out_path = ARTIFACTS / f"dependency_matrix_{tag}.json"
    D.save_json(out_path)
    print(f"  wrote {out_path}")

    summary = D.summary()
    print("  summary (off-diag pairs):")
    print("    median kappa            =", summary["kappa"]["median"])
    print("    median joint/indep ratio=", summary["joint_fp_over_indep_ratio"]["median"])
    print("    median excess           =", summary["excess_over_indep"]["median"])
    print("    median CIG              =", summary["cig"]["median"])
    print("    pi range                =", summary["marginals_pi"])

    if do_bootstrap and n_prob >= 30:
        bs = bootstrap_ci(accept, solver_correct, extractor_ids,
                          n_bootstrap=500, seed=42)
        save_artifact(bs, ARTIFACTS / f"bootstrap_ci_{tag}.json")
        print("  wrote bootstrap CIs")

    return {
        "tag": tag, "bench": bench,
        "n_problems": n_prob, "n_wrong": n_wrong,
        "extractor_ids": extractor_ids,
        "summary": summary,
    }


def main() -> None:
    summaries = []
    # Group A: 2-way on math500
    summaries.append(run_one(GROUP_A, None, "A_math500_full"))
    # Group B: 4-way, full and per-bench
    summaries.append(run_one(GROUP_B, None,        "B_full"))
    summaries.append(run_one(GROUP_B, "math175",   "B_math175"))
    summaries.append(run_one(GROUP_B, "aime",      "B_aime"))
    summaries.append(run_one(GROUP_B, "cleanmath", "B_cleanmath"))

    save_artifact([s for s in summaries if s is not None],
                  ARTIFACTS / "dependency_summary.json")
    print("\nAll done. See artifacts/dependency_summary.json")


if __name__ == "__main__":
    main()

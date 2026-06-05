"""Full baseline comparison: DAJV vs naive vs SE (Math/NLI) vs p(True) vs PRMs.

Loads:
  exp35_semantic_entropy.json  - SE scores (math + NLI clustering)
  exp36_ptrue.json             - p(True) scores
  exp40_qwen_prm.json          - Qwen PRM scores
  exp40_skywork_prm.json       - Skywork PRM scores
  exp46/47/48/50 extractor caches - for DAJV votes + solver_correct

Each baseline scored at matched DAJV coverage. Reports per-bench
precision + Clopper-Pearson CIs + McNemar mid-p vs DAJV.

Output:
  artifacts/full_baseline_compare.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.posterior import clopper_pearson
from verifyensemble.evaluation.mcnemar import mcnemar_mid_p
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP_B = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}
SE_PATH = HERE.parent / "../MATHAI/results/exp35_semantic_entropy.json"
PTRUE_PATH = HERE.parent / "../MATHAI/results/exp36_ptrue.json"
QWEN_PRM = HERE.parent / "../MATHAI/results/exp40_qwen_prm.json"
SKY_PRM = HERE.parent / "../MATHAI/results/exp40_skywork_prm.json"
ARTIFACTS = HERE.parent / "artifacts"
SEED = 42


def load_se(path: Path, variant: str) -> dict[str, tuple[float, bool]]:
    """{pid: (-entropy, plurality_correct)} -- higher score = more confident."""
    with path.open() as f:
        d = json.load(f)
    return {r["id"]: (-r[variant]["entropy"], r["plurality_correct"])
            for r in d["rows"]}


def load_ptrue(path: Path) -> dict[str, tuple[float, bool]]:
    with path.open() as f:
        d = json.load(f)
    return {r["id"]: (r["p_true"], r["plurality_correct"]) for r in d["rows"]
            if r["p_true"] is not None}


def load_prm(path: Path, bench: str, agg: str) -> dict[str, float]:
    with path.open() as f:
        d = json.load(f)
    out = {}
    for pid, entry in d.get(bench, {}).items():
        vals = [s.get(agg) for s in entry.get("per_sample", []) if s.get(agg) is not None]
        if vals:
            out[pid] = max(vals)
    return out


def top_k(scores: list[float], correct: list[bool], k: int) -> tuple[int, int]:
    if not scores or k <= 0:
        return 0, 0
    pairs = sorted(zip(scores, correct), key=lambda t: -t[0])
    top = pairs[: min(k, len(pairs))]
    return len(top), sum(1 for _, c in top if c)


def run(bench: str) -> dict:
    aligned = align_extractor_caches(GROUP_B, bench=bench)
    pids = aligned["problem_ids"]
    accept = aligned["accept"]
    correct = aligned["solver_correct"]
    extractor_ids = aligned["extractor_ids"]
    k = len(extractor_ids); n = len(pids)
    if n == 0:
        return {"skipped": True}

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

    # DAJV operating point
    dajv_commit_correct = []
    n_dajv_commits = 0
    n_dajv_correct = 0
    for jj in range(len(test_idx)):
        v = [accept_test[i][jj] for i in range(k)]
        r = dajv_aggregate(v, cal)
        committed = r["recommendation"] == "COMMIT"
        dajv_commit_correct.append(committed and test_correct[jj])
        if committed:
            n_dajv_commits += 1
            if test_correct[jj]:
                n_dajv_correct += 1
    dajv_lo, dajv_hi = clopper_pearson(n_dajv_correct, max(n_dajv_commits, 1))

    out = {
        "bench": bench, "n_test": len(test_idx),
        "dajv": {
            "n_committed": n_dajv_commits, "n_correct": n_dajv_correct,
            "precision": n_dajv_correct / max(n_dajv_commits, 1),
            "coverage": n_dajv_commits / len(test_idx),
            "ci": [dajv_lo, dajv_hi],
        },
        "baselines": {},
    }

    # Build score sources
    se_math_map = load_se(SE_PATH, "se_math")
    se_nli_map = load_se(SE_PATH, "se_nli")
    ptrue_map = load_ptrue(PTRUE_PATH)

    score_sources: dict[str, dict[str, float]] = {
        "se_math_neg_entropy": {pid: se_math_map[pid][0] for pid in se_math_map},
        "se_nli_neg_entropy":  {pid: se_nli_map[pid][0] for pid in se_nli_map},
        "p_true":              {pid: ptrue_map[pid][0] for pid in ptrue_map},
    }
    for agg in ["min", "mean", "product", "last"]:
        score_sources[f"qwen_prm_{agg}"] = load_prm(QWEN_PRM, bench, agg)
        score_sources[f"skywork_prm_{agg}"] = load_prm(SKY_PRM, bench, agg)

    for name, scores_map in score_sources.items():
        scores, ys, has_score = [], [], []
        for jj, pid in enumerate(test_pids):
            if pid in scores_map:
                scores.append(scores_map[pid])
                ys.append(test_correct[jj])
                has_score.append(True)
            else:
                has_score.append(False)
        if not scores:
            out["baselines"][name] = {"skipped": "no scores"}
            continue
        n_c, n_correct = top_k(scores, ys, n_dajv_commits)
        if n_c == 0:
            prec, lo, hi = None, None, None
        else:
            prec = n_correct / n_c
            lo, hi = clopper_pearson(n_correct, n_c)
        # Build full-test commit-correct for McNemar
        pairs_sorted = sorted(
            enumerate(zip(scores, ys)), key=lambda kv: -kv[1][0]
        )
        commit_score_idx = {p[0] for p in pairs_sorted[:n_c]}
        # Map back to full test indices
        baseline_commit_correct = [False] * len(test_idx)
        i_score = 0
        for jj, h in enumerate(has_score):
            if h:
                if i_score in commit_score_idx and test_correct[jj]:
                    baseline_commit_correct[jj] = True
                i_score += 1
        mc = mcnemar_mid_p(dajv_commit_correct, baseline_commit_correct)
        out["baselines"][name] = {
            "n_with_score": len(scores),
            "n_committed": n_c,
            "n_correct": n_correct,
            "precision": prec,
            "ci": [lo, hi] if lo is not None else None,
            "mcnemar_vs_dajv_mid_p": mc["mid_p"],
        }
    return out


def main() -> None:
    out: dict[str, dict] = {}
    for bench in ["math175", "aime", "cleanmath"]:
        out[bench] = run(bench)
        if out[bench].get("skipped"):
            continue
        d = out[bench]["dajv"]
        print(f"\n=== bench={bench} n_test={out[bench]['n_test']} ===")
        print(f"  DAJV: {d['n_correct']}/{d['n_committed']} = "
              f"{d['precision']:.3f}  CI=[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]")
        for name, b in out[bench]["baselines"].items():
            if b.get("skipped"):
                print(f"  {name:22s}: {b['skipped']}")
                continue
            prec_str = f"{b['precision']:.3f}" if b["precision"] is not None else "n/a"
            ci_str = (f"[{b['ci'][0]:.3f},{b['ci'][1]:.3f}]"
                      if b["ci"] is not None else "n/a")
            print(f"  {name:22s}: {b['n_correct']}/{b['n_committed']} = "
                  f"{prec_str}  CI={ci_str}  McN p={b['mcnemar_vs_dajv_mid_p']:.4g}")
    save_artifact(out, ARTIFACTS / "full_baseline_compare.json")
    print(f"\nWrote {ARTIFACTS / 'full_baseline_compare.json'}")


if __name__ == "__main__":
    main()

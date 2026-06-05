"""VERGE-Z3 (faithful replication) vs DAJV head-to-head.

Loads the same 6/7-extractor cache, applies the VERGE-Z3 aggregator
(with Z3 SMT formal-verification stage), and compares against DAJV and
the simpler VERGE proxy.

Saves: artifacts/verge_z3_compare.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.aggregate.naive import naive_unanimous
from verifyensemble.aggregate.verge_proxy import verge_proxy_aggregate
from verifyensemble.aggregate.verge_z3 import verge_z3_aggregate
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
}

ARTIFACTS = HERE.parent / "artifacts"
SEED = 42


def run_on_bench(bench: str | None, tag: str) -> dict:
    aligned = align_extractor_caches(GROUP, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "skipped": "too few"}
    k = len(aligned["extractor_ids"])
    pids = aligned["problem_ids"]

    rng = random.Random(SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    cal_n = int(0.7 * n)
    cal_idx, test_idx = indices[:cal_n], indices[cal_n:]

    def _slice(arr, idx):
        return [arr[i] for i in idx]

    accept_cal = [_slice(aligned["accept"][i], cal_idx) for i in range(k)]
    accept_test = [_slice(aligned["accept"][i], test_idx) for i in range(k)]
    classification_test = [_slice(aligned["classification"][i], test_idx)
                           for i in range(k)]
    cv_test = [_slice(aligned["candidate_verdict"][i], test_idx)
               for i in range(k)]
    correct_test = _slice(aligned["solver_correct"], test_idx)
    correct_cal = _slice(aligned["solver_correct"], cal_idx)

    # Scripts and candidates per test problem
    # Load scripts from the original caches via the aligned helper
    # (the helper drops scripts, so we re-load here).
    scripts_test = [[None] * len(test_idx) for _ in range(k)]
    candidates_test = [None] * len(test_idx)
    for i, eid in enumerate(aligned["extractor_ids"]):
        cache = json.load(open(GROUP[eid]))
        per_pid = {r["id"]: r for r in cache["results"]
                   if not bench or r.get("bench") == bench}
        for j_out, j_full in enumerate(test_idx):
            pid = pids[j_full]
            rec = per_pid.get(pid)
            if rec:
                scripts_test[i][j_out] = rec.get("script")
                if i == 0:
                    candidates_test[j_out] = str(rec.get("candidate", ""))

    cal = DajvCalibration.fit(accept_cal, correct_cal,
                              aligned["extractor_ids"])

    rows = []
    for j in range(len(test_idx)):
        votes_j = [bool(accept_test[i][j]) for i in range(k)]
        cls_j = [classification_test[i][j] for i in range(k)]
        cv_j = [cv_test[i][j] for i in range(k)]
        scripts_j = [scripts_test[i][j] for i in range(k)]
        cand_j = candidates_test[j] or ""
        rows.append({
            "correct": bool(correct_test[j]),
            "naive": naive_unanimous(votes_j),
            "dajv": dajv_aggregate(votes_j, cal),
            "verge_proxy": verge_proxy_aggregate(votes_j, cls_j, cv_j, min_agree=3),
            "verge_z3": verge_z3_aggregate(votes_j, cls_j, cv_j,
                                            scripts_j, cand_j, min_agree=3),
        })

    def head(key: str, commits: set[str]) -> dict:
        committed = [r for r in rows
                     if r[key].get("recommendation") in commits]
        n_c = len(committed)
        n_correct = sum(1 for r in committed if r["correct"])
        return {
            "n_committed": n_c,
            "n_correct": n_correct,
            "precision": (n_correct / n_c) if n_c else None,
            "coverage": n_c / len(rows),
        }

    summary = {
        "tag": tag, "bench": bench, "n_test": len(rows), "k": k,
        "naive_unanimous":    head("naive", {"COMMIT"}),
        "dajv":               head("dajv", {"COMMIT"}),
        "verge_proxy":        head("verge_proxy", {"COMMIT"}),
        "verge_proxy_mcs":    head("verge_proxy", {"COMMIT", "COMMIT_MCS"}),
        "verge_z3":           head("verge_z3", {"COMMIT"}),
        "verge_z3_mcs":       head("verge_z3", {"COMMIT", "COMMIT_MCS"}),
    }
    print(f"\n=== {tag} k={k} n_test={len(rows)} ===")
    for key in ("naive_unanimous", "dajv", "verge_proxy", "verge_proxy_mcs",
                "verge_z3", "verge_z3_mcs"):
        d = summary[key]
        prec = d["precision"]
        ps = f"{prec:.3f}" if prec is not None else "  ---"
        print(f"  {key:18s} cov={d['coverage']:.3f} prec={ps} "
              f"n={d['n_committed']}/{len(rows)}")
    return summary


def main() -> None:
    out = []
    for bench, tag in [("math175", "math175"), ("aime", "aime"),
                       ("cleanmath", "cleanmath"), (None, "full")]:
        try:
            out.append(run_on_bench(bench, tag))
        except Exception as e:
            print(f"  ERROR {tag}: {e}")
    save_artifact(out, ARTIFACTS / "verge_z3_compare.json")
    print(f"\nWrote {ARTIFACTS / 'verge_z3_compare.json'}")


if __name__ == "__main__":
    main()

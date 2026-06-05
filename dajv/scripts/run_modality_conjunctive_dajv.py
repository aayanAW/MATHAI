"""Modality-conjunctive DAJV: intersect two DAJV operating points.

Hypothesis: the cross-modality independence finding implies that
requiring DAJV-style COMMIT in BOTH the structural modality (script
classifies as working) and the executable modality (script accepts
candidate) should produce a stricter gate that improves precision
at high-coverage thresholds.

We fit two independent DajvCalibrations on the 6-extractor cache
(struct-only and exec-only), and COMMIT iff both fire.

Saves: artifacts/modality_conjunctive.json
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from verifyensemble.aggregate.dajv import DajvCalibration, dajv_aggregate
from verifyensemble.evaluation.ece import expected_calibration_error
from verifyensemble.utils.io import align_extractor_caches, save_artifact

GROUP = {
    "E05_gpt_oss_120B":      HERE.parent / "../MATHAI/results/exp46_gptoss_extractor.json",
    "E06_gpt_5_mini":        HERE.parent / "../MATHAI/results/exp50_gpt5_extractor.json",
    "E07_claude_sonnet_4_6": HERE.parent / "../MATHAI/results/exp47_claude_extractor.json",
    "E09_qwen3_coder_480B":  HERE.parent / "../MATHAI/results/exp48_qwen3coder_extractor.json",
    "E08S_gpt_4o":           HERE.parent / "../MATHAI/results/exp55_gpt4o_extractor.json",
    "E04S_gpt_4_1":          HERE.parent / "../MATHAI/results/exp56_gpt41_extractor.json",
}
_GPT5 = HERE.parent / "../MATHAI/results/exp57_gpt5_extractor.json"
if _GPT5.exists():
    try:
        import json as _j
        if len(_j.load(open(_GPT5)).get("results", [])) >= 300:
            GROUP["E10S_gpt_5"] = _GPT5
    except Exception:
        pass
_OPUS47 = HERE.parent / "../MATHAI/results/exp58_opus47_extractor.json"
if _OPUS47.exists():
    try:
        import json as _j
        if len(_j.load(open(_OPUS47)).get("results", [])) >= 300:
            GROUP["E01A_claude_opus_4_7"] = _OPUS47
    except Exception:
        pass
_OPUS46 = HERE.parent / "../MATHAI/results/exp59_opus46_extractor.json"
if _OPUS46.exists():
    try:
        import json as _j
        if len(_j.load(open(_OPUS46)).get("results", [])) >= 300:
            GROUP["E02A_claude_opus_4_6"] = _OPUS46
    except Exception:
        pass
_HAIKU45 = HERE.parent / "../MATHAI/results/exp61_haiku45_extractor.json"
if _HAIKU45.exists():
    try:
        import json as _j
        if len(_j.load(open(_HAIKU45)).get("results", [])) >= 300:
            GROUP["E03A_claude_haiku_4_5"] = _HAIKU45
    except Exception:
        pass
_LLAMA33 = HERE.parent / "../MATHAI/results/exp62_llama33_70b_extractor.json"
if _LLAMA33.exists():
    try:
        import json as _j
        if len(_j.load(open(_LLAMA33)).get("results", [])) >= 300:
            GROUP["E13_llama_3_3_70B"] = _LLAMA33
    except Exception:
        pass
_QWEN235 = HERE.parent / "../MATHAI/results/exp63_qwen3_235b_extractor.json"
if _QWEN235.exists():
    try:
        import json as _j
        if len(_j.load(open(_QWEN235)).get("results", [])) >= 300:
            GROUP["E14_qwen_3_235B"] = _QWEN235
    except Exception:
        pass

ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def eval_one(bench: str | None, tag: str, seeds: list[int]) -> dict:
    aligned = align_extractor_caches(GROUP, bench=bench)
    n = len(aligned["problem_ids"])
    if n < 30:
        return {"tag": tag, "skipped": "too few"}
    k = len(aligned["extractor_ids"])
    eids = aligned["extractor_ids"]
    classification = aligned["classification"]
    cand_verdict = aligned["candidate_verdict"]
    accept = aligned["accept"]
    correct = [bool(c) for c in aligned["solver_correct"]]

    struct_signal = [[(classification[i][j] == "working")
                      for j in range(n)] for i in range(k)]
    exec_signal = [[(cand_verdict[i][j] is True)
                    for j in range(n)] for i in range(k)]
    accept_signal = [[bool(accept[i][j]) for j in range(n)] for i in range(k)]

    per_seed = []
    for seed in seeds:
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        cal_n = int(0.7 * n)
        cal_idx = indices[:cal_n]
        test_idx = indices[cal_n:]

        def _slice(arr, idx):
            return [[arr[i][j] for j in idx] for i in range(k)]

        struct_cal = _slice(struct_signal, cal_idx)
        struct_test = _slice(struct_signal, test_idx)
        exec_cal = _slice(exec_signal, cal_idx)
        exec_test = _slice(exec_signal, test_idx)
        accept_cal = _slice(accept_signal, cal_idx)
        accept_test = _slice(accept_signal, test_idx)
        correct_cal = [correct[j] for j in cal_idx]
        correct_test = [correct[j] for j in test_idx]

        cal_struct = DajvCalibration.fit(
            struct_cal, correct_cal,
            [f"s_{e}" for e in eids])
        cal_exec = DajvCalibration.fit(
            exec_cal, correct_cal,
            [f"x_{e}" for e in eids])
        cal_accept = DajvCalibration.fit(
            accept_cal, correct_cal,
            [f"a_{e}" for e in eids])

        n_test = len(test_idx)
        # Metrics
        confs_acc, ys = [], []
        n_acc_commit = n_acc_commit_correct = 0
        n_conj_commit = n_conj_commit_correct = 0
        confs_conj = []
        for j in range(n_test):
            v_s = [bool(struct_test[i][j]) for i in range(k)]
            v_x = [bool(exec_test[i][j]) for i in range(k)]
            v_a = [bool(accept_test[i][j]) for i in range(k)]
            out_s = dajv_aggregate(v_s, cal_struct)
            out_x = dajv_aggregate(v_x, cal_exec)
            out_a = dajv_aggregate(v_a, cal_accept)

            if out_a.get("P_correct") is not None:
                confs_acc.append(out_a["P_correct"])
                ys.append(correct_test[j])
            if out_a.get("recommendation") == "COMMIT":
                n_acc_commit += 1
                if correct_test[j]:
                    n_acc_commit_correct += 1

            # Conjunctive: both struct AND exec DAJV commit
            conj_commit = (out_s.get("recommendation") == "COMMIT"
                           and out_x.get("recommendation") == "COMMIT")
            # Conjunctive posterior = product (independence assumption)
            ps, px = out_s.get("P_correct"), out_x.get("P_correct")
            if ps is not None and px is not None:
                conj_p = ps * px / (ps * px + (1-ps) * (1-px))
                confs_conj.append(conj_p)
            if conj_commit:
                n_conj_commit += 1
                if correct_test[j]:
                    n_conj_commit_correct += 1

        per_seed.append({
            "seed": seed,
            "accept_dajv": {
                "coverage": n_acc_commit / n_test if n_test else 0.0,
                "precision": (n_acc_commit_correct / n_acc_commit
                              if n_acc_commit else None),
                "n_commit": n_acc_commit,
                "ece": (expected_calibration_error(confs_acc, ys, n_bins=5)
                        if confs_acc else None),
            },
            "conj_dajv": {
                "coverage": n_conj_commit / n_test if n_test else 0.0,
                "precision": (n_conj_commit_correct / n_conj_commit
                              if n_conj_commit else None),
                "n_commit": n_conj_commit,
                "ece": (expected_calibration_error(confs_conj, ys, n_bins=5)
                        if confs_conj else None),
            },
        })

    def agg(method: str, metric: str):
        vals = [s[method][metric] for s in per_seed
                if s[method].get(metric) is not None]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        denom = max(len(vals) - 1, 1)
        v = sum((x - m) ** 2 for x in vals) / denom if len(vals) > 1 else 0.0
        return {"mean": m, "std": v ** 0.5, "n": len(vals)}

    return {
        "tag": tag,
        "bench": bench,
        "n_total": n,
        "k_extractors": k,
        "seeds": seeds,
        "accept_dajv_summary": {
            "coverage": agg("accept_dajv", "coverage"),
            "precision": agg("accept_dajv", "precision"),
            "ece": agg("accept_dajv", "ece"),
        },
        "conj_dajv_summary": {
            "coverage": agg("conj_dajv", "coverage"),
            "precision": agg("conj_dajv", "precision"),
            "ece": agg("conj_dajv", "ece"),
        },
        "per_seed": per_seed,
    }


def main() -> None:
    seeds = list(range(1, 11))
    results = []
    for bench, tag in [("math175", "math175"), (None, "B_full")]:
        r = eval_one(bench, tag, seeds)
        results.append(r)
        if r.get("skipped"):
            continue
        print(f"\n=== {tag} k={r['k_extractors']} ===")
        for name, key in [("default DAJV (accept)", "accept_dajv_summary"),
                          ("conj DAJV (struct & exec)", "conj_dajv_summary")]:
            s = r[key]
            cov = s["coverage"]
            prec = s["precision"]
            ece = s["ece"]
            covs = f"{cov['mean']:.3f}±{cov['std']:.3f}" if cov else "n/a"
            precs = f"{prec['mean']:.3f}±{prec['std']:.3f}" if prec else "n/a"
            eces = f"{ece['mean']:.3f}±{ece['std']:.3f}" if ece else "n/a"
            print(f"  {name:30s} cov={covs} prec={precs} ECE={eces}")
    save_artifact(results, ARTIFACTS / "modality_conjunctive.json")
    print(f"\nWrote {ARTIFACTS / 'modality_conjunctive.json'}")


if __name__ == "__main__":
    main()

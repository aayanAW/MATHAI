"""Experiment 33: Deployment-time adversarial FP filter, re-scored on exp30+31.

Does NOT call the extractor API. Re-runs only the deployment-time sanity check
on every extracted verifier script stored in exp30/31 result JSONs. Then
recomputes top-tier precision under the filter.

The evaluation-time adversarial tests in exp30/31 used gold±1, gold+7, 42.
The deployment-time filter uses candidate±1, candidate+7, candidate*2, candidate+100.
Different probe set → different filter outcomes. Expected: the 1 AIME false
positive (aime2025_29 geometry) gets filtered out because its verifier accepts
adversarial probes around the candidate too.

Outputs:
    results/exp33_xsgrv_filtered.json — per-problem filter verdicts + summary
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import deployment_time_sanity_check

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_PATH = RESULTS_DIR / "exp33_xsgrv_filtered.json"

SOURCE_FILES = {
    "aime_llama70b":      RESULTS_DIR / "exp30_aime_llama70b.json",
    "aime_qwen7b":        RESULTS_DIR / "exp30_aime_qwen7b.json",
    "math175_llama70b":   RESULTS_DIR / "exp31_xsgrv_math175_llama70b.json",
    "math50_qwen7b":      RESULTS_DIR / "exp30_math50_qwen7b.json",
    "cleanmath_llama70b": RESULTS_DIR / "exp34_cleanmath_llama70b.json",
    "aime_deepseek":      RESULTS_DIR / "exp32_aime_deepseek.json",
    "math175_deepseek":   RESULTS_DIR / "exp32_math175_deepseek.json",
}


def rescore_one(rec: dict) -> dict:
    """Apply the deployment-time filter to a single stored result.

    Returns the original record augmented with:
        filter_broken: bool
        filter_probes: list[dict]
        filtered_classification: str  (post-filter label)
    """
    base = dict(rec)
    base["filter_broken"] = None
    base["filter_probes"] = []
    base["filtered_classification"] = rec.get("classification")

    if rec.get("outcome") != "verifier_produced":
        return base
    script = rec.get("script")
    if not script:
        return base

    candidate = str(rec.get("candidate", ""))
    broken, probe_results = deployment_time_sanity_check(script, candidate, timeout=10.0)
    base["filter_broken"] = broken
    base["filter_probes"] = probe_results

    # Determine post-filter classification
    # If the filter says broken, we abstain → re-classify as "filtered_broken".
    # Otherwise classification stays the same.
    if broken:
        base["filtered_classification"] = "filtered_broken"
    return base


def summarize_condition(rescored: list[dict], label: str) -> dict:
    """Compute pre/post-filter stats for one condition."""
    n = len(rescored)
    pre_working = sum(1 for r in rescored if r.get("classification") == "working")
    pre_fp = sum(1 for r in rescored if r.get("classification") == "false_positive")

    post_working = sum(1 for r in rescored if r.get("filtered_classification") == "working")
    post_fp = sum(1 for r in rescored if r.get("filtered_classification") == "false_positive")
    filtered_out = sum(1 for r in rescored if r.get("filtered_classification") == "filtered_broken")

    # Pre-filter top tier (verifier said PASS on solver answer)
    pre_top_tier = [r for r in rescored if r.get("candidate_verdict") is True]
    pre_top_tier_correct = sum(1 for r in pre_top_tier if r.get("solver_correct"))

    # Post-filter top tier: same PASS, but only if the filter did NOT mark broken
    post_top_tier = [r for r in pre_top_tier if r.get("filter_broken") is not True]
    post_top_tier_correct = sum(1 for r in post_top_tier if r.get("solver_correct"))

    stats = {
        "label": label,
        "n": n,
        "pre_filter": {
            "working": pre_working,
            "false_positive": pre_fp,
            "top_tier_size": len(pre_top_tier),
            "top_tier_correct": pre_top_tier_correct,
            "top_tier_precision": pre_top_tier_correct / len(pre_top_tier) if pre_top_tier else None,
        },
        "post_filter": {
            "working": post_working,
            "false_positive": post_fp,
            "filtered_out": filtered_out,
            "top_tier_size": len(post_top_tier),
            "top_tier_correct": post_top_tier_correct,
            "top_tier_precision": post_top_tier_correct / len(post_top_tier) if post_top_tier else None,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"n = {n}")
    print(f"Pre-filter:  working={pre_working}  fp={pre_fp}  "
          f"top_tier={len(pre_top_tier)}/{n}  "
          f"prec={pre_top_tier_correct}/{len(pre_top_tier)}"
          f"{'' if not pre_top_tier else f' = {pre_top_tier_correct/len(pre_top_tier):.4f}'}")
    print(f"Post-filter: working={post_working}  fp={post_fp}  "
          f"filtered_out={filtered_out}  "
          f"top_tier={len(post_top_tier)}/{n}  "
          f"prec={post_top_tier_correct}/{len(post_top_tier)}"
          f"{'' if not post_top_tier else f' = {post_top_tier_correct/len(post_top_tier):.4f}'}")
    if filtered_out:
        bad_ids = [r["id"] for r in rescored if r.get("filtered_classification") == "filtered_broken"]
        print(f"Filtered-out IDs: {bad_ids}")
    return stats


def main():
    all_rescored = {}
    all_stats = {}

    for key, path in SOURCE_FILES.items():
        if not path.exists():
            print(f"SKIP {key}: {path.name} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        records = data.get("results", [])
        print(f"\n>>> Re-scoring {key}  ({len(records)} records from {path.name})")

        rescored = []
        for i, rec in enumerate(records):
            r = rescore_one(rec)
            rescored.append(r)
            if (i + 1) % 25 == 0:
                print(f"    ... rescored {i+1}/{len(records)}", flush=True)

        all_rescored[key] = rescored
        all_stats[key] = summarize_condition(rescored, label=key)

    # Drop verbose `script` and `filter_probes` fields from the saved rescored
    # records to keep the output file compact. Keep the filter verdict + id.
    compact = {}
    for key, recs in all_rescored.items():
        slim = []
        for r in recs:
            slim.append({
                "id": r.get("id"),
                "classification": r.get("classification"),
                "filtered_classification": r.get("filtered_classification"),
                "filter_broken": r.get("filter_broken"),
                "candidate": r.get("candidate"),
                "gold": r.get("gold"),
                "candidate_verdict": r.get("candidate_verdict"),
                "solver_correct": r.get("solver_correct"),
            })
        compact[key] = slim

    with open(OUT_PATH, "w") as f:
        json.dump({
            "summary": all_stats,
            "per_condition": compact,
        }, f, indent=2, default=str)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()

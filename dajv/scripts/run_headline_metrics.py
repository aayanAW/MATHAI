"""Compute the consolidated headline-metrics JSON used in the paper.

Reads all artifacts/*.json and produces a single
artifacts/headline_metrics.json that mirrors every number cited in
the abstract, headline tables, and discussion.

This is the load-bearing source of truth for paper numbers. If a paper
table disagrees with this file, the paper is wrong.

Saves: artifacts/headline_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ARTIFACTS = HERE.parent / "artifacts"


def _load(name: str):
    p = ARTIFACTS / name
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def main() -> None:
    out: dict = {}

    # H2 / H1: dependency on the 4-extractor cache
    dep = _load("dependency_matrix_B_math175.json")
    if dep:
        out["h2_math175"] = {
            "median_kappa": dep.get("median_kappa"),
            "median_joint_fp": dep.get("median_joint_fp"),
            "median_independence": dep.get("median_independence"),
            "median_ratio": dep.get("median_ratio"),
            "n_wrong": dep.get("n_wrong"),
        }

    # 4-extractor aggregation comparison (default baseline)
    agg = _load("aggregation_results.json")
    if agg:
        out["aggregation_4extractor"] = {
            r["tag"]: {
                "n_test": r.get("n_test"),
                "naive_unanimous": r.get("headline", {}).get("naive_unanimous"),
                "dajv": r.get("headline", {}).get("dajv"),
                "care": r.get("headline", {}).get("care"),
                "ece_naive": r.get("calibration", {}).get("naive_unanimous", {}).get("ece"),
                "ece_dajv": r.get("calibration", {}).get("dajv", {}).get("ece"),
            }
            for r in agg if isinstance(r, dict) and not r.get("skipped")
        }

    # 6-extractor aggregation (scale-up)
    agg6 = _load("aggregation_6extractor_full.json")
    if agg6:
        out["aggregation_6extractor"] = {
            r["tag"]: {
                "k": r.get("k_extractors"),
                "n_test": r.get("n_test"),
                "naive_unanimous": r.get("operating_points", {}).get("naive_unanimous"),
                "dajv": r.get("operating_points", {}).get("dajv"),
                "verge_proxy": r.get("operating_points", {}).get("verge_proxy_full"),
                "ece_dajv": r.get("calibration", {}).get("dajv", {}).get("ece"),
            }
            for r in agg6 if isinstance(r, dict) and not r.get("skipped")
        }

    # H6 significance
    h6 = _load("h6_significance.json")
    if h6:
        out["h6_significance"] = [
            {
                "tag": r.get("tag"),
                "k": r.get("k_extractors"),
                "n_within": r.get("within_lab", {}).get("n_pairs"),
                "n_cross": r.get("cross_lab", {}).get("n_pairs"),
                "within_median": r.get("within_lab", {}).get("median"),
                "within_ci": [r.get("within_lab", {}).get("ci95_low"),
                              r.get("within_lab", {}).get("ci95_high")],
                "cross_median": r.get("cross_lab", {}).get("median"),
                "cross_ci": [r.get("cross_lab", {}).get("ci95_low"),
                             r.get("cross_lab", {}).get("ci95_high")],
                "obs_diff": r.get("permutation_test", {}).get("observed_diff"),
                "p_value": r.get("permutation_test", {}).get("p_value"),
            }
            for r in h6 if isinstance(r, dict) and not r.get("skipped")
        ]

    # Cross-modality independence
    mod = _load("hybrid_modality.json")
    if mod:
        out["cross_modality"] = [
            {
                "tag": r.get("tag"),
                "k": len(r.get("extractor_ids", [])),
                "n_wrong": r.get("n_wrong"),
                "within_struct_median":   r.get("summary", {}).get("within_modality_struct", {}).get("median"),
                "within_exec_median":     r.get("summary", {}).get("within_modality_exec", {}).get("median"),
                "within_llm_cross_median": r.get("summary", {}).get("within_llm_cross_modality", {}).get("median"),
                "cross_llm_cross_median":  r.get("summary", {}).get("cross_llm_cross_modality", {}).get("median"),
            }
            for r in mod if isinstance(r, dict) and not r.get("skipped")
        ]

    # Copula-order ablation
    cop = _load("copula_ablation.json")
    if cop:
        out["copula_ablation"] = [
            {
                "tag": r.get("tag"),
                "indep_only_ece":   r.get("summary", {}).get("indep_only", {}).get("ece"),
                "second_order_ece": r.get("summary", {}).get("second_order", {}).get("ece"),
            }
            for r in cop if isinstance(r, dict) and not r.get("skipped")
        ]

    # VERGE proxy head-to-head
    verge = _load("verge_proxy_compare.json")
    if verge:
        out["verge_proxy"] = [
            {
                "tag": r.get("tag"),
                "naive_unanimous": r.get("naive_unanimous"),
                "verge_proxy_full": r.get("verge_proxy_full"),
                "dajv": r.get("dajv"),
            }
            for r in verge if isinstance(r, dict) and not r.get("skipped")
        ]

    # PRM head-to-head
    prm = _load("prm_head_to_head.json")
    if prm:
        out["prm_h2h"] = prm

    out_path = ARTIFACTS / "headline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")
    print(json.dumps({k: ("..." if isinstance(v, (list, dict)) else v)
                      for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()

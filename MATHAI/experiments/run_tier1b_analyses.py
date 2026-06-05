"""Tier 1b free analyses: CleanMath per-competition, trivial-verifier audit,
McNemar X-SGRV vs SE, filter-with-no-probes, error cases."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")


def load(name: str) -> dict:
    with open(RESULTS / name) as f:
        return json.load(f)


# =====================================================================
# A) CleanMath per-competition breakdown
# =====================================================================

def cleanmath_per_competition() -> dict:
    d = load("exp34_cleanmath_llama70b.json")
    res = d["results"]
    by_comp = defaultdict(list)
    for r in res:
        pid = r.get("id", "")
        # id format: hmmt_feb_2025_1, brumo_2025_1, smt_2025_1, apex_2025_1
        if pid.startswith("hmmt"):
            comp = "HMMT-Feb-2025"
        elif pid.startswith("brumo"):
            comp = "BRUMO-2025"
        elif pid.startswith("smt"):
            comp = "SMT-2025"
        elif pid.startswith("apex"):
            comp = "APEX-2025"
        else:
            comp = "unknown"
        by_comp[comp].append(r)

    out = {}
    for comp, rows in sorted(by_comp.items()):
        n = len(rows)
        solver_acc = sum(1 for r in rows if r.get("solver_correct")) / n if n else 0
        top_tier = [r for r in rows if r.get("candidate_verdict") is True]
        top_correct = [r for r in top_tier if r.get("solver_correct")]
        working = [r for r in rows if r.get("classification") == "working"]
        adv_fps = sum(r.get("adv_fp_count", 0) or 0 for r in rows)
        adv_tests = 4 * len(working)
        out[comp] = {
            "n": n,
            "solver_accuracy": solver_acc,
            "n_working_verifiers": len(working),
            "working_fraction": len(working) / n if n else 0,
            "n_top_tier": len(top_tier),
            "top_tier_coverage": len(top_tier) / n if n else 0,
            "top_tier_precision": len(top_correct) / len(top_tier) if top_tier else None,
            "adv_fp_rate": adv_fps / adv_tests if adv_tests else 0,
        }
    return out


# =====================================================================
# B) Trivial-verifier audit
# =====================================================================

def trivial_verifier_audit(fn: str, n_benchmark: int) -> dict:
    """Classify each verifier script by complexity.

    - trivial: `return answer == <literal>` or equivalent hardcoded comparison
    - simple_sympy: uses sympy/math/Fraction for a single comparison
    - constraint_check: evaluates an equation/inequality the problem specifies
    - enumeration: loops over a bounded set
    - compound: multiple checks and logic
    """
    d = load(fn)
    res = d["results"]
    classes = Counter()
    total_working = 0
    examples = {k: [] for k in ["trivial", "simple_sympy", "constraint_check", "enumeration", "compound"]}
    for r in res:
        s = (r.get("script") or "").strip()
        if r.get("classification") != "working":
            continue
        total_working += 1
        # Strip def verify line + try/except boilerplate
        body = "\n".join(s.split("\n")[1:])
        # Check for trivial literal equality
        normalized = re.sub(r"\s", "", body)
        trivial_patterns = [
            r"returnanswer==-?\d+",
            r"returnint\(answer\)==-?\d+",
            r"returnstr\(answer\)==['\"].*?['\"]",
            r"returnanswer==['\"].*?['\"]",
            r"returnfloat\(answer\)==-?\d+",
        ]
        is_trivial = any(re.search(p, normalized) for p in trivial_patterns)
        has_loop = bool(re.search(r"\bfor\b|\bwhile\b|itertools", body))
        has_and_or = bool(re.search(r"\band\b|\bor\b|\bif\b.*:", body)) and "if" in body.replace("except:", "").replace("except Exception:", "")
        has_multiple_returns = body.count("return ") > 1
        has_sympy = "sympy" in s or "sp." in s or "Symbol" in s
        has_math = "math." in s
        has_eq_check = bool(re.search(r"==|<=|>=|<[^=]|>[^=]", body))

        if is_trivial:
            classes["trivial"] += 1
            examples["trivial"].append(r["id"])
        elif has_loop:
            classes["enumeration"] += 1
            examples["enumeration"].append(r["id"])
        elif has_multiple_returns or (has_and_or and has_eq_check):
            classes["compound"] += 1
            examples["compound"].append(r["id"])
        elif has_eq_check and (has_sympy or has_math):
            classes["simple_sympy"] += 1
            examples["simple_sympy"].append(r["id"])
        elif has_eq_check:
            classes["constraint_check"] += 1
            examples["constraint_check"].append(r["id"])
        else:
            classes["compound"] += 1
            examples["compound"].append(r["id"])

    return {
        "file": fn,
        "total_working_verifiers": total_working,
        "classes": dict(classes),
        "fractions": {k: v / total_working for k, v in classes.items()} if total_working else {},
        "example_ids": {k: v[:3] for k, v in examples.items()},
    }


# =====================================================================
# C) McNemar X-SGRV vs SE on MATH-175 matched coverage
# =====================================================================

def mcnemar_xsgrv_vs_se() -> dict:
    """Match by problem id; for each problem, X-SGRV verdict and SE verdict."""
    xs = load("exp31_xsgrv_math175_llama70b.json")
    se = load("exp35_semantic_entropy.json")
    # X-SGRV: per-problem, top-tier accept = candidate_verdict == True AND working
    xsgrv_accept = {}
    xsgrv_correct = {}
    for r in xs["results"]:
        pid = r["id"]
        top = r.get("candidate_verdict") is True
        correct = r.get("solver_correct") is True
        xsgrv_accept[pid] = top
        xsgrv_correct[pid] = correct if top else None

    se_accept = {}
    se_correct = {}
    for r in se["rows"]:
        if r.get("bench") != "math175":
            continue
        pid = r["id"]
        ent = r.get("se_math", {}).get("entropy", 999)
        top = ent < 0.01
        correct = r.get("plurality_correct", False)
        se_accept[pid] = top
        se_correct[pid] = correct if top else None

    common = set(xsgrv_accept) & set(se_accept)
    # Restrict to problems where both produce a top-tier verdict (matched coverage intersection).
    both_top = {pid for pid in common if xsgrv_accept[pid] and se_accept[pid]}
    xs_top_only = {pid for pid in common if xsgrv_accept[pid] and not se_accept[pid]}
    se_top_only = {pid for pid in common if se_accept[pid] and not xsgrv_accept[pid]}
    neither = {pid for pid in common if not xsgrv_accept[pid] and not se_accept[pid]}

    # Within both_top, how many are correct by each? They are the same problems, so both verdicts of "accept" map to whether the plurality answer is correct.
    # Correct-agreement table for the intersection
    xs_c = sum(1 for pid in both_top if xsgrv_correct[pid])
    se_c = sum(1 for pid in both_top if se_correct[pid])

    # Over all common problems: construct 2x2 for correctness-vs-abstain
    # x-axis: XSGRV accept-and-correct vs not
    # y-axis: SE accept-and-correct vs not
    a = sum(1 for pid in common if xsgrv_accept[pid] and xsgrv_correct[pid]
            and se_accept[pid] and se_correct[pid])
    b = sum(1 for pid in common if xsgrv_accept[pid] and xsgrv_correct[pid]
            and not (se_accept[pid] and se_correct[pid]))
    c = sum(1 for pid in common if not (xsgrv_accept[pid] and xsgrv_correct[pid])
            and se_accept[pid] and se_correct[pid])
    dd = sum(1 for pid in common if not (xsgrv_accept[pid] and xsgrv_correct[pid])
             and not (se_accept[pid] and se_correct[pid]))

    table = [[a, b], [c, dd]]
    result = {
        "n_common": len(common),
        "intersection_top_tier": len(both_top),
        "xsgrv_only_top_tier": len(xs_top_only),
        "se_only_top_tier": len(se_top_only),
        "both_correct_in_intersection": min(xs_c, se_c),
        "mcnemar_table": table,
    }

    if HAS_STATSMODELS and (b + c) > 0:
        m = mcnemar(table, exact=True)
        result["mcnemar_pvalue"] = float(m.pvalue)
        result["mcnemar_statistic"] = float(m.statistic)
    else:
        result["mcnemar_pvalue"] = None
    return result


# =====================================================================
# D) Filter-with-no-probes ablation row
# =====================================================================

def filter_no_probes_row() -> dict:
    """Raw X-SGRV numbers (no filter, no consensus) computed from exp31."""
    d = load("exp31_xsgrv_math175_llama70b.json")
    res = d["results"]
    top_tier = [r for r in res if r.get("candidate_verdict") is True]
    correct = [r for r in top_tier if r.get("solver_correct")]
    n = len(res)
    return {
        "mechanism": "no_filter_no_consensus",
        "benchmark": "MATH-500 n=175 (Llama extractor)",
        "top_tier_size": len(top_tier),
        "coverage": len(top_tier) / n,
        "top_tier_precision": len(correct) / len(top_tier) if top_tier else None,
        "ci95": tuple(binomtest(len(correct), len(top_tier)).proportion_ci(0.95, "exact")) if top_tier else None,
    }


# =====================================================================
# E) Error-case inspection (the 5 MATH FPs caught by the filter)
# =====================================================================

def error_cases_math_fps() -> list[dict]:
    """Find the problems where Llama's raw verifier accepts a wrong candidate."""
    d = load("exp31_xsgrv_math175_llama70b.json")
    fps = []
    for r in d["results"]:
        if r.get("candidate_verdict") is True and r.get("solver_correct") is False:
            fps.append({
                "id": r["id"],
                "gold": r.get("gold"),
                "candidate": r.get("candidate"),
                "script_excerpt": (r.get("script") or "")[:300],
                "adv_fp_count": r.get("adv_fp_count"),
            })
    return fps


def main():
    out = {}
    print("[1/5] CleanMath per-competition breakdown...")
    out["cleanmath_per_competition"] = cleanmath_per_competition()

    print("[2/5] Trivial-verifier audit (exp31 Llama, exp32 DeepSeek, exp34 CleanMath)...")
    out["trivial_audit"] = {
        "llama_math175": trivial_verifier_audit("exp31_xsgrv_math175_llama70b.json", 175),
        "deepseek_math175": trivial_verifier_audit("exp32_math175_deepseek.json", 175),
        "llama_cleanmath": trivial_verifier_audit("exp34_cleanmath_llama70b.json", 125),
    }

    print("[3/5] McNemar X-SGRV vs SE on MATH-175...")
    out["mcnemar"] = mcnemar_xsgrv_vs_se()

    print("[4/5] Filter-with-no-probes (raw X-SGRV) ablation row...")
    out["filter_no_probes"] = filter_no_probes_row()

    print("[5/5] Error-case inspection on MATH-175 FPs...")
    out["math_fps"] = error_cases_math_fps()

    with open(RESULTS / "tier1b_analyses.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {RESULTS/'tier1b_analyses.json'}")

    # Print human summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nCleanMath per-competition:")
    for comp, r in out["cleanmath_per_competition"].items():
        prec = r["top_tier_precision"]
        prec_str = f"{prec:.2f}" if prec is not None else "n/a"
        print(f"  {comp}: n={r['n']}, solver={r['solver_accuracy']:.0%}, working={r['working_fraction']:.0%}, tier={r['n_top_tier']}, prec={prec_str}")

    print("\nTrivial-verifier audit (fractions of working verifiers):")
    for key, v in out["trivial_audit"].items():
        print(f"  {key}: total_working={v['total_working_verifiers']}")
        for cls, frac in sorted(v["fractions"].items(), key=lambda x: -x[1]):
            print(f"    {cls}: {frac:.2%}")

    print("\nMcNemar X-SGRV vs SE (MATH-175):")
    m = out["mcnemar"]
    print(f"  n_common={m['n_common']}")
    print(f"  intersection top-tier={m['intersection_top_tier']}")
    print(f"  XSGRV-only top-tier={m['xsgrv_only_top_tier']}")
    print(f"  SE-only top-tier={m['se_only_top_tier']}")
    print(f"  McNemar p={m['mcnemar_pvalue']}")

    print(f"\nRaw X-SGRV ablation row (no filter, no consensus):")
    fnp = out["filter_no_probes"]
    print(f"  {fnp['benchmark']}: tier={fnp['top_tier_size']}, cov={fnp['coverage']:.2%}, prec={fnp['top_tier_precision']:.3f}, CI={fnp['ci95']}")

    print(f"\n{len(out['math_fps'])} raw MATH FPs to inspect:")
    for fp in out["math_fps"]:
        print(f"  {fp['id']}: gold={fp['gold']}, candidate={fp['candidate']}, adv_fps={fp['adv_fp_count']}")


if __name__ == "__main__":
    main()

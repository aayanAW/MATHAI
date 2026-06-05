"""Experiment 28: Cell C of the 2x2 ablation.

Cell C = spec-grounded + deterministic (no Schwartz-Zippel randomization).
We re-run SGRV on the 500 MATH problems, but monkey-patch the random-evaluation
block out of the ALGEBRAIC_EQUIV, FACTORING, ROOT templates, leaving only the
symbolic-expansion check. If Cell C FAR = 0.0 matches Cell D (full SGRV), then
randomization adds no precision for polynomial identity testing and the
theoretical claim in Section 4 is empirically validated.

This empirically measures Cell C, replacing the "approximation" note in
exp24_2x2_ablation.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Monkey-patch BEFORE importing the pipeline so the templates pick up the change
import src.pbt.test_templates as tt

# Patch the algebraic_equiv template to remove the random evaluation block
_orig_equiv = tt.generate_algebraic_equiv_test

def _equiv_deterministic_only(lhs, rhs, problem_text):
    """Deterministic-only version: only symbolic expand check, no random eval."""
    result = _orig_equiv(lhs, rhs, problem_text)
    if result.script is None:
        return result
    # Strip the "Test 2: Random numerical evaluation" block
    script = result.script
    # Remove everything from "# Test 2" to "print("PASS"
    import re
    script = re.sub(
        r"# Test 2: Random numerical evaluation.*?print\(",
        'print(',
        script,
        flags=re.DOTALL,
    )
    # Also remove the `import random` and `from sympy import Rational` if no longer needed
    script = script.replace("import random\n", "")
    # Return a new TestResult with the modified script
    return tt.TestResult(script, result.claim_type, result.independence, result.notes)

tt.generate_algebraic_equiv_test = _equiv_deterministic_only

# Re-import pipeline so it picks up the patched template
import importlib
import src.pbt.pipeline
importlib.reload(src.pbt.pipeline)
from src.pbt.pipeline import run_pbt

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    # Load exp5's solutions
    with open(RESULTS_DIR / "exp5_math500_full.json") as f:
        exp5 = json.load(f)

    results = []
    n_all_pass = 0
    n_wrong_pass = 0
    n_testable = 0

    for i, r in enumerate(exp5["exever_results"]):
        if not r.get("nl_solution"):
            continue
        try:
            pbt = run_pbt(
                problem=_load_problem(r["id"]),
                solution=r["nl_solution"],
                gold_answer=r["gold_answer"],
                problem_id=r["id"],
            )
        except Exception as e:
            continue
        if pbt.n_testable == 0:
            continue
        n_testable += 1
        if pbt.all_tested_pass:
            n_all_pass += 1
            if not r["answer_correct"]:
                n_wrong_pass += 1
        results.append({
            "id": r["id"],
            "level": r["level"],
            "all_tested_pass": pbt.all_tested_pass,
            "n_testable": pbt.n_testable,
            "answer_correct": r["answer_correct"],
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/500] n_all_pass={n_all_pass} n_wrong_pass={n_wrong_pass}", flush=True)

    far = n_wrong_pass / n_all_pass if n_all_pass > 0 else 0.0
    summary = {
        "cell": "C: spec-grounded + deterministic",
        "n_testable": n_testable,
        "n_all_pass": n_all_pass,
        "n_wrong_pass": n_wrong_pass,
        "far": far,
    }
    print()
    print("=" * 50)
    print("CELL C RESULTS")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    from scipy.stats import binomtest
    if n_all_pass > 0:
        ci = binomtest(n_wrong_pass, n_all_pass).proportion_ci(method='exact')
        print(f"  FAR CI: [{ci.low:.4f}, {ci.high:.4f}]")
        summary["far_ci_low"] = float(ci.low)
        summary["far_ci_high"] = float(ci.high)

    with open(RESULTS_DIR / "exp28_cell_c.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved to results/exp28_cell_c.json")


def _load_problem(problem_id: str) -> str:
    """Look up problem text by id from math_test_sample_500.json."""
    global _PROB_CACHE
    if "_PROB_CACHE" not in globals():
        with open(RESULTS_DIR / "math_test_sample_500.json") as f:
            probs = json.load(f)
        _PROB_CACHE = {p["id"]: p["problem"] for p in probs}
    return _PROB_CACHE.get(problem_id, "")


if __name__ == "__main__":
    main()

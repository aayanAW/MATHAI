"""Experiment 31: X-SGRV on full n=175 MATH-500 stratified sample.

Extends exp30's MATH-500 × Llama-70B condition from n=50 to n=175
(all problems from exp25_selective_prediction.json, same stratification).

Seeds from exp30_math50_llama70b.json to avoid re-running the 50 already done.
Saves to results/exp31_xsgrv_math175_llama70b.json.

Budget: ~$0.75 (125 new problems × ~2k tokens × $0.003/k)
Time:   ~25 min wall clock at median 5s per problem
"""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier

EXTRACTOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_PATH = RESULTS_DIR / "exp31_xsgrv_math175_llama70b.json"

api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=2,
)


def make_extractor():
    def _call(prompt):
        resp = client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content
    return _call


def adversarial_candidates(gold_answer: str) -> list[str]:
    """Generate 4 adversarial wrong candidates near the gold answer."""
    try:
        g = int(str(gold_answer).strip())
        return [str(g - 1), str(g + 1), str(g + 7), "42" if g != 42 else "41"]
    except Exception:
        return ["0", "1", "100", "42"]


def process_problem(prob: dict, candidate: str, solver_correct: bool, gold: str, extractor_fn) -> dict:
    """Run extraction + verification + adversarial testing on one problem."""
    t0 = time.time()
    ext = extract_verifier(prob["problem"], extractor_fn)
    ext_time = time.time() - t0

    if ext.unverifiable:
        return {
            "id": prob["id"],
            "outcome": "unverifiable",
            "script": None,
            "elapsed": ext_time,
        }
    if ext.error or ext.script is None:
        return {
            "id": prob["id"],
            "outcome": "extraction_error",
            "script": None,
            "error": ext.error,
            "elapsed": ext_time,
        }

    # Test against gold, candidate, and 4 adversarial values
    gold_str = str(gold)
    tests = [("gold", gold_str), ("candidate", str(candidate))]
    for i, adv in enumerate(adversarial_candidates(gold)):
        if adv != gold_str:
            tests.append((f"adv{i}", adv))

    test_results = {}
    for label, val in tests:
        ver = execute_verifier(ext.script, val, timeout=10.0)
        test_results[label] = {
            "value": val,
            "verdict": ver.verdict,
            "error": ver.error,
            "exec_time": ver.execution_time,
        }

    gold_verdict = test_results["gold"]["verdict"]
    cand_verdict = test_results["candidate"]["verdict"]
    adv_verdicts = [test_results[k]["verdict"] for k in test_results if k.startswith("adv")]

    working = gold_verdict is True and all(v is False for v in adv_verdicts)
    adv_fp = sum(1 for v in adv_verdicts if v is True)
    compute_abstain = gold_verdict is None or any(v is None for v in adv_verdicts)
    logic_broken = gold_verdict is False and adv_fp == 0

    if working:
        classification = "working"
    elif adv_fp > 0:
        classification = "false_positive"
    elif compute_abstain:
        classification = "compute_abstain"
    elif logic_broken:
        classification = "logic_broken"
    else:
        classification = "other"

    return {
        "id": prob["id"],
        "outcome": "verifier_produced",
        "classification": classification,
        "script_chars": len(ext.script),
        "script": ext.script,
        "gold": gold_str,
        "candidate": str(candidate),
        "solver_correct": solver_correct,
        "gold_verdict": gold_verdict,
        "candidate_verdict": cand_verdict,
        "adv_verdicts": adv_verdicts,
        "adv_fp_count": adv_fp,
        "elapsed": time.time() - t0,
    }


def summarize(results: list[dict], label: str) -> None:
    n = len(results)
    unver = sum(1 for r in results if r.get("outcome") == "unverifiable")
    ext_err = sum(1 for r in results if r.get("outcome") == "extraction_error")
    produced = sum(1 for r in results if r.get("outcome") == "verifier_produced")
    working = sum(1 for r in results if r.get("classification") == "working")
    compute_abstain = sum(1 for r in results if r.get("classification") == "compute_abstain")
    logic_broken = sum(1 for r in results if r.get("classification") == "logic_broken")
    fp = sum(1 for r in results if r.get("classification") == "false_positive")

    total_adv_tests = 0
    adv_fp_total = 0
    for r in results:
        if r.get("outcome") != "verifier_produced":
            continue
        for v in (r.get("adv_verdicts") or []):
            if v is not None:
                total_adv_tests += 1
                if v is True:
                    adv_fp_total += 1

    top_tier = [r for r in results if r.get("candidate_verdict") is True]
    top_tier_correct = sum(1 for r in top_tier if r.get("solver_correct"))

    print(f"\n{'=' * 60}")
    print(f"{label}  (n={n})")
    print(f"{'=' * 60}")
    print(f"Extraction outcomes:")
    print(f"  UNVERIFIABLE:      {unver}")
    print(f"  Extraction error:  {ext_err}")
    print(f"  Verifier produced: {produced}")
    print(f"Verifier classifications:")
    print(f"  working:           {working}  ({100*working/n:.1f}%)")
    print(f"  logic_broken:      {logic_broken}")
    print(f"  compute_abstain:   {compute_abstain}")
    print(f"  false_positive:    {fp}")
    if total_adv_tests:
        print(f"Adversarial FP rate: {adv_fp_total}/{total_adv_tests} = {adv_fp_total/total_adv_tests*100:.2f}%")
    else:
        print("Adversarial FP rate: N/A")
    print(f"Candidate-verdict top tier:")
    print(f"  verifier PASS on solver answer: {len(top_tier)} ({100*len(top_tier)/n:.1f}%)")
    if top_tier:
        print(f"  top-tier precision: {top_tier_correct}/{len(top_tier)} = {top_tier_correct/len(top_tier):.4f}")
        # Clopper-Pearson 95% CI
        from scipy.stats import binomtest
        res = binomtest(top_tier_correct, len(top_tier))
        ci = res.proportion_ci(confidence_level=0.95, method="exact")
        print(f"  top-tier precision 95% CI: [{ci.low:.4f}, {ci.high:.4f}]")
    else:
        print("  top-tier empty")


def main():
    # --- Load problem pool (all 175 exp25 problems) ---
    with open(RESULTS_DIR / "exp25_selective_prediction.json") as f:
        exp25 = json.load(f)
    exp25_by_id = {r["id"]: r for r in exp25}

    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    math_by_id = {p["id"]: p for p in math_all}

    # All 175 exp25 IDs, ordered by level then original list order for reproducibility
    all_ids = [r["id"] for r in exp25]
    all_probs = [math_by_id[i] for i in all_ids if i in math_by_id]
    assert len(all_probs) == 175, f"Expected 175 exp25 problems, got {len(all_probs)}"
    print(f"Total exp25 problems: {len(all_probs)}")

    # --- Seed from exp30 math50 results (50 already done) ---
    results: list[dict] = []
    seed_path = RESULTS_DIR / "exp30_math50_llama70b.json"
    if seed_path.exists():
        try:
            existing = json.load(open(seed_path))
            if "results" in existing:
                results = existing["results"]
                print(f"Seeded {len(results)} results from {seed_path.name}")
        except Exception as e:
            print(f"Warning: could not load seed from {seed_path.name}: {e}")

    # Also load any partial exp31 progress
    if OUT_PATH.exists():
        try:
            partial = json.load(open(OUT_PATH))
            if "results" in partial:
                partial_results = partial["results"]
                # Merge: prefer exp31 entries over exp30 seed for same IDs
                partial_ids = {r["id"] for r in partial_results}
                results = [r for r in results if r["id"] not in partial_ids] + partial_results
                print(f"Merged {len(partial_results)} partial exp31 results (total so far: {len(results)})")
        except Exception as e:
            print(f"Warning: could not load partial exp31 results: {e}")

    done_ids = {r["id"] for r in results}
    to_do = [p for p in all_probs if p["id"] not in done_ids]
    print(f"Already done: {len(done_ids)}  To do: {len(to_do)}")

    if not to_do:
        print("All problems already processed!")
        summarize(results, "MATH-500 n=175 × Llama-70B (CROSS-FAMILY) — exp31")
        return

    extractor_fn = make_extractor()

    for prob in to_do:
        er = exp25_by_id[prob["id"]]
        candidate = er["sample0_answer"]
        solver_correct = er["sample0_correct"]
        gold = prob["answer"]

        r = process_problem(prob, candidate, solver_correct, gold, extractor_fn)
        results.append(r)

        if r["outcome"] == "verifier_produced":
            print(
                f"  [{len(results)}/{len(all_probs)}] {prob['id']}: {r['classification']} "
                f"(gold={r['gold_verdict']}, cand={r['candidate_verdict']}, "
                f"adv_fp={r['adv_fp_count']}, {r['elapsed']:.1f}s)",
                flush=True,
            )
        else:
            print(f"  [{len(results)}/{len(all_probs)}] {prob['id']}: {r['outcome']} ({r['elapsed']:.1f}s)", flush=True)

        # Save after every problem
        with open(OUT_PATH, "w") as f:
            json.dump({"extractor": EXTRACTOR_MODEL, "results": results}, f, indent=2, default=str)

    summarize(results, "MATH-500 n=175 × Llama-70B (CROSS-FAMILY) — exp31")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()

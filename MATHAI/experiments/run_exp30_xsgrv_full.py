"""Experiment 30: Full X-SGRV evaluation.

Four conditions:
1. AIME 2025 × Llama-3.3-70B extractor (cross-family) — main result
2. AIME 2025 × Qwen2.5-7B-Instruct extractor (same-family) — ablation
3. MATH-500 subset (50 stratified) × Llama-3.3-70B extractor
4. MATH-500 subset × Qwen2.5-7B-Instruct extractor

For each problem, we measure:
- Extraction outcome (UNVERIFIABLE / script / error)
- Working verifier (accepts gold)
- Adversarial FP test (gold-1, gold+1, gold*2, 42)
- Qwen's actual answer outcome (accept / reject)

Computes:
- Coverage = fraction with working verifier
- Adversarial FP rate = false accepts / total adversarial tests on working verifiers
- Top-tier composition = problems where verifier says PASS on Qwen's answer
- Top-tier precision = fraction of top tier where Qwen's answer was actually correct
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier

EXTRACTOR_MODELS = {
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",  # same-family ablation
}

api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key, timeout=120.0, max_retries=2)


def make_extractor(model_name):
    def _call(prompt):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content
    return _call


def adversarial_candidates(gold_answer):
    """Generate 4 adversarial wrong candidates near the gold answer."""
    try:
        g = int(str(gold_answer).strip())
        return [str(g - 1), str(g + 1), str(g + 7), "42" if g != 42 else "41"]
    except Exception:
        return ["0", "1", "100", "42"]


def process_problem(prob, candidate_answer, solver_correct, extractor_fn, gold):
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

    # Test against gold, candidate, and 4 adversarial
    gold_str = str(gold)
    tests = [
        ("gold", gold_str),
        ("candidate", str(candidate_answer)),
    ]
    for i, adv in enumerate(adversarial_candidates(gold)):
        if adv != gold_str:
            tests.append((f"adv{i}", adv))

    test_results = {}
    for label, val in tests:
        ver = execute_verifier(ext.script, val, timeout=10.0)
        test_results[label] = {
            "value": val,
            "verdict": ver.verdict,  # True / False / None (error)
            "error": ver.error,
            "exec_time": ver.execution_time,
        }

    # Classify
    gold_verdict = test_results["gold"]["verdict"]
    cand_verdict = test_results["candidate"]["verdict"]
    adv_verdicts = [test_results[k]["verdict"] for k in test_results if k.startswith("adv")]

    working = gold_verdict is True and all(v is False for v in adv_verdicts)
    # FP = any adversarial test returns True (accepts wrong)
    adv_fp = sum(1 for v in adv_verdicts if v is True)
    # "Compute abstain" = errors on gold or adversarial (timeout, math error, etc.)
    compute_abstain = gold_verdict is None or any(v is None for v in adv_verdicts)
    # "Logic broken" = rejects gold but accepts none of adversarial (safe but useless)
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
        "candidate": str(candidate_answer),
        "solver_correct": solver_correct,
        "gold_verdict": gold_verdict,
        "candidate_verdict": cand_verdict,
        "adv_verdicts": adv_verdicts,
        "adv_fp_count": adv_fp,
        "elapsed": time.time() - t0,
    }


def run_condition(problems, candidate_fn, extractor_key, out_path, limit=None):
    extractor_fn = make_extractor(EXTRACTOR_MODELS[extractor_key])
    results = []
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            if "results" in existing:
                results = existing["results"]
        except Exception:
            pass
    done_ids = set(r["id"] for r in results)
    to_do = [p for p in problems if p["id"] not in done_ids]
    if limit:
        to_do = to_do[: max(0, limit - len(results))]

    print(f"\n=== Extractor: {extractor_key} ({EXTRACTOR_MODELS[extractor_key]}) ===")
    print(f"  Already done: {len(done_ids)}  To do: {len(to_do)}", flush=True)

    for i, prob in enumerate(to_do):
        candidate, solver_correct, gold = candidate_fn(prob)
        r = process_problem(prob, candidate, solver_correct, extractor_fn, gold)
        results.append(r)
        if r["outcome"] == "verifier_produced":
            print(f"  [{len(results)}] {prob['id']}: {r['classification']} "
                  f"(gold_verdict={r['gold_verdict']}, adv_fp={r['adv_fp_count']}, {r['elapsed']:.1f}s)",
                  flush=True)
        else:
            print(f"  [{len(results)}] {prob['id']}: {r['outcome']} ({r['elapsed']:.1f}s)", flush=True)
        # Save every problem
        with open(out_path, "w") as f:
            json.dump({"extractor": EXTRACTOR_MODELS[extractor_key], "results": results}, f, indent=2, default=str)

    return results


def summarize(results, label):
    n = len(results)
    unver = sum(1 for r in results if r.get("outcome") == "unverifiable")
    ext_err = sum(1 for r in results if r.get("outcome") == "extraction_error")
    produced = sum(1 for r in results if r.get("outcome") == "verifier_produced")
    working = sum(1 for r in results if r.get("classification") == "working")
    compute_abstain = sum(1 for r in results if r.get("classification") == "compute_abstain")
    logic_broken = sum(1 for r in results if r.get("classification") == "logic_broken")
    fp = sum(1 for r in results if r.get("classification") == "false_positive")

    # Adversarial FP rate across all scripts that produced verdicts on adv tests
    total_adv_tests = 0
    adv_fp_total = 0
    for r in results:
        if r.get("outcome") != "verifier_produced":
            continue
        adv = r.get("adv_verdicts") or []
        for v in adv:
            if v is not None:
                total_adv_tests += 1
                if v is True:
                    adv_fp_total += 1

    # Top-tier on Qwen's candidates: problems where candidate_verdict == True
    top_tier = [r for r in results if r.get("candidate_verdict") is True]
    top_tier_correct = sum(1 for r in top_tier if r.get("solver_correct"))

    print(f"\n{'=' * 60}")
    print(f"{label}  (n={n})")
    print(f"{'=' * 60}")
    print(f"Extractor outcomes:")
    print(f"  UNVERIFIABLE (extractor abstained): {unver}")
    print(f"  Extraction error:                   {ext_err}")
    print(f"  Verifier produced:                  {produced}")
    print(f"Verifier classifications:")
    print(f"  working (accepts gold, rejects 4 adv):     {working}")
    print(f"  logic_broken (rejects gold, also rejects adv): {logic_broken}")
    print(f"  compute_abstain (timeout/error):           {compute_abstain}")
    print(f"  false_positive (accepts an adv wrong):     {fp}")
    print(f"Adversarial FP rate: {adv_fp_total}/{total_adv_tests} = {adv_fp_total/total_adv_tests*100:.1f}%" if total_adv_tests else "  Adversarial FP rate: N/A")
    print(f"Candidate-verdict top tier:")
    print(f"  verifier PASS on Qwen's answer: {len(top_tier)}")
    if top_tier:
        print(f"  top-tier precision: {top_tier_correct}/{len(top_tier)} = {top_tier_correct/len(top_tier):.3f}")
    else:
        print(f"  top-tier empty")


def main():
    import random
    random.seed(42)

    # AIME 2025
    aime = json.load(open(Path(__file__).parent.parent / "results/aime_2025.json"))
    exp27 = json.load(open(Path(__file__).parent.parent / "results/exp27_aime2025_contamination.json"))
    exp27_by_id = {r["id"]: r for r in exp27["results"]}

    def aime_candidate(prob):
        er = exp27_by_id[prob["id"]]
        return er["sample0_answer"], er["sample0_correct"], prob["answer"]

    # MATH-500 stratified 50
    with open(Path(__file__).parent.parent / "results/exp25_selective_prediction.json") as f:
        exp25 = json.load(f)
    exp25_by_id = {r["id"]: r for r in exp25}
    with open(Path(__file__).parent.parent / "results/math_test_sample_500.json") as f:
        math_all = json.load(f)
    math_by_id = {p["id"]: p for p in math_all}

    math_50 = []
    for lv in [1, 2, 3, 4, 5]:
        lv_ids = [p["id"] for p in math_all if p.get("level") == lv and p["id"] in exp25_by_id]
        random.shuffle(lv_ids)
        math_50.extend(lv_ids[:10])

    math_50_probs = [math_by_id[i] for i in math_50]

    def math_candidate(prob):
        er = exp25_by_id[prob["id"]]
        return er["sample0_answer"], er["sample0_correct"], prob["answer"]

    OUT = Path(__file__).parent.parent / "results"
    results_files = {
        "aime_llama70b": OUT / "exp30_aime_llama70b.json",
        "aime_qwen7b": OUT / "exp30_aime_qwen7b.json",
        "math50_llama70b": OUT / "exp30_math50_llama70b.json",
        "math50_qwen7b": OUT / "exp30_math50_qwen7b.json",
    }

    # Run in order: AIME x Llama first (main result), then MATH x Llama, then the same-family ablations
    r_aime_llama = run_condition(aime, aime_candidate, "llama70b", results_files["aime_llama70b"])
    r_math_llama = run_condition(math_50_probs, math_candidate, "llama70b", results_files["math50_llama70b"])
    r_aime_qwen = run_condition(aime, aime_candidate, "qwen7b", results_files["aime_qwen7b"])
    r_math_qwen = run_condition(math_50_probs, math_candidate, "qwen7b", results_files["math50_qwen7b"])

    summarize(r_aime_llama, "AIME 2025 × Llama-70B extractor (CROSS-FAMILY)")
    summarize(r_math_llama, "MATH-500 n=50 × Llama-70B extractor (CROSS-FAMILY)")
    summarize(r_aime_qwen, "AIME 2025 × Qwen-7B extractor (SAME-FAMILY ablation)")
    summarize(r_math_qwen, "MATH-500 n=50 × Qwen-7B extractor (SAME-FAMILY ablation)")


if __name__ == "__main__":
    main()

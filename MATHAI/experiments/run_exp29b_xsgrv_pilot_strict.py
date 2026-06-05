"""Experiment 29b: X-SGRV pilot with STRICT prompt + adversarial FP testing.

Tests on 10 AIME 2025 problems. For each:
1. Extract verifier with strict prompt (more abstentions expected)
2. If extracted, test against Qwen's wrong answer → should REJECT
3. Test against the GOLD answer → should ACCEPT (this validates the verifier)
4. Test against 3 adversarial wrong answers (gold-1, gold+1, 42) → should REJECT all

A verifier is COUNTED as correct if: accepts gold AND rejects all 4 wrong candidates.
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
api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key, timeout=120.0, max_retries=2)


def llm_call(prompt):
    resp = client.chat.completions.create(
        model=EXTRACTOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.0,
    )
    return resp.choices[0].message.content


def main():
    with open(Path(__file__).parent.parent / "results/aime_2025.json") as f:
        problems = json.load(f)
    with open(Path(__file__).parent.parent / "results/exp27_aime2025_contamination.json") as f:
        exp27 = json.load(f)
    exp27_by_id = {r["id"]: r for r in exp27["results"]}

    pilot_problems = problems[:10]
    print(f"X-SGRV pilot (strict prompt): {len(pilot_problems)} AIME 2025 problems")
    print(f"Extractor: {EXTRACTOR_MODEL}")
    print("=" * 70, flush=True)

    results = []
    for i, prob in enumerate(pilot_problems):
        print(f"\n[{i+1}/{len(pilot_problems)}] {prob['id']}")
        print(f"  Gold: {prob['answer']}  Qwen: {exp27_by_id[prob['id']]['sample0_answer']} ({'right' if exp27_by_id[prob['id']]['sample0_correct'] else 'wrong'})")

        t0 = time.time()
        ext = extract_verifier(prob["problem"], llm_call)
        ext_time = time.time() - t0

        if ext.unverifiable:
            print(f"  Extractor: UNVERIFIABLE ({ext_time:.1f}s) — abstain")
            results.append({
                "id": prob["id"],
                "gold": prob["answer"],
                "unverifiable": True,
                "script": None,
            })
            continue
        if ext.error or ext.script is None:
            print(f"  Extractor ERROR: {ext.error}")
            results.append({
                "id": prob["id"],
                "unverifiable": False,
                "script": None,
                "error": ext.error,
            })
            continue

        print(f"  Extracted script: {len(ext.script)} chars ({ext_time:.1f}s)")

        # Test against gold + Qwen's answer + 3 adversarial
        gold = str(prob["answer"])
        qwen_ans = str(exp27_by_id[prob["id"]]["sample0_answer"])
        try:
            gold_int = int(gold)
            adversarial = [str(gold_int - 1), str(gold_int + 1), "42"]
        except Exception:
            adversarial = ["0", "100", "42"]

        tests = [("GOLD", gold, "accept"), ("QWEN", qwen_ans, "reject")] + \
                [(f"ADV{i}", a, "reject") for i, a in enumerate(adversarial)]

        test_results = []
        broken = False
        for label, val, expected in tests:
            ver = execute_verifier(ext.script, val, timeout=12.0)
            if ver.verdict is None:
                outcome = "ERROR"
                broken = True
            elif expected == "accept":
                outcome = "PASS" if ver.verdict else "FAIL-rejected-gold"
                if not ver.verdict: broken = True
            else:  # expected reject
                outcome = "PASS" if not ver.verdict else "FAIL-accepted-wrong"
                if ver.verdict: broken = True
            print(f"    {label}={val[:30]}: {outcome}")
            test_results.append({"label": label, "value": val, "expected": expected, "verdict": ver.verdict, "outcome": outcome, "error": ver.error})

        results.append({
            "id": prob["id"],
            "gold": prob["answer"],
            "qwen": qwen_ans,
            "unverifiable": False,
            "script": ext.script,
            "tests": test_results,
            "broken": broken,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n = len(results)
    unver = sum(1 for r in results if r.get("unverifiable"))
    script_produced = sum(1 for r in results if r.get("script"))
    working = sum(1 for r in results if r.get("script") and not r.get("broken"))
    broken = sum(1 for r in results if r.get("script") and r.get("broken"))

    print(f"Total: {n}")
    print(f"  UNVERIFIABLE (extractor abstained): {unver}")
    print(f"  Script produced: {script_produced}")
    print(f"    → Working (accepts gold, rejects all 4 wrong): {working}")
    print(f"    → Broken (rejected gold or accepted a wrong): {broken}")
    print()
    print(f"Effective coverage: {working}/{n} = {working/n:.1%}")
    print(f"Effective false-accept rate on working verifiers: 0/{working*4 if working else 1} (each working verifier tested on 4 wrong candidates)")

    with open(Path(__file__).parent.parent / "results/exp29b_xsgrv_pilot_strict.json", "w") as f:
        json.dump({
            "extractor": EXTRACTOR_MODEL,
            "n": n,
            "unverifiable": unver,
            "script_produced": script_produced,
            "working": working,
            "broken": broken,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to results/exp29b_xsgrv_pilot_strict.json")


if __name__ == "__main__":
    main()

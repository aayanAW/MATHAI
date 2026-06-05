"""Experiment 29: X-SGRV pilot on 5 AIME 2025 problems.

Tests whether a cross-family LLM extractor (Llama-3.3-70B) can produce
executable SymPy verifiers for AIME problems. The solver is Qwen2.5-7B-Instruct.

Success criteria:
- At least 2/5 problems yield a runnable verifier (not UNVERIFIABLE, not code errors)
- On those problems, verify(candidate) returns True/False deterministically
- The verifier should REJECT the wrong answers (Qwen gets 0/5 on early AIME, probably)
- The verifier should not hardcode answers (manually audit the scripts)

If pilot passes, run full exp30 on 30 AIME + 175 MATH.
"""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import run_xsgrv, extract_verifier, execute_verifier

EXTRACTOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
SOLVER_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"

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


def main():
    # Load AIME 2025 problems AND their solver answers
    with open(Path(__file__).parent.parent / "results/aime_2025.json") as f:
        problems = json.load(f)
    with open(Path(__file__).parent.parent / "results/exp27_aime2025_contamination.json") as f:
        exp27 = json.load(f)
    exp27_by_id = {r["id"]: r for r in exp27["results"]}

    pilot_problems = problems[:5]
    extractor_fn = make_extractor(EXTRACTOR_MODEL)

    print(f"X-SGRV pilot: {len(pilot_problems)} AIME 2025 problems")
    print(f"Extractor: {EXTRACTOR_MODEL}")
    print(f"Solver (for candidate answer): {SOLVER_MODEL}")
    print("=" * 70, flush=True)

    pilot_results = []
    for i, prob in enumerate(pilot_problems):
        print(f"\n[{i+1}/{len(pilot_problems)}] {prob['id']}")
        print(f"  Problem: {prob['problem'][:200]}")
        print(f"  Gold answer: {prob['answer']}")

        solver_result = exp27_by_id.get(prob["id"])
        candidate_answer = solver_result["sample0_answer"] if solver_result else "UNKNOWN"
        solver_correct = solver_result["sample0_correct"] if solver_result else None
        print(f"  Qwen's answer: {candidate_answer} ({'CORRECT' if solver_correct else 'WRONG'})")

        t0 = time.time()
        result = run_xsgrv(
            problem=prob["problem"],
            candidate_answer=candidate_answer,
            extractor_fn=extractor_fn,
            timeout=10.0,
        )
        dt = time.time() - t0

        ext = result["extraction"]
        ver = result["verification"]
        verdict = result["verdict"]
        cov_fires = result["cov_fires"]

        print(f"  Extraction:")
        if ext.unverifiable:
            print(f"    UNVERIFIABLE (model abstained)")
        elif ext.error:
            print(f"    ERROR: {ext.error}")
        else:
            print(f"    script len: {len(ext.script)} chars")
            # Print first 10 lines of script
            script_lines = ext.script.split('\n')[:10]
            for line in script_lines:
                print(f"      {line}")
            if len(ext.script.split('\n')) > 10:
                print(f"      ... ({len(ext.script.split(chr(10))) - 10} more lines)")

        print(f"  Verification:")
        if ver is None:
            print(f"    (no script to run)")
        elif not ver.executed:
            print(f"    EXECUTION FAILED: {ver.error}")
        elif ver.verdict is None:
            print(f"    RUNTIME ERROR: {ver.error}")
        else:
            verdict_str = "PASS" if ver.verdict else "FAIL"
            print(f"    verdict: {verdict_str}  (ran in {ver.execution_time:.2f}s)")
            # Sanity check: if solver was wrong, did we reject? If right, did we accept?
            if solver_correct is not None:
                if solver_correct and ver.verdict:
                    print(f"    ✓ Correctly ACCEPTED a right answer")
                elif solver_correct and not ver.verdict:
                    print(f"    ✗ WRONGLY REJECTED a right answer (false fail)")
                elif not solver_correct and ver.verdict:
                    print(f"    ✗✗ WRONGLY ACCEPTED a wrong answer (FALSE POSITIVE — dangerous!)")
                else:
                    print(f"    ✓ Correctly REJECTED a wrong answer")
        print(f"  Total time: {dt:.1f}s")

        pilot_results.append({
            "id": prob["id"],
            "gold": prob["answer"],
            "candidate": candidate_answer,
            "solver_correct": solver_correct,
            "extraction_unverifiable": ext.unverifiable,
            "extraction_error": ext.error,
            "extraction_script": ext.script,
            "verification_executed": ver.executed if ver else False,
            "verification_verdict": ver.verdict if ver else None,
            "verification_error": ver.error if ver else None,
            "elapsed": dt,
            "cov_fires": cov_fires,
        })

    # Summary
    print("\n" + "=" * 70)
    print("PILOT SUMMARY")
    print("=" * 70)
    n = len(pilot_results)
    extracted = sum(1 for r in pilot_results if not r["extraction_unverifiable"] and not r["extraction_error"])
    executed = sum(1 for r in pilot_results if r["verification_executed"])
    verdicts = sum(1 for r in pilot_results if r["verification_verdict"] is not None)
    correct_on_wrong = sum(1 for r in pilot_results if r["solver_correct"] is False and r["verification_verdict"] is False)
    correct_on_right = sum(1 for r in pilot_results if r["solver_correct"] is True and r["verification_verdict"] is True)
    false_pos = sum(1 for r in pilot_results if r["solver_correct"] is False and r["verification_verdict"] is True)
    false_neg = sum(1 for r in pilot_results if r["solver_correct"] is True and r["verification_verdict"] is False)

    print(f"Extraction: {extracted}/{n} produced a script (non-UNVERIFIABLE)")
    print(f"Execution: {executed}/{n} scripts ran without error")
    print(f"Verdicts: {verdicts}/{n} returned True or False")
    print()
    print(f"True negatives  (correctly rejected wrong): {correct_on_wrong}")
    print(f"True positives  (correctly accepted right): {correct_on_right}")
    print(f"False positives (accepted wrong): {false_pos}  {'← DANGER' if false_pos else ''}")
    print(f"False negatives (rejected right): {false_neg}")

    # Save
    with open(Path(__file__).parent.parent / "results/exp29_xsgrv_pilot.json", "w") as f:
        json.dump({
            "extractor": EXTRACTOR_MODEL,
            "solver": SOLVER_MODEL,
            "n": n,
            "extracted": extracted,
            "executed": executed,
            "verdicts": verdicts,
            "results": pilot_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to results/exp29_xsgrv_pilot.json")

    # Go/no-go
    print()
    if verdicts >= 2 and false_pos == 0:
        print("✓ PILOT PASSES: >=2 verdicts, 0 false positives.")
        print("  → Proceed to full exp30 run.")
    elif false_pos > 0:
        print("✗ PILOT FAILS: false positives detected. Extractor is not independent.")
        print("  → Investigate extraction prompt or pivot.")
    else:
        print(f"⚠ PILOT INCONCLUSIVE: only {verdicts} verdicts.")
        print("  → Try different extractor or prompt.")


if __name__ == "__main__":
    main()

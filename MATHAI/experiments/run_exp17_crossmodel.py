"""Experiment 17: Cross-model evaluation with Llama-3.3-70B.

Generate solutions with Llama, run PBT on them to show
the verification generalizes across model families.

Uses n=50 problems (subset) to conserve API credits.
"""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

RESULTS_DIR = Path("results")

SOLVE_PROMPT = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

## Step 2: [brief title]
[reasoning and computation for this step]

...continue for all steps...

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pbt.pipeline import run_pbt

    api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
    client = OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key)
    model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        all_problems = json.load(f)

    # Take 50 stratified problems (10 per level)
    problems = []
    for lv in [1, 2, 3, 4, 5]:
        lv_probs = [p for p in all_problems if p["level"] == lv]
        problems.extend(lv_probs[:10])

    print(f"Cross-model eval: {len(problems)} problems with {model}")
    print("=" * 60)

    results = []
    for i, prob in enumerate(problems):
        print(f"  [{i+1}/{len(problems)}] {prob['id']} (L{prob['level']})...", end=" ")

        # Generate solution with Llama
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": SOLVE_PROMPT.format(problem=prob["problem"])}],
                max_tokens=2048,
                temperature=0.0,
            )
            solution = resp.choices[0].message.content
        except Exception as e:
            print(f"API ERROR: {e}")
            break

        # Run PBT on Llama solution
        pbt_result = run_pbt(
            problem=prob["problem"],
            solution=solution,
            gold_answer=prob["answer"],
            problem_id=prob["id"],
        )

        results.append(pbt_result.to_dict())
        status = "PASS" if pbt_result.all_tested_pass and pbt_result.n_testable > 0 else "FAIL/UNTESTED"
        print(f"correct={pbt_result.answer_correct}, tested={pbt_result.n_testable}, {status}")

        time.sleep(0.3)

    # Compute metrics
    n = len(results)
    if n == 0:
        print("No results. Exiting.")
        return

    n_correct = sum(1 for r in results if r["answer_correct"])
    n_testable = sum(1 for r in results if r["n_testable"] > 0)
    all_pass = [r for r in results if r["all_tested_pass"] and r["n_testable"] > 0]
    n_all_pass = len(all_pass)
    n_fp = sum(1 for r in all_pass if not r["answer_correct"])

    avg_tested = sum(r["n_testable"] for r in results) / n
    avg_steps = sum(r["n_steps"] for r in results) / n

    print(f"\n{'='*60}")
    print(f"CROSS-MODEL RESULTS (Llama-3.3-70B, n={n})")
    print(f"{'='*60}")
    print(f"  Accuracy: {n_correct}/{n} ({n_correct/n*100:.1f}%)")
    print(f"  Problems with tests: {n_testable}/{n} ({n_testable/n*100:.1f}%)")
    print(f"  Avg steps/problem: {avg_steps:.1f}")
    print(f"  Avg tested steps: {avg_tested:.1f}")
    print(f"  ALL_PASS: {n_all_pass}/{n}")
    print(f"  FPVR: {n_fp}/{n_all_pass} ({n_fp/n_all_pass*100:.1f}%)" if n_all_pass > 0 else "  FPVR: N/A")
    if n_all_pass > 0:
        acc_pass = sum(1 for r in all_pass if r["answer_correct"])
        acc_rej = sum(1 for r in results if r not in all_pass and r["answer_correct"]) if len(results) > n_all_pass else 0
        n_rej = n - n_all_pass
        print(f"  PBT-accepted accuracy: {acc_pass}/{n_all_pass} ({acc_pass/n_all_pass*100:.1f}%)")
        print(f"  PBT-rejected accuracy: {acc_rej}/{n_rej} ({acc_rej/n_rej*100:.1f}%)" if n_rej > 0 else "")

    output = {
        "experiment": "exp17_crossmodel",
        "model": model,
        "n_problems": n,
        "accuracy": n_correct / n,
        "coverage": n_testable / n,
        "fpvr": n_fp / n_all_pass if n_all_pass > 0 else None,
        "n_all_pass": n_all_pass,
        "results": results,
    }

    out_path = RESULTS_DIR / "exp17_crossmodel.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

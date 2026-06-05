"""Experiment 14: Clean-benchmark sanity check on AIME problems.

Runs ExeVer pipeline on n=100 AIME problems (2024-2025) to confirm
coverage and FPVR metrics transfer to an uncontaminated benchmark.

Requires: TOGETHER_API_KEY environment variable.
Model: Qwen/Qwen2.5-7B-Instruct-Turbo (same as exp13 for consistency)

Usage:
    export TOGETHER_API_KEY=your_key
    python3 experiments/run_exp14_aime_sanity.py
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

RESULTS_DIR = Path("results")

# AIME 2024 problems (publicly available, post-training cutoff for most models)
# These are competition math problems NOT in MATH dataset
AIME_PROBLEMS = [
    # AIME I 2024 (first 10 problems)
    {"id": "aime_2024_I_1", "problem": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks at $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.", "answer": "204", "level": 3, "type": "algebra"},
    {"id": "aime_2024_I_2", "problem": "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.", "answer": "25", "level": 4, "type": "intermediate_algebra"},
    {"id": "aime_2024_I_3", "problem": "Alice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either $1$ token or $4$ tokens from the stack. Whoever removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ for which there exists a strategy for Bob that guarantees that Bob will win the game.", "answer": "809", "level": 4, "type": "counting_and_probability"},
    {"id": "aime_2024_I_4", "problem": "Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}$. $4$ numbers are randomly chosen from $S$. She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\frac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.", "answer": "116", "level": 5, "type": "counting_and_probability"},
    {"id": "aime_2024_I_5", "problem": "Rectangle $ABCD$ has dimensions $AB = 107$ and $BC = 16$. Point $E$ is the midpoint of $\\overline{AD}$. Point $F$ is on segment $\\overline{BC}$ so that $\\overline{EF} \\perp \\overline{BD}$ and $F$ is between $B$ and $C$. The length of $\\overline{EF}$ can be expressed as $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.", "answer": "104", "level": 4, "type": "geometry"},
]

# NOTE: For a complete n=100 AIME check, need to add AIME 2024 II + AIME 2025 I+II
# problems. These 5 serve as a proof-of-concept. Full set TBD.

SOLVE_PROMPT = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

## Step 2: [brief title]
[reasoning and computation for this step]

...continue for all steps...

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""

VERIFY_PROMPT = """Below is a step-by-step math solution. Write a Python/SymPy script that checks each step.

Rules:
- Start with: from sympy import *
- For each step write: # === STEP k === then assert statements
- Use expand() not simplify()
- End with: print("ANSWER:", answer)
- Output ONLY the Python code, nothing else.

Solution to verify:
{solution}"""


def main():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set. Set it and rerun.")
        print("  export TOGETHER_API_KEY=your_key")
        sys.exit(1)

    client = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=api_key,
    )
    model = "Qwen/Qwen2.5-7B-Instruct-Turbo"

    problems = AIME_PROBLEMS
    print(f"Running AIME sanity check on {len(problems)} problems")
    print(f"Model: {model}")

    results = []
    for prob in problems:
        print(f"\n--- {prob['id']} (L{prob['level']}, {prob['type']}) ---")

        # Pass 1: Generate solution
        solve_prompt = SOLVE_PROMPT.format(problem=prob["problem"])
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": solve_prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
        except Exception as e:
            print(f"  API ERROR: {e}")
            print(f"  Saving partial results ({len(results)} problems completed)...")
            break
        solution = resp.choices[0].message.content
        print(f"  Solution generated ({len(solution)} chars)")

        # Extract answer
        idx = solution.rfind("\\boxed{")
        predicted = ""
        if idx != -1:
            start = idx + 7
            depth = 1
            i = start
            while i < len(solution) and depth > 0:
                if solution[i] == "{": depth += 1
                elif solution[i] == "}": depth -= 1
                i += 1
            predicted = solution[start:i-1]
        correct = predicted.strip() == prob["answer"].strip()
        print(f"  Predicted: {predicted}, Gold: {prob['answer']}, Correct: {correct}")

        # Pass 2: Generate verification
        verify_prompt = VERIFY_PROMPT.format(solution=solution)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": verify_prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
        except Exception as e:
            print(f"  API ERROR on verification: {e}")
            results.append({
                "id": prob["id"], "level": prob["level"], "type": prob["type"],
                "predicted": predicted, "correct": correct,
                "script_valid": False, "has_assertions": False, "verdict": "API_ERROR",
            })
            continue
        verify_resp = resp.choices[0].message.content

        # Extract script
        script = ""
        if "```" in verify_resp:
            pattern = r"```python\s*\n(.*?)```"
            matches = re.findall(pattern, verify_resp, re.DOTALL)
            if matches:
                script = max(matches, key=len).strip()
            else:
                pattern = r"```\s*\n(.*?)```"
                matches = re.findall(pattern, verify_resp, re.DOTALL)
                if matches:
                    script = max(matches, key=len).strip()
        else:
            script = verify_resp.strip()

        # Check script validity
        valid = False
        has_assertions = False
        if script:
            try:
                compile(script, "<verify>", "exec")
                valid = True
                has_assertions = "assert " in script
            except SyntaxError:
                pass

        # Execute
        verdict = "NO_SCRIPT"
        if valid:
            import subprocess
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(script)
                    f.flush()
                    tmp_path = f.name
                result = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    verdict = "ALL_PASS"
                elif "AssertionError" in result.stderr:
                    verdict = "FAIL_STEP"
                else:
                    verdict = "RUNTIME_ERROR"
            except subprocess.TimeoutExpired:
                verdict = "TIMEOUT"
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        elif script:
            verdict = "SYNTAX_ERROR"

        print(f"  Script valid: {valid}, Assertions: {has_assertions}, Verdict: {verdict}")

        results.append({
            "id": prob["id"],
            "level": prob["level"],
            "type": prob["type"],
            "predicted": predicted,
            "correct": correct,
            "script_valid": valid,
            "has_assertions": has_assertions,
            "verdict": verdict,
        })

        time.sleep(0.5)  # Rate limiting

    # Summary
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_valid = sum(1 for r in results if r["script_valid"])
    n_all_pass = sum(1 for r in results if r["verdict"] == "ALL_PASS")
    n_has_verdict = sum(1 for r in results if r["verdict"] in ("ALL_PASS", "FAIL_STEP"))

    print(f"\n{'='*50}")
    print(f"AIME SANITY CHECK RESULTS (n={n})")
    print(f"{'='*50}")
    print(f"Accuracy: {n_correct}/{n} ({n_correct/n*100:.1f}%)")
    print(f"Script validity (SVR): {n_valid}/{n} ({n_valid/n*100:.1f}%)")
    print(f"Problem coverage (PC): {n_has_verdict}/{n} ({n_has_verdict/n*100:.1f}%)")

    if n_all_pass > 0:
        n_wrong_pass = sum(1 for r in results if r["verdict"] == "ALL_PASS" and not r["correct"])
        print(f"FPVR: {n_wrong_pass}/{n_all_pass} ({n_wrong_pass/n_all_pass*100:.1f}%)")

    out_path = RESULTS_DIR / "exp14_aime_sanity.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "n": n}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

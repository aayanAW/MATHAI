"""Tier 3: Cell A vs Cell B empirical measurement --- internally consistent.

Qwen/Qwen2.5-Math-7B-Instruct is non-serverless on Together API, so we use
Qwen/Qwen2.5-7B-Instruct-Turbo as the single solver for BOTH Pass 1
(solution generation) and Pass 2 (verifier script generation). The original
exp5 Cell A measurement used Qwen-Math-7B; the present exp41 measurement is
an internally consistent 2-cell comparison on a different solver (Turbo).
We run BOTH Cell A (deterministic VERIFY_PROMPT) and Cell B (randomized
VERIFY_PROMPT) on the same 120 fresh Pass 1 solutions, so we can directly
compare solution-grounded deterministic vs solution-grounded randomized.

Hypothesis (from the paper's 2x2 claim): Cell A and Cell B have comparable
FAR because randomizing the Pass 2 test inputs does not break correlated-
error collapse when both test inputs and test oracle come from Pass 1.

Protocol:
  1. Sample 120 MATH-500 problems stratified by level.
  2. Generate Pass 1 NL solution with Qwen-7B-Instruct-Turbo.
  3. Run Pass 2 deterministic (Cell A) --- standard ExeVer VERIFY_PROMPT.
  4. Run Pass 2 randomized (Cell B) --- modified prompt, multi-input checks.
  5. Execute both scripts, record ALL_PASS vs. wrong-answer.
  6. Report FAR for each cell with Clopper-Pearson 95% CIs.

Budget: ~$0.15 (120 problems * 3 Turbo API calls each).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.exever.executor import execute_verification_script  # type: ignore
from src.eval.answer_check import answers_equivalent  # type: ignore

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp41_cell_b_empirical.json"
SOLVER = "Qwen/Qwen2.5-7B-Instruct-Turbo"  # serverless Together replacement for Math-7B

API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY",
)
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=API_KEY,
    timeout=60.0,
    max_retries=1,
)


SOLVE_PROMPT = """Solve the following math problem. Think step by step and format your answer as \\boxed{{answer}} at the end.

Problem: {problem}

Step-by-step solution:"""


DETERMINISTIC_VERIFY_PROMPT = """Below is a step-by-step math solution. Write a CUMULATIVE Python/SymPy script that verifies each step via transition assertions.

RULES:
1. CUMULATIVE --- each step's code builds on prior variables.
2. Start with: from sympy import *
3. For EACH step, write: # === STEP k ===
4. In each section, write assert statements that check the step's claim against the prior state. Use transition assertions (verify the claimed result, do not re-derive).
5. Label: assert condition, "FAIL:Step k: description"
6. Use expand() for algebraic comparison (NOT simplify(), which can hang).
7. Generate EXACTLY one code section per solution step.
8. At the end: print("ANSWER:", final_answer)

Solution to verify:
{solution}

Write the complete cumulative Python/SymPy verification script:"""


RANDOMIZED_VERIFY_PROMPT = """Below is a step-by-step math solution. Write a CUMULATIVE Python/SymPy script that verifies each step by evaluating the claim at MULTIPLE RANDOM numeric inputs, not just the claimed value.

RULES:
1. CUMULATIVE --- each step's code builds on prior variables.
2. Start with:
   from sympy import *
   import random
   random.seed(42)
3. For EACH step, write: # === STEP k ===
4. In each section:
   - Assign key symbolic variables.
   - Extract the step's claimed identity or equation.
   - Evaluate the claim at 5 randomly-chosen numeric substitutions of the free variables.
   - Write: assert all_random_checks_pass, "FAIL:Step k: description"
5. Example:
     # === STEP 2 ===
     lhs = x**2 - 5*x + 6
     rhs = (x - 2)*(x - 3)
     passed = all(lhs.subs(x, v).equals(rhs.subs(x, v)) for v in [random.randint(-10, 10) for _ in range(5)])
     assert passed, "FAIL:Step 2: factoring does not hold at random points"
6. Use expand() for algebraic simplification (NOT simplify(), which can hang).
7. Generate EXACTLY one code section per solution step.
8. At the end: print("ANSWER:", final_answer)

Solution to verify:
{solution}

Write the complete cumulative Python/SymPy verification script:"""


def sample_stratified(n_total: int = 120, seed: int = 42) -> list[dict]:
    """Draw n_total problems stratified by level from MATH-500."""
    with open(RESULTS / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    by_level: dict[int, list] = {}
    for p in math_all:
        try:
            lvl_int = int(str(p.get("level", 3)).replace("Level ", ""))
        except Exception:
            lvl_int = 3
        by_level.setdefault(lvl_int, []).append(p)
    rng = random.Random(seed)
    picked: list[dict] = []
    per_level = max(1, n_total // max(1, len(by_level)))
    for lvl in sorted(by_level):
        rng.shuffle(by_level[lvl])
        picked.extend(by_level[lvl][:per_level])
    rng.shuffle(picked)
    return picked[:n_total]


def _equiv(a: str, b: str) -> bool:
    r = answers_equivalent(a, b)
    return bool(r[0]) if isinstance(r, tuple) else bool(r)


def _extract_boxed(text: str) -> str:
    i = text.rfind("\\boxed{")
    if i < 0:
        return ""
    start = i + len("\\boxed{")
    depth = 1
    j = start
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j].strip()
        j += 1
    return text[start:].strip()


def solve(problem: str, max_retries: int = 2) -> str:
    prompt = SOLVE_PROMPT.format(problem=problem)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=SOLVER,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2.0)
                continue
            return ""
    return ""


def extract_python_code(raw: str) -> str:
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _verify(solution: str, prompt_template: str, max_retries: int = 2) -> str:
    prompt = prompt_template.format(solution=solution)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=SOLVER,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            return extract_python_code(text)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2.0)
                continue
            return ""
    return ""


def deterministic_verify(solution: str) -> str:
    return _verify(solution, DETERMINISTIC_VERIFY_PROMPT)


def randomized_verify(solution: str) -> str:
    return _verify(solution, RANDOMIZED_VERIFY_PROMPT)


def main():
    # Reset: drop any stale rows
    if OUT.exists():
        try:
            stale = json.load(open(OUT))
            if any(r.get("status") == "verify_api_error" for r in stale.get("rows", [])):
                print("Dropping stale rows (Qwen-Math-7B-Instruct non-serverless errors)")
                os.remove(OUT)
        except Exception:
            pass

    probs = sample_stratified(n_total=120, seed=42)
    print(f"Sampled {len(probs)} MATH-500 problems stratified by level")

    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {"solver": SOLVER, "rows": []}

    done_ids = {r["id"] for r in out["rows"]}
    todo = [p for p in probs if p["id"] not in done_ids]
    print(f"Already done: {len(done_ids)}  To do: {len(todo)}")

    for i, p in enumerate(todo):
        pid = p["id"]
        problem = p.get("problem", "")
        gold = str(p.get("answer", ""))
        if not problem:
            continue
        t0 = time.time()

        # --- Pass 1: generate NL solution ---
        sol_text = solve(problem)
        predicted = _extract_boxed(sol_text)
        answer_correct = _equiv(predicted, gold)

        if not sol_text:
            out["rows"].append({
                "id": pid,
                "status": "pass1_error",
                "gold": gold,
                "elapsed": time.time() - t0,
            })
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2, default=str)
            continue

        # --- Pass 2 deterministic (Cell A) ---
        script_a = deterministic_verify(sol_text)
        if script_a:
            try:
                res_a = execute_verification_script(script_a, timeout=15.0)
                verdict_a = res_a.verdict
            except Exception as e:
                verdict_a = f"exec_error:{type(e).__name__}"
        else:
            verdict_a = "verify_api_error"
        all_pass_a = (verdict_a == "ALL_PASS")

        # --- Pass 2 randomized (Cell B) ---
        script_b = randomized_verify(sol_text)
        if script_b:
            try:
                res_b = execute_verification_script(script_b, timeout=15.0)
                verdict_b = res_b.verdict
            except Exception as e:
                verdict_b = f"exec_error:{type(e).__name__}"
        else:
            verdict_b = "verify_api_error"
        all_pass_b = (verdict_b == "ALL_PASS")

        row = {
            "id": pid,
            "gold": gold,
            "predicted": predicted,
            "answer_correct": answer_correct,
            "cell_a": {
                "verdict": verdict_a,
                "all_pass": all_pass_a,
                "script_chars": len(script_a) if script_a else 0,
            },
            "cell_b": {
                "verdict": verdict_b,
                "all_pass": all_pass_b,
                "script_chars": len(script_b) if script_b else 0,
            },
            "elapsed": time.time() - t0,
        }
        out["rows"].append(row)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)

        print(
            f"  [{i+1}/{len(todo)}] {pid}: correct={answer_correct}  "
            f"A={verdict_a}  B={verdict_b}  ({row['elapsed']:.1f}s)",
            flush=True,
        )

    # Summary FARs for both cells
    rows = out["rows"]
    total = len(rows)
    ap_a = [r for r in rows if r.get("cell_a", {}).get("all_pass")]
    ap_b = [r for r in rows if r.get("cell_b", {}).get("all_pass")]
    wrong_a = [r for r in ap_a if not r.get("answer_correct")]
    wrong_b = [r for r in ap_b if not r.get("answer_correct")]

    from scipy.stats import binomtest, fisher_exact

    def _summary(name, ap_rows, wrong_rows, total):
        n_ap = len(ap_rows)
        n_wrong = len(wrong_rows)
        if not n_ap:
            return {"cell": name, "all_pass": 0, "far": None}
        far = n_wrong / n_ap
        ci = binomtest(n_wrong, n_ap).proportion_ci(0.95, "exact")
        return {
            "cell": name,
            "n_total": total,
            "all_pass": n_ap,
            "all_pass_fraction": n_ap / total,
            "n_wrong": n_wrong,
            "far": far,
            "ci_lo": float(ci.low),
            "ci_hi": float(ci.high),
        }

    summ_a = _summary("A (deterministic)", ap_a, wrong_a, total)
    summ_b = _summary("B (randomized)", ap_b, wrong_b, total)

    print("\n" + "=" * 60)
    print("Cell A vs Cell B empirical (both solution-grounded)")
    print("=" * 60)
    print(f"n_total = {total}")
    for s in [summ_a, summ_b]:
        if s.get("far") is not None:
            print(f"  {s['cell']}: ALL_PASS={s['all_pass']}, FAR={s['n_wrong']}/{s['all_pass']}"
                  f" = {s['far']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
        else:
            print(f"  {s['cell']}: tier empty")

    # Paired 2x2: A-accepts-wrong vs B-accepts-wrong on the same problems
    a_wrong_ids = {r["id"] for r in wrong_a}
    b_wrong_ids = {r["id"] for r in wrong_b}
    both = len(a_wrong_ids & b_wrong_ids)
    a_only = len(a_wrong_ids - b_wrong_ids)
    b_only = len(b_wrong_ids - a_wrong_ids)
    neither = total - both - a_only - b_only
    print(f"\nPaired 2x2 (problem-level FAR overlap):")
    print(f"                A wrong   A right")
    print(f"  B wrong       {both:6d}    {b_only:6d}")
    print(f"  B right       {a_only:6d}    {neither:6d}")
    try:
        _, p_fisher = fisher_exact([[both, b_only], [a_only, neither]])
        print(f"  Fisher exact p = {p_fisher:.4f}")
    except Exception:
        p_fisher = None

    out["summary"] = {
        "cell_a": summ_a,
        "cell_b": summ_b,
        "paired": {
            "both_wrong": both,
            "a_only_wrong": a_only,
            "b_only_wrong": b_only,
            "neither_wrong": neither,
            "fisher_p": p_fisher,
        },
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

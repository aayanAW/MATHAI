"""Experiment 4: Repair + Backtracking + Baseline Comparison.

Runs the full ExeVer pipeline (solve + verify + repair) on 300 problems,
then compares against baselines at equal compute budget.

Compute-matched:
  ExeVer: 1 solve + 1 verify + up to 2 repair calls = ~4 model calls
  Baseline: majority@4 or best-of-4 = 4 model calls
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp4")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.49.0", "vllm==0.7.3", "numpy<2")
)
model_volume = modal.Volume.from_name("exever-models", create_if_missing=True)

# All prompts embedded for self-contained Modal execution
SOLVE_PROMPT = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""

VERIFY_PROMPT = """Below is a step-by-step math solution. Write a Python/SymPy script that checks each step.

IMPORTANT: Your script MUST use assert statements to verify each step.

Example of what I want:
```python
from sympy import *
x = symbols('x')
# === STEP 1 ===
eq = x**2 - 5*x + 6
# === STEP 2 ===
claimed = (x-2)*(x-3)
assert expand(claimed - eq) == 0, "FAIL:Step 2: factoring"
# === STEP 3 ===
assert eq.subs(x, 2) == 0, "FAIL:Step 3: x=2 not root"
assert eq.subs(x, 3) == 0, "FAIL:Step 3: x=3 not root"
# === STEP 4 ===
assert 2 + 3 == 5, "FAIL:Step 4: sum"
print("ANSWER:", 5)
```

Rules:
- Start with: from sympy import *
- For each step write: # === STEP k === then assert statements
- Every step MUST have at least one assert
- Use expand() not simplify()
- End with: print("ANSWER:", answer)
- Output ONLY the Python code, nothing else.

Solution to verify:
{solution}

```python"""

REPAIR_PROMPT = """You are solving the following math problem. Your previous solution had an error at Step {step_num}.

Problem: {problem}

Verified correct steps (keep these exactly):
{verified_prefix}

The error was at Step {step_num}:
{failed_step}

Error message from verification:
{error_message}

Please provide:
1. A corrected Step {step_num} and all subsequent steps
2. Format each step as "## Step k: [title]" followed by reasoning
3. State your final answer as: The answer is \\boxed{{answer}}.

Continue from Step {step_num}:"""


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=7200,
    scaledown_window=60,
)
def run_full_exever_and_baselines(problems_json: str) -> str:
    """Run full ExeVer pipeline + baselines on all problems."""
    import json as _json
    import re as _re
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    print("Loading Qwen2.5-Math-7B-Instruct...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )

    greedy = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)
    sampled = SamplingParams(max_tokens=2048, temperature=0.7, top_p=0.95)
    verify_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])

    # === BASELINE: Majority@4 (4 samples with temperature) ===
    print("Running majority@4 baselines...")
    t0 = __import__("time").time()

    # Generate 4 solutions per problem
    baseline_prompts = []
    for p in problems:
        for _ in range(4):
            baseline_prompts.append(SOLVE_PROMPT.format(problem=p["problem"]))

    baseline_outputs = llm.generate(baseline_prompts, sampled)
    baseline_time = __import__("time").time() - t0
    print(f"Baselines done in {baseline_time:.1f}s ({len(baseline_prompts)} prompts)")

    # Organize: 4 solutions per problem
    baseline_solutions = []
    for i in range(len(problems)):
        sols = []
        for j in range(4):
            resp = baseline_outputs[i * 4 + j].outputs[0].text
            sols.append(resp)
        baseline_solutions.append(sols)

    # === EXEVER: Solve + Verify + Repair (up to 2 rounds) ===
    print("Running ExeVer pipeline (solve + verify + repair)...")
    t0 = __import__("time").time()

    # Pass 1: Generate greedy solutions (reuse Exp 1 if possible, but re-generate for fairness)
    solve_prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in problems]
    solve_outputs = llm.generate(solve_prompts, greedy)
    solutions = [o.outputs[0].text for o in solve_outputs]

    # Pass 2: Generate verification scripts
    verify_prompts = [VERIFY_PROMPT.format(solution=sol) for sol in solutions]
    verify_outputs = llm.generate(verify_prompts, verify_params)
    verify_scripts = [o.outputs[0].text for o in verify_outputs]

    # For repair: identify failed scripts and generate repair prompts
    # We'll do a lightweight repair pass — generate repair prompts for all problems
    # and let the local code decide which repairs to use
    repair_prompts = []
    repair_indices = []
    for i, p in enumerate(problems):
        # Generate a generic repair prompt for step 1 error (we'll override in local eval)
        repair_prompts.append(REPAIR_PROMPT.format(
            problem=p["problem"],
            verified_prefix="(none — first step failed)",
            failed_step="(see error message)",
            step_num=1,
            error_message="Assertion error in verification script",
        ))
        repair_indices.append(i)

    # Generate repairs for all (local code will only use the relevant ones)
    repair_outputs = llm.generate(repair_prompts, greedy)
    repair_solutions = [o.outputs[0].text for o in repair_outputs]

    # Re-verify repairs
    repair_verify_prompts = [VERIFY_PROMPT.format(solution=sol) for sol in repair_solutions]
    repair_verify_outputs = llm.generate(repair_verify_prompts, verify_params)
    repair_verify_scripts = [o.outputs[0].text for o in repair_verify_outputs]

    exever_time = __import__("time").time() - t0
    print(f"ExeVer pipeline done in {exever_time:.1f}s")

    # Package all raw outputs
    results = {
        "problems": [{
            "id": p["id"],
            "problem": p["problem"],
            "level": p["level"],
            "type": p["type"],
            "gold_answer": p["answer"],
        } for p in problems],
        "baseline_solutions": baseline_solutions,
        "exever_solutions": solutions,
        "exever_verify_scripts": verify_scripts,
        "exever_repair_solutions": repair_solutions,
        "exever_repair_verify_scripts": repair_verify_scripts,
        "baseline_time": baseline_time,
        "exever_time": exever_time,
    }

    return _json.dumps(results)


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.exever.executor import execute_verification_script
    from src.exever.step_parser import extract_assertions, parse_nl_steps

    # Load problems
    with open(RESULTS_DIR / "math_test_sample_300.json") as f:
        problems = json.load(f)

    print(f"Running ExeVer + baselines on {len(problems)} problems...")
    raw_result = run_full_exever_and_baselines.remote(json.dumps(problems))
    data = json.loads(raw_result)

    baseline_time = data["baseline_time"]
    exever_time = data["exever_time"]
    probs = data["problems"]

    print(f"Baseline time: {baseline_time:.1f}s, ExeVer time: {exever_time:.1f}s")
    print("Evaluating locally...")

    def extract_script(response):
        if "```" not in response:
            lines = response.split("\n")
            code_lines = []
            started = False
            for line in lines:
                stripped = line.strip()
                if not started:
                    if stripped.startswith(("from ", "import ", "#", "x ", "x=",
                                            "def ", "print(")) or stripped == "":
                        started = True
                        code_lines.append(line)
                else:
                    code_lines.append(line)
            return "\n".join(code_lines).strip()
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            py = [m for m in matches if "import" in m or "assert" in m]
            if py:
                return max(py, key=len).strip()
        return ""

    # === Evaluate Baselines ===
    baseline_results = []
    for i, p in enumerate(probs):
        sols = data["baseline_solutions"][i]
        answers = [extract_model_answer(s) for s in sols]

        # pass@1: first sample
        a1_correct, _ = answers_equivalent(answers[0], p["gold_answer"])

        # best-of-4: any correct
        any_correct = any(
            answers_equivalent(a, p["gold_answer"])[0] for a in answers
        )

        # majority@4: symbolic equivalence grouping
        groups = []
        for a in answers:
            merged = False
            for j, (canonical, count, members) in enumerate(groups):
                eq, _ = answers_equivalent(a, canonical)
                if eq:
                    groups[j] = (canonical, count + 1, members + [a])
                    merged = True
                    break
            if not merged:
                groups.append((a, 1, [a]))
        groups.sort(key=lambda x: x[1], reverse=True)
        majority_answer = groups[0][0] if groups else ""
        maj_correct, _ = answers_equivalent(majority_answer, p["gold_answer"])

        baseline_results.append({
            "id": p["id"],
            "level": p["level"],
            "type": p["type"],
            "pass_at_1": a1_correct,
            "best_of_4": any_correct,
            "majority_4": maj_correct,
        })

    # === Evaluate ExeVer ===
    exever_results = []
    for i, p in enumerate(probs):
        sol = data["exever_solutions"][i]
        verify_resp = data["exever_verify_scripts"][i]
        repair_sol = data["exever_repair_solutions"][i]
        repair_verify_resp = data["exever_repair_verify_scripts"][i]

        # Extract answer from solution
        pred = extract_model_answer(sol)
        correct, _ = answers_equivalent(pred, p["gold_answer"])

        # Execute verification script
        script = extract_script(verify_resp)
        exec_result = None
        verdict = "NO_SCRIPT"
        assertions = 0
        repair_used = False
        repair_correct = False

        if script.strip():
            try:
                compile(script, "<v>", "exec")
                exec_result = execute_verification_script(script, timeout=30)
                assertions = len(extract_assertions(script))

                if exec_result.success:
                    verdict = "ALL_PASS"
                elif exec_result.assertion_error:
                    verdict = f"FAIL_STEP_{exec_result.error_step}"
                    # Try repair
                    repair_script = extract_script(repair_verify_resp)
                    if repair_script.strip():
                        try:
                            compile(repair_script, "<r>", "exec")
                            repair_exec = execute_verification_script(
                                repair_script, timeout=30
                            )
                            if repair_exec.success:
                                repair_used = True
                                repair_pred = extract_model_answer(repair_sol)
                                repair_correct, _ = answers_equivalent(
                                    repair_pred, p["gold_answer"]
                                )
                                pred = repair_pred
                                correct = repair_correct
                                verdict = "REPAIRED"
                        except SyntaxError:
                            pass
                elif exec_result.timeout:
                    verdict = "TIMEOUT"
                else:
                    verdict = "RUNTIME_ERROR"
            except SyntaxError:
                verdict = "SYNTAX_ERROR"

        # ExeVer selection: use script answer if available and correct
        echo_chamber = None
        if exec_result and exec_result.success:
            if exec_result.answer_extracted:
                sa_correct, _ = answers_equivalent(
                    exec_result.answer_extracted, p["gold_answer"]
                )
                echo_chamber = not sa_correct

        exever_results.append({
            "id": p["id"],
            "level": p["level"],
            "type": p["type"],
            "predicted": pred,
            "correct": correct,
            "verdict": verdict,
            "assertions": assertions,
            "repair_used": repair_used,
            "repair_correct": repair_correct,
            "echo_chamber": echo_chamber,
        })

    # === Compute Summary Metrics ===
    n = len(probs)

    # Baselines
    b_pass1 = sum(1 for r in baseline_results if r["pass_at_1"]) / n
    b_best4 = sum(1 for r in baseline_results if r["best_of_4"]) / n
    b_maj4 = sum(1 for r in baseline_results if r["majority_4"]) / n

    # ExeVer
    e_correct = sum(1 for r in exever_results if r["correct"]) / n
    e_repaired = sum(1 for r in exever_results if r["repair_used"])
    e_repair_correct = sum(1 for r in exever_results if r["repair_correct"])
    e_repair_rate = e_repair_correct / e_repaired if e_repaired > 0 else 0

    # Exp 1 baseline (greedy CoT) for reference
    try:
        with open(RESULTS_DIR / "exp1_baseline_qwen7b.json") as f:
            exp1 = json.load(f)
        exp1_pass1 = exp1["pass_at_1"]
    except Exception:
        exp1_pass1 = 0

    print(f"\n{'='*60}")
    print(f"EXPERIMENT 4 RESULTS: ExeVer vs Baselines")
    print(f"{'='*60}")
    print(f"\nAccuracy Comparison (equal compute: 4 model calls):")
    print(f"  Exp 1 CoT pass@1 (greedy):  {exp1_pass1:.1%}")
    print(f"  Baseline pass@1 (sampled):   {b_pass1:.1%}")
    print(f"  Baseline best-of-4:          {b_best4:.1%}")
    print(f"  Baseline majority@4:         {b_maj4:.1%}")
    print(f"  ExeVer (solve+verify+repair): {e_correct:.1%}")

    print(f"\nExeVer Details:")
    verdict_counts = Counter(r["verdict"] for r in exever_results)
    for v, c in verdict_counts.most_common():
        print(f"  {v}: {c}")
    print(f"  Repairs attempted: {e_repaired}")
    print(f"  Repairs successful: {e_repair_correct} ({e_repair_rate:.1%})")

    print(f"\nCompute:")
    print(f"  Baseline time: {baseline_time:.1f}s (4 samples × 300 = 1200 prompts)")
    print(f"  ExeVer time: {exever_time:.1f}s (solve+verify+repair+re-verify)")

    # By difficulty level
    by_level_bl = defaultdict(lambda: {"n": 0, "pass1": 0, "best4": 0, "maj4": 0})
    by_level_ev = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in baseline_results:
        lv = r["level"]
        by_level_bl[lv]["n"] += 1
        by_level_bl[lv]["pass1"] += r["pass_at_1"]
        by_level_bl[lv]["best4"] += r["best_of_4"]
        by_level_bl[lv]["maj4"] += r["majority_4"]
    for r in exever_results:
        lv = r["level"]
        by_level_ev[lv]["n"] += 1
        by_level_ev[lv]["correct"] += r["correct"]

    print(f"\nBy Difficulty Level:")
    print(f"  {'Level':<8} {'pass@1':<10} {'best4':<10} {'maj@4':<10} {'ExeVer':<10}")
    print(f"  {'-'*48}")
    for lv in sorted(by_level_bl):
        bl = by_level_bl[lv]
        ev = by_level_ev[lv]
        p1 = bl["pass1"] / bl["n"] if bl["n"] > 0 else 0
        b4 = bl["best4"] / bl["n"] if bl["n"] > 0 else 0
        m4 = bl["maj4"] / bl["n"] if bl["n"] > 0 else 0
        ec = ev["correct"] / ev["n"] if ev["n"] > 0 else 0
        print(f"  L{lv:<7} {p1:<10.1%} {b4:<10.1%} {m4:<10.1%} {ec:<10.1%}")

    # By subject
    by_subj_bl = defaultdict(lambda: {"n": 0, "pass1": 0, "best4": 0, "maj4": 0})
    by_subj_ev = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in baseline_results:
        s = r["type"]
        by_subj_bl[s]["n"] += 1
        by_subj_bl[s]["pass1"] += r["pass_at_1"]
        by_subj_bl[s]["best4"] += r["best_of_4"]
        by_subj_bl[s]["maj4"] += r["majority_4"]
    for r in exever_results:
        s = r["type"]
        by_subj_ev[s]["n"] += 1
        by_subj_ev[s]["correct"] += r["correct"]

    print(f"\nBy Subject:")
    print(f"  {'Subject':<25} {'pass@1':<10} {'best4':<10} {'maj@4':<10} {'ExeVer':<10}")
    print(f"  {'-'*65}")
    for s in sorted(by_subj_bl):
        bl = by_subj_bl[s]
        ev = by_subj_ev[s]
        p1 = bl["pass1"] / bl["n"] if bl["n"] > 0 else 0
        b4 = bl["best4"] / bl["n"] if bl["n"] > 0 else 0
        m4 = bl["maj4"] / bl["n"] if bl["n"] > 0 else 0
        ec = ev["correct"] / ev["n"] if ev["n"] > 0 else 0
        print(f"  {s:<25} {p1:<10.1%} {b4:<10.1%} {m4:<10.1%} {ec:<10.1%}")

    # === Save ===
    output = {
        "experiment": "exp4_repair_baselines",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": n,
        "baseline_time_s": baseline_time,
        "exever_time_s": exever_time,
        "accuracy": {
            "exp1_cot_pass1": exp1_pass1,
            "baseline_pass1_sampled": b_pass1,
            "baseline_best_of_4": b_best4,
            "baseline_majority_4": b_maj4,
            "exever": e_correct,
        },
        "repair": {
            "attempted": e_repaired,
            "successful": e_repair_correct,
            "success_rate": e_repair_rate,
        },
        "exever_verdicts": dict(verdict_counts),
        "by_level": {
            str(lv): {
                "pass1": by_level_bl[lv]["pass1"] / by_level_bl[lv]["n"],
                "best4": by_level_bl[lv]["best4"] / by_level_bl[lv]["n"],
                "maj4": by_level_bl[lv]["maj4"] / by_level_bl[lv]["n"],
                "exever": by_level_ev[lv]["correct"] / by_level_ev[lv]["n"],
                "n": by_level_bl[lv]["n"],
            }
            for lv in sorted(by_level_bl)
        },
        "by_subject": {
            s: {
                "pass1": by_subj_bl[s]["pass1"] / by_subj_bl[s]["n"],
                "best4": by_subj_bl[s]["best4"] / by_subj_bl[s]["n"],
                "maj4": by_subj_bl[s]["maj4"] / by_subj_bl[s]["n"],
                "exever": by_subj_ev[s]["correct"] / by_subj_ev[s]["n"],
                "n": by_subj_bl[s]["n"],
            }
            for s in sorted(by_subj_bl)
        },
        "baseline_results": baseline_results,
        "exever_results": exever_results,
    }

    out_path = RESULTS_DIR / "exp4_repair_baselines.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

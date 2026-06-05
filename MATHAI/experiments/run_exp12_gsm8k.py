"""Experiment 12: GSM8K Coverage Test.

Runs the ExeVer pipeline on GSM8K (first 500 problems) to measure
verification coverage on simpler arithmetic problems. GSM8K is easier
than MATH, so we expect HIGHER verification coverage here.

Key hypothesis: ExeVer coverage should be significantly higher on GSM8K
because the problems are straightforward arithmetic — easier to generate
correct verification scripts for.

Metrics:
- Greedy CoT accuracy
- ExeVer accuracy (with repair)
- Verification coverage (% steps with PASS/FAIL vs ERROR)
- Echo chamber rate
- Comparison to MATH-500 results from exp5
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp12-gsm8k")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.49.0", "vllm==0.7.3", "numpy<2", "datasets")
)
model_volume = modal.Volume.from_name("exever-models", create_if_missing=True)

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


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numeric answer from GSM8K's '#### NUMBER' format.

    GSM8K answers end with '#### <number>', e.g. '#### 42' or '#### 1,200'.
    We extract just the number, stripping commas.
    """
    match = re.search(r"####\s*(.+?)$", answer_text, re.MULTILINE)
    if match:
        ans = match.group(1).strip()
        # Remove commas from numbers like 1,200
        ans = ans.replace(",", "")
        return ans
    # Fallback: try to get the last number in the text
    nums = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if nums:
        return nums[-1].replace(",", "")
    return answer_text.strip()


def extract_script(response: str) -> str:
    """Extract Python script from model response."""
    if "```" not in response:
        lines = response.split("\n")
        code_lines = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                if stripped.startswith(("from ", "import ", "#", "x ", "x=", "eq",
                                        "def ", "class ", "print(")) or stripped == "":
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
        py_blocks = [m for m in matches if any(
            kw in m for kw in ["import", "assert", "def ", "print(", "from "]
        )]
        if py_blocks:
            return max(py_blocks, key=len).strip()
    return ""


def _is_trivial_assert(a: str) -> bool:
    body = a.replace("assert ", "", 1).strip()
    if body.startswith("True"):
        return True
    if "==" in body:
        parts = body.split("==", 1)
        if parts[0].strip() == parts[1].strip().split(",")[0].strip():
            return True
    if body.startswith("isinstance("):
        return True
    return False


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=7200,
    scaledown_window=120,
)
def run_gsm8k_inference(problems_json: str) -> str:
    """Generate greedy solutions + verification scripts + repair on GPU.

    Pipeline:
    1. Greedy CoT solutions for all problems
    2. Verification scripts for each greedy solution
    3. (Repair is handled locally after execution results are known)
    """
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    print(f"Loading Qwen2.5-Math-7B-Instruct on H100...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    )
    from modal import Volume
    Volume.from_name("exever-models").commit()

    results = {}

    # === Phase 1: Greedy CoT solutions ===
    greedy_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
    )
    greedy_prompts = [SOLVE_PROMPT.format(problem=p["question"]) for p in problems]

    print(f"\n[Phase 1] Generating {len(greedy_prompts)} greedy CoT solutions...")
    t0 = __import__("time").time()
    greedy_outputs = llm.generate(greedy_prompts, greedy_params)
    t1 = __import__("time").time()
    print(f"  Done in {t1-t0:.1f}s")

    results["greedy_solutions"] = [
        out.outputs[0].text for out in greedy_outputs
    ]

    # === Phase 2: Verification scripts ===
    verify_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        stop=["```"],
    )
    verify_prompts = [
        VERIFY_PROMPT.format(solution=sol)
        for sol in results["greedy_solutions"]
    ]

    print(f"\n[Phase 2] Generating {len(verify_prompts)} verification scripts...")
    t0 = __import__("time").time()
    verify_outputs = llm.generate(verify_prompts, verify_params)
    t1 = __import__("time").time()
    print(f"  Done in {t1-t0:.1f}s")

    results["verify_responses"] = [
        out.outputs[0].text for out in verify_outputs
    ]

    print(f"\nAll GPU inference complete. Returning results.")
    return _json.dumps(results)


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=3600,
    scaledown_window=120,
)
def run_repair_inference(repair_inputs_json: str) -> str:
    """Run repair inference for problems that failed verification."""
    import json as _json
    from vllm import LLM, SamplingParams

    repair_inputs = _json.loads(repair_inputs_json)
    if not repair_inputs:
        return _json.dumps({"repairs": [], "reverify": []})

    print(f"Loading model for repair ({len(repair_inputs)} problems)...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
    )

    # Generate repairs
    repair_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0)
    repair_prompts = [inp["repair_prompt"] for inp in repair_inputs]

    print(f"Generating {len(repair_prompts)} repair solutions...")
    repair_outputs = llm.generate(repair_prompts, repair_params)
    repairs = [out.outputs[0].text for out in repair_outputs]

    # Generate re-verification scripts for repaired solutions
    verify_params = SamplingParams(max_tokens=2048, temperature=0.0, top_p=1.0, stop=["```"])
    verify_prompts = [
        VERIFY_PROMPT.format(solution=rep) for rep in repairs
    ]

    print(f"Generating {len(verify_prompts)} re-verification scripts...")
    verify_outputs = llm.generate(verify_prompts, verify_params)
    reverify = [out.outputs[0].text for out in verify_outputs]

    return _json.dumps({
        "repairs": repairs,
        "reverify": reverify,
    })


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.exever.executor import execute_verification_script
    from src.exever.step_parser import extract_assertions, parse_nl_steps

    # === Load GSM8K from HuggingFace ===
    from datasets import load_dataset

    print("Loading GSM8K test set from HuggingFace...")
    ds = load_dataset("gsm8k", "main", split="test")
    print(f"Full GSM8K test set: {len(ds)} problems")

    # Take first 500 for computational reasons
    n_problems = 500
    problems = []
    for i in range(n_problems):
        row = ds[i]
        gold_answer = extract_gsm8k_answer(row["answer"])
        problems.append({
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "full_answer": row["answer"],
            "answer": gold_answer,
        })
    print(f"Using first {n_problems} problems (gold answers extracted)")

    # === Run GPU inference ===
    print("\n" + "=" * 60)
    print("PHASE 1: GPU Inference (greedy CoT + verification scripts)")
    print("=" * 60)
    raw = run_gsm8k_inference.remote(json.dumps(problems))
    gpu_results = json.loads(raw)

    greedy_solutions = gpu_results["greedy_solutions"]
    verify_responses = gpu_results["verify_responses"]

    # === Evaluate Greedy CoT Baseline ===
    print("\n" + "=" * 60)
    print("PHASE 2: Evaluate Greedy CoT Baseline")
    print("=" * 60)

    greedy_correct = 0
    greedy_results = []
    for i, (prob, sol) in enumerate(zip(problems, greedy_solutions)):
        pred = extract_model_answer(sol)
        # For GSM8K, answers are plain numbers. Use answers_equivalent
        # which handles numeric comparison via SymPy.
        correct, method = answers_equivalent(pred, prob["answer"])
        greedy_correct += int(correct)
        greedy_results.append({
            "id": prob["id"],
            "gold_answer": prob["answer"],
            "predicted_answer": pred,
            "correct": correct,
            "method": method,
        })
    greedy_acc = greedy_correct / len(problems)
    print(f"Greedy CoT pass@1: {greedy_acc:.3f} ({greedy_correct}/{len(problems)})")

    # === ExeVer Pipeline ===
    print("\n" + "=" * 60)
    print("PHASE 3: ExeVer Pipeline (verify + repair)")
    print("=" * 60)

    exever_results = []
    repair_needed = []
    n_valid = 0
    n_executed = 0
    n_all_pass = 0
    n_assertion_fail = 0
    n_runtime_error = 0
    n_empty = 0
    n_syntax_error = 0
    n_timeout = 0
    total_assertions = 0
    total_nontrivial = 0

    for i, (prob, sol, ver_resp) in enumerate(zip(problems, greedy_solutions, verify_responses)):
        pred = extract_model_answer(sol)
        correct, _ = answers_equivalent(pred, prob["answer"])

        script = extract_script(ver_resp)
        if not script.strip():
            n_empty += 1
            exever_results.append({
                "id": prob["id"],
                "gold_answer": prob["answer"],
                "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "NO_SCRIPT",
                "assertions": 0,
                "echo_chamber": None,
                "repaired": False,
                "nl_solution": sol,
            })
            continue

        # Check validity
        try:
            compile(script, "<verify>", "exec")
            n_valid += 1
        except SyntaxError:
            n_syntax_error += 1
            exever_results.append({
                "id": prob["id"],
                "gold_answer": prob["answer"],
                "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "SYNTAX_ERROR",
                "assertions": 0,
                "echo_chamber": None,
                "repaired": False,
                "nl_solution": sol,
            })
            continue

        assertions = extract_assertions(script)
        n_asserts = len(assertions)
        total_assertions += n_asserts
        nontrivial = sum(1 for a in assertions if not _is_trivial_assert(a))
        total_nontrivial += nontrivial

        exec_result = execute_verification_script(script, timeout=30)

        if exec_result.success:
            n_executed += 1
            n_all_pass += 1
            script_answer = exec_result.answer_extracted or ""
            echo = None
            if script_answer:
                ans_ok, _ = answers_equivalent(script_answer, prob["answer"])
                echo = not ans_ok
            else:
                echo = not correct
            exever_results.append({
                "id": prob["id"],
                "gold_answer": prob["answer"],
                "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "ALL_PASS",
                "assertions": n_asserts,
                "echo_chamber": echo,
                "repaired": False,
                "script_answer": script_answer,
                "nl_solution": sol,
            })
        elif exec_result.assertion_error:
            n_executed += 1
            n_assertion_fail += 1
            step = exec_result.error_step
            error_msg = exec_result.error_message

            # Queue for repair
            nl_steps = parse_nl_steps(sol)
            step_idx = step if step >= 0 else 0
            verified_prefix = "\n\n".join(
                f"## Step {j+1}: {s}" for j, s in enumerate(nl_steps[:step_idx])
            )
            failed_step = nl_steps[step_idx] if step_idx < len(nl_steps) else ""

            repair_prompt = REPAIR_PROMPT.format(
                problem=prob["question"],
                verified_prefix=verified_prefix,
                failed_step=failed_step,
                step_num=step_idx + 1,
                error_message=error_msg[:300],
            )
            repair_needed.append({
                "idx": i,
                "repair_prompt": repair_prompt,
                "original_pred": pred,
                "original_correct": correct,
            })
            exever_results.append({
                "id": prob["id"],
                "gold_answer": prob["answer"],
                "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": f"FAIL_STEP_{step}",
                "assertions": n_asserts,
                "echo_chamber": None,
                "repaired": False,
                "nl_solution": sol,
            })
        else:
            if exec_result.timeout:
                verdict = "TIMEOUT"
                n_timeout += 1
            else:
                n_runtime_error += 1
                verdict = "RUNTIME_ERROR"
            exever_results.append({
                "id": prob["id"],
                "gold_answer": prob["answer"],
                "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": verdict,
                "assertions": n_asserts,
                "echo_chamber": None,
                "repaired": False,
                "nl_solution": sol,
            })

    print(f"\nPre-repair stats:")
    print(f"  Valid scripts: {n_valid}/{len(problems)}")
    print(f"  Executed: {n_executed}")
    print(f"  All pass: {n_all_pass}")
    print(f"  Assertion fail: {n_assertion_fail}")
    print(f"  Runtime error: {n_runtime_error}")
    print(f"  Syntax error: {n_syntax_error}")
    print(f"  Timeout: {n_timeout}")
    print(f"  Empty: {n_empty}")
    print(f"  Total assertions: {total_assertions} (avg {total_assertions/max(n_valid,1):.1f}/script)")
    print(f"  Nontrivial assertions: {total_nontrivial} (avg {total_nontrivial/max(n_valid,1):.1f}/script)")

    # === Run Repairs ===
    if repair_needed:
        print(f"\n[Repair] Running repair for {len(repair_needed)} problems...")
        repair_raw = run_repair_inference.remote(json.dumps(repair_needed))
        repair_data = json.loads(repair_raw)

        n_repair_success = 0
        for ri, (repair_info, repaired_sol, reverify_resp) in enumerate(
            zip(repair_needed, repair_data["repairs"], repair_data["reverify"])
        ):
            idx = repair_info["idx"]
            prob = problems[idx]

            # Check repaired answer
            repaired_pred = extract_model_answer(repaired_sol)
            repaired_correct, _ = answers_equivalent(repaired_pred, prob["answer"])

            # Check re-verification
            reverify_script = extract_script(reverify_resp)
            repair_pass = False
            if reverify_script.strip():
                try:
                    compile(reverify_script, "<reverify>", "exec")
                    rexec = execute_verification_script(reverify_script, timeout=30)
                    if rexec.success:
                        repair_pass = True
                        n_repair_success += 1
                except SyntaxError:
                    pass

            if repair_pass and repaired_correct:
                exever_results[idx]["verdict"] = "REPAIRED"
                exever_results[idx]["predicted_answer"] = repaired_pred
                exever_results[idx]["answer_correct"] = True
                exever_results[idx]["repaired"] = True
            elif repaired_correct:
                # Answer correct but verification didn't pass
                exever_results[idx]["predicted_answer"] = repaired_pred
                exever_results[idx]["answer_correct"] = True
                exever_results[idx]["repaired"] = True
                exever_results[idx]["verdict"] = "REPAIRED_UNVERIFIED"

        print(f"  Repair success (verification passes): {n_repair_success}/{len(repair_needed)}")

    # === Compute ExeVer Accuracy ===
    exever_correct = sum(1 for r in exever_results if r["answer_correct"])
    exever_acc = exever_correct / len(problems)
    print(f"\nExeVer accuracy: {exever_acc:.3f} ({exever_correct}/{len(problems)})")

    # === Verification Coverage ===
    # Coverage = fraction of problems where verification script produced a
    # meaningful result (ALL_PASS or FAIL_STEP) vs crashed/empty/error
    n_meaningful = sum(
        1 for r in exever_results
        if r["verdict"] in ("ALL_PASS", "REPAIRED") or r["verdict"].startswith("FAIL_STEP")
    )
    coverage = n_meaningful / len(problems)
    print(f"Verification coverage: {coverage:.3f} ({n_meaningful}/{len(problems)})")

    # === Echo Chamber ===
    echo_results = [r for r in exever_results if r.get("echo_chamber") is not None]
    echo_pos = sum(1 for r in echo_results if r["echo_chamber"])
    echo_rate = echo_pos / len(echo_results) if echo_results else 0
    print(f"Echo chamber rate: {echo_rate:.3f} ({echo_pos}/{len(echo_results)})")

    # === Verdict Distribution ===
    verdicts = Counter(r["verdict"] for r in exever_results)
    print(f"\nVerdict distribution:")
    for v, c in verdicts.most_common():
        print(f"  {v}: {c} ({c/len(problems):.1%})")

    # === Load MATH-500 results for comparison (if available) ===
    math500_comparison = {}
    math500_path = RESULTS_DIR / "exp5_math500_full.json"
    if math500_path.exists():
        with open(math500_path) as f:
            math500 = json.load(f)
        math500_coverage = (
            math500["verification_stats"]["n_all_pass"]
            + math500["verification_stats"]["n_assertion_fail"]
        ) / math500["n_problems"]
        math500_comparison = {
            "math500_greedy_acc": math500["accuracy"]["greedy_cot"],
            "math500_exever_acc": math500["accuracy"]["exever"],
            "math500_coverage": math500_coverage,
            "math500_echo_rate": math500["echo_chamber"]["rate"],
        }
        print(f"\n{'='*60}")
        print(f"COMPARISON: GSM8K vs MATH-500")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'GSM8K':>10} {'MATH-500':>10} {'Delta':>10}")
        print("-" * 57)
        print(f"{'Greedy CoT acc':<25} {greedy_acc:>9.1%} {math500_comparison['math500_greedy_acc']:>9.1%} "
              f"{greedy_acc - math500_comparison['math500_greedy_acc']:>+9.1%}")
        print(f"{'ExeVer acc':<25} {exever_acc:>9.1%} {math500_comparison['math500_exever_acc']:>9.1%} "
              f"{exever_acc - math500_comparison['math500_exever_acc']:>+9.1%}")
        print(f"{'Verification coverage':<25} {coverage:>9.1%} {math500_comparison['math500_coverage']:>9.1%} "
              f"{coverage - math500_comparison['math500_coverage']:>+9.1%}")
        print(f"{'Echo chamber rate':<25} {echo_rate:>9.1%} {math500_comparison['math500_echo_rate']:>9.1%} "
              f"{echo_rate - math500_comparison['math500_echo_rate']:>+9.1%}")

    # === Save Results ===
    output = {
        "experiment": "exp12_gsm8k",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "dataset": "gsm8k",
        "n_problems": len(problems),
        "accuracy": {
            "greedy_cot": greedy_acc,
            "exever": exever_acc,
        },
        "verification_stats": {
            "n_valid": n_valid,
            "n_executed": n_executed,
            "n_all_pass": n_all_pass,
            "n_assertion_fail": n_assertion_fail,
            "n_runtime_error": n_runtime_error,
            "n_syntax_error": n_syntax_error,
            "n_timeout": n_timeout,
            "n_empty": n_empty,
            "total_assertions": total_assertions,
            "total_nontrivial": total_nontrivial,
            "avg_assertions": total_assertions / max(n_valid, 1),
            "avg_nontrivial": total_nontrivial / max(n_valid, 1),
        },
        "coverage": {
            "verification_coverage": coverage,
            "n_meaningful": n_meaningful,
        },
        "echo_chamber": {
            "rate": echo_rate,
            "n_echo": echo_pos,
            "n_total": len(echo_results),
        },
        "repair": {
            "attempted": len(repair_needed),
            "successful": sum(1 for r in exever_results if r.get("repaired", False)),
        },
        "verdicts": dict(verdicts),
        "math500_comparison": math500_comparison,
        "exever_results": exever_results,
    }

    out_path = RESULTS_DIR / "exp12_gsm8k.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 12 SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset:                GSM8K (first {n_problems})")
    print(f"Model:                  Qwen2.5-Math-7B-Instruct")
    print(f"Greedy CoT accuracy:    {greedy_acc:.1%}")
    print(f"ExeVer accuracy:        {exever_acc:.1%}")
    print(f"Verification coverage:  {coverage:.1%}")
    print(f"Echo chamber rate:      {echo_rate:.1%}")
    print(f"Repairs attempted:      {len(repair_needed)}")
    print(f"Repairs successful:     {sum(1 for r in exever_results if r.get('repaired', False))}")

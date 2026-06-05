"""Experiment 3: Cross-Model Verification + Coverage Map.

Uses DeepSeek-R1-Distill-Qwen-7B to generate verification scripts
for the Qwen2.5-Math-7B solutions (from Exp 1). Compares echo chamber
rates: same-model vs cross-model verification.

Also produces the Verifiability Map data (subject × difficulty).
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp3")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.49.0", "vllm==0.7.3", "numpy<2")
)
model_volume = modal.Volume.from_name("exever-models", create_if_missing=True)

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


def extract_script_from_response(response: str) -> str:
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


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=3600,
    scaledown_window=60,
)
def generate_crossmodel_verification(solutions_json: str) -> str:
    """Generate verification scripts using DeepSeek-R1-Distill-Qwen-7B."""
    import json as _json
    from vllm import LLM, SamplingParams

    solutions = _json.loads(solutions_json)

    print("Loading DeepSeek-R1-Distill-Qwen-7B for cross-model verification...")
    llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    from modal import Volume
    Volume.from_name("exever-models").commit()

    params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        stop=["```"],
    )

    prompts = [VERIFY_PROMPT.format(solution=s["response"]) for s in solutions]

    print(f"Generating cross-model verification for {len(prompts)} solutions...")
    t0 = __import__("time").time()
    outputs = llm.generate(prompts, params)
    elapsed = __import__("time").time() - t0
    print(f"Cross-model Pass 2 done in {elapsed:.1f}s")

    results = []
    for i, (s, out) in enumerate(zip(solutions, outputs)):
        verify_response = out.outputs[0].text
        results.append({
            "id": s["id"],
            "level": s["level"],
            "type": s["type"],
            "gold_answer": s["gold_answer"],
            "nl_response": s["response"],
            "predicted_answer": s.get("predicted_answer", ""),
            "answer_correct": s.get("correct", False),
            "verify_response": verify_response,
        })

    return _json.dumps({"results": results, "inference_time": elapsed})


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.exever.executor import execute_verification_script
    from src.exever.step_parser import extract_assertions, parse_nl_steps
    from src.eval.metrics import assertion_quality

    # Load Exp 1 results (Qwen solutions)
    with open(RESULTS_DIR / "exp1_baseline_qwen7b.json") as f:
        exp1_data = json.load(f)
    solutions = exp1_data["results"]

    # Load Exp 2 same-model results for comparison
    with open(RESULTS_DIR / "exp2_twopass_feasibility.json") as f:
        exp2_data = json.load(f)
    same_model_results = {r["id"]: r for r in exp2_data["results"]}

    print(f"Generating cross-model verification scripts (DeepSeek verifier)...")
    raw_result = generate_crossmodel_verification.remote(json.dumps(solutions))
    data = json.loads(raw_result)
    inference_time = data["inference_time"]
    results = data["results"]

    print(f"Cross-model Pass 2 completed in {inference_time:.1f}s")
    print("Evaluating verification scripts locally...")

    # Process cross-model results
    cross_results = []
    n_executed = 0
    n_all_pass = 0
    n_assertion_fail = 0
    n_runtime_error = 0
    total_assertions = 0

    for r in results:
        script = extract_script_from_response(r["verify_response"])
        if not script.strip():
            cross_results.append({
                "id": r["id"], "level": r["level"], "type": r["type"],
                "gold_answer": r["gold_answer"], "answer_correct": r["answer_correct"],
                "verdict": "EMPTY_SCRIPT", "assertions": 0,
                "echo_chamber": None, "script_answer": "",
            })
            continue

        try:
            compile(script, "<verification>", "exec")
        except SyntaxError:
            cross_results.append({
                "id": r["id"], "level": r["level"], "type": r["type"],
                "gold_answer": r["gold_answer"], "answer_correct": r["answer_correct"],
                "verdict": "SYNTAX_ERROR", "assertions": 0,
                "echo_chamber": None, "script_answer": "",
            })
            continue

        assertions = extract_assertions(script)
        n_asserts = len(assertions)
        total_assertions += n_asserts

        exec_result = execute_verification_script(script, timeout=30)

        verdict = "ERROR"
        echo_chamber = None
        script_answer = exec_result.answer_extracted or ""

        if exec_result.success:
            n_executed += 1
            n_all_pass += 1
            verdict = "ALL_PASS"
            if script_answer:
                ans_correct, _ = answers_equivalent(script_answer, r["gold_answer"])
                echo_chamber = not ans_correct
            else:
                echo_chamber = not r["answer_correct"]
        elif exec_result.assertion_error:
            n_executed += 1
            n_assertion_fail += 1
            verdict = f"FAIL_STEP_{exec_result.error_step}"
        elif exec_result.timeout:
            verdict = "TIMEOUT"
        else:
            n_runtime_error += 1
            verdict = "RUNTIME_ERROR"

        cross_results.append({
            "id": r["id"], "level": r["level"], "type": r["type"],
            "gold_answer": r["gold_answer"], "answer_correct": r["answer_correct"],
            "verdict": verdict, "assertions": n_asserts,
            "echo_chamber": echo_chamber, "script_answer": script_answer,
        })

    # === Compare same-model vs cross-model echo chamber ===
    # Same-model echo chamber (from Exp 2)
    same_echo_results = [r for r in exp2_data["results"]
                          if r.get("echo_chamber") is not None]
    same_echo_pos = sum(1 for r in same_echo_results if r["echo_chamber"])
    same_echo_rate = same_echo_pos / len(same_echo_results) if same_echo_results else 0

    # Cross-model echo chamber
    cross_echo_results = [r for r in cross_results if r["echo_chamber"] is not None]
    cross_echo_pos = sum(1 for r in cross_echo_results if r["echo_chamber"])
    cross_echo_rate = cross_echo_pos / len(cross_echo_results) if cross_echo_results else 0

    # === Build Verifiability Map ===
    # Combine same-model + cross-model data for the richest coverage picture
    # Use same-model (Exp 2) as primary since we have more detail
    subjects = sorted(set(r["type"] for r in exp2_data["results"]))
    levels = sorted(set(r["level"] for r in exp2_data["results"]))

    verifiability_map = {}
    for subj in subjects:
        for lv in levels:
            same_subset = [r for r in exp2_data["results"]
                           if r["type"] == subj and r["level"] == lv]
            if not same_subset:
                continue

            total = len(same_subset)
            has_assertions = sum(1 for r in same_subset if r["assertions_total"] > 0)
            executed = sum(1 for r in same_subset if r["executed"])
            all_pass = sum(1 for r in same_subset if r["verdict"] == "ALL_PASS")
            fail = sum(1 for r in same_subset if "FAIL" in r["verdict"])

            n_assertions = sum(r["assertions_nontrivial"] for r in same_subset
                              if r["executed"])
            n_steps = sum(r["n_nl_steps"] for r in same_subset if r["executed"])
            step_coverage = n_assertions / n_steps if n_steps > 0 else 0

            key = f"{subj}_{lv}"
            verifiability_map[key] = {
                "subject": subj,
                "level": lv,
                "n_problems": total,
                "scripts_with_assertions_pct": has_assertions / total if total > 0 else 0,
                "execution_rate": executed / total if total > 0 else 0,
                "all_pass_rate": all_pass / total if total > 0 else 0,
                "fail_rate": fail / total if total > 0 else 0,
                "step_coverage": step_coverage,
            }

    # === Print Results ===
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3 RESULTS: Cross-Model + Coverage Map")
    print(f"{'='*60}")

    print(f"\nEcho Chamber Comparison:")
    print(f"  Same-model (Qwen→Qwen):     {same_echo_rate:.1%} ({same_echo_pos}/{len(same_echo_results)})")
    print(f"  Cross-model (Qwen→DeepSeek): {cross_echo_rate:.1%} ({cross_echo_pos}/{len(cross_echo_results)})")
    delta = same_echo_rate - cross_echo_rate
    print(f"  Reduction:                   {delta:+.1%}")

    print(f"\nCross-Model DeepSeek Verification:")
    cross_executed = sum(1 for r in cross_results if r["verdict"] not in
                         ("EMPTY_SCRIPT", "SYNTAX_ERROR", "RUNTIME_ERROR", "TIMEOUT", "ERROR"))
    cross_total = len(cross_results)
    print(f"  Execution rate: {cross_executed}/{cross_total}")
    print(f"  All pass: {n_all_pass}")
    print(f"  Assertion fail: {n_assertion_fail}")
    print(f"  Runtime error: {n_runtime_error}")

    # Print verifiability map
    print(f"\n{'='*60}")
    print(f"VERIFIABILITY MAP (Step Coverage %)")
    print(f"{'='*60}")
    print(f"{'Subject':<25}", end="")
    for lv in levels:
        print(f"{'L'+str(lv):>8}", end="")
    print(f"{'Avg':>8}")
    print("-" * (25 + 8 * (len(levels) + 1)))

    for subj in subjects:
        print(f"{subj:<25}", end="")
        subj_coverages = []
        for lv in levels:
            key = f"{subj}_{lv}"
            if key in verifiability_map:
                cov = verifiability_map[key]["step_coverage"]
                subj_coverages.append(cov)
                print(f"{cov:>7.1%} ", end="")
            else:
                print(f"{'N/A':>7} ", end="")
        if subj_coverages:
            avg = sum(subj_coverages) / len(subj_coverages)
            print(f"{avg:>7.1%}")
        else:
            print(f"{'N/A':>7}")

    # Print execution success map
    print(f"\n{'='*60}")
    print(f"EXECUTION SUCCESS MAP (All Pass %)")
    print(f"{'='*60}")
    print(f"{'Subject':<25}", end="")
    for lv in levels:
        print(f"{'L'+str(lv):>8}", end="")
    print(f"{'Avg':>8}")
    print("-" * (25 + 8 * (len(levels) + 1)))

    for subj in subjects:
        print(f"{subj:<25}", end="")
        subj_pass = []
        for lv in levels:
            key = f"{subj}_{lv}"
            if key in verifiability_map:
                pr = verifiability_map[key]["all_pass_rate"]
                subj_pass.append(pr)
                print(f"{pr:>7.1%} ", end="")
            else:
                print(f"{'N/A':>7} ", end="")
        if subj_pass:
            avg = sum(subj_pass) / len(subj_pass)
            print(f"{avg:>7.1%}")
        else:
            print(f"{'N/A':>7}")

    # === Per-problem agreement analysis ===
    # For problems where both verifiers ran: do they agree on verdict?
    agree_count = 0
    disagree_count = 0
    both_ran = 0
    cross_by_id = {r["id"]: r for r in cross_results}

    for r in exp2_data["results"]:
        pid = r["id"]
        if pid in cross_by_id:
            same_v = r["verdict"]
            cross_v = cross_by_id[pid]["verdict"]
            # Only compare if both actually executed
            if r["executed"] and cross_v not in ("EMPTY_SCRIPT", "SYNTAX_ERROR",
                                                    "RUNTIME_ERROR", "TIMEOUT", "ERROR"):
                both_ran += 1
                same_pass = same_v == "ALL_PASS"
                cross_pass = cross_v == "ALL_PASS"
                if same_pass == cross_pass:
                    agree_count += 1
                else:
                    disagree_count += 1

    print(f"\nCross-Model Agreement:")
    print(f"  Both ran: {both_ran}")
    if both_ran > 0:
        print(f"  Agree: {agree_count} ({agree_count/both_ran:.1%})")
        print(f"  Disagree: {disagree_count} ({disagree_count/both_ran:.1%})")

    # === Save ===
    output = {
        "experiment": "exp3_crossmodel",
        "solver_model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "verifier_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_problems": len(results),
        "inference_time_s": inference_time,
        "echo_chamber": {
            "same_model_rate": same_echo_rate,
            "same_model_n": len(same_echo_results),
            "cross_model_rate": cross_echo_rate,
            "cross_model_n": len(cross_echo_results),
            "reduction": delta,
        },
        "cross_model_summary": {
            "n_executed": cross_executed,
            "n_all_pass": n_all_pass,
            "n_assertion_fail": n_assertion_fail,
            "n_runtime_error": n_runtime_error,
        },
        "verifiability_map": verifiability_map,
        "agreement": {
            "both_ran": both_ran,
            "agree": agree_count,
            "disagree": disagree_count,
        },
        "cross_results": cross_results,
    }

    out_path = RESULTS_DIR / "exp3_crossmodel.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

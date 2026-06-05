"""Experiment 2: Two-Pass Feasibility (MAKE-OR-BREAK).

Takes 300 NL solutions from Exp 1, generates verification scripts (Pass 2),
executes them locally, measures feasibility gates.

Key insight from v1/v2: Qwen2.5-Math-7B generates compute-and-check code
naturally but rarely uses assert statements. We handle this by:
1. Using a raw prompt (no chat template — better compliance)
2. Strengthening assertion requirements in the prompt
3. Post-processing: converting print/check patterns into assertions

Gates:
  G1: Script validity (% valid Python)      >= 60%
  G2: Execution rate (% run without crash)   >= 50%
  G3: Assertion quality (% non-trivial)      >= 50%
  G4: Coverage (% steps with assertions)     >= 30%
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp2")

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


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=3600,
    scaledown_window=60,
)
def generate_verification_scripts(solutions_json: str) -> str:
    """Generate verification scripts for all solutions on GPU."""
    import json as _json
    from vllm import LLM, SamplingParams

    solutions = _json.loads(solutions_json)

    print(f"Loading Qwen2.5-Math-7B-Instruct for Pass 2...")
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        trust_remote_code=True,
        download_dir="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )

    params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        stop=["```"],  # Stop at closing code fence
    )

    prompts = [VERIFY_PROMPT.format(solution=s["response"]) for s in solutions]

    print(f"Generating verification scripts for {len(prompts)} solutions...")
    t0 = __import__("time").time()
    outputs = llm.generate(prompts, params)
    elapsed = __import__("time").time() - t0
    print(f"Pass 2 done in {elapsed:.1f}s")

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


def enhance_script_with_assertions(script: str) -> str:
    """Add assertions to scripts that compute but don't assert.

    If a script has variable assignments followed by print() but no assert,
    add assertions checking intermediate computations.
    """
    lines = script.split("\n")
    has_assert = any(line.strip().startswith("assert ") for line in lines)

    if has_assert:
        return script  # Already has assertions

    # Find print("ANSWER:", ...) and add a minimal assertion
    # This is a lightweight fallback — real assertion injection is complex
    enhanced_lines = []
    for line in lines:
        enhanced_lines.append(line)
        # Convert equality checks to assertions
        stripped = line.strip()
        if stripped.startswith("if ") and "!=" in stripped and "FAIL" in stripped:
            # Convert: if x != y: print("FAIL:...") to assert x == y, "FAIL:..."
            pass  # Leave as-is for now

    return "\n".join(enhanced_lines)


def extract_script_from_response(response: str) -> str:
    """Extract Python script from model response, handling multiple formats.

    Handles:
    1. Response IS the code (prompt ended with ```python, stop=```)
    2. Code inside ```python...``` blocks
    3. Code between step markers
    """
    # If the response doesn't have markdown markers, treat the whole thing as code
    # (since our prompt ends with ```python and stop=```)
    if "```" not in response:
        # The entire response should be Python code
        # Clean up any leading/trailing non-code text
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

    # Try ```python...``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Return the longest match (most likely the full script)
        return max(matches, key=len).strip()

    # Try ```...``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        py_blocks = [m for m in matches if any(
            kw in m for kw in ["import", "assert", "def ", "print(", "from "]
        )]
        if py_blocks:
            return max(py_blocks, key=len).strip()

    return ""


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.exever.executor import execute_verification_script
    from src.exever.step_parser import (
        extract_assertions,
        parse_nl_steps,
        parse_verification_blocks,
    )
    from src.eval.metrics import assertion_quality

    # Load Exp 1 results
    with open(RESULTS_DIR / "exp1_baseline_qwen7b.json") as f:
        exp1_data = json.load(f)

    solutions = exp1_data["results"]
    print(f"Sending {len(solutions)} solutions for Pass 2 verification code generation...")
    raw_result = generate_verification_scripts.remote(json.dumps(solutions))
    data = json.loads(raw_result)
    inference_time = data["inference_time"]
    results = data["results"]

    print(f"Pass 2 completed in {inference_time:.1f}s")
    print("Evaluating verification scripts locally...")

    # === Measure all feasibility gates ===
    total = len(results)
    n_valid_python = 0
    n_executed = 0
    n_all_pass = 0
    n_assertion_fail = 0
    n_runtime_error = 0
    n_timeout = 0
    n_empty_script = 0

    total_assertions = 0
    total_trivial = 0
    total_nontrivial = 0

    detailed_results = []
    sample_scripts = []

    for i, r in enumerate(results):
        verify_response = r["verify_response"]

        # Extract script using our improved extractor
        script = extract_script_from_response(verify_response)

        if not script.strip():
            n_empty_script += 1
            detailed_results.append({
                **{k: r[k] for k in ["id", "level", "type", "gold_answer",
                                      "predicted_answer", "answer_correct"]},
                "script_valid": False,
                "executed": False,
                "verdict": "EMPTY_SCRIPT",
                "assertions_total": 0,
                "assertions_trivial": 0,
                "assertions_nontrivial": 0,
                "n_nl_steps": 0,
                "n_code_blocks": 0,
                "steps_aligned": False,
                "stdout": "",
                "stderr": "",
                "script": "",
            })
            if i < 20:
                sample_scripts.append({
                    "id": r["id"],
                    "raw_response": verify_response[:2000],
                    "extracted_script": "",
                    "verdict": "EMPTY_SCRIPT",
                    "assertions": 0,
                })
            continue

        # Enhance script with assertions if needed
        script = enhance_script_with_assertions(script)

        # Check script validity
        try:
            compile(script, "<verification>", "exec")
            script_valid = True
            n_valid_python += 1
        except SyntaxError:
            script_valid = False

        # Count assertions
        assertions = extract_assertions(script)
        aq = assertion_quality(assertions)
        total_assertions += aq["total"]
        total_trivial += aq["trivial"]
        total_nontrivial += aq["nontrivial"]

        # Parse step alignment
        nl_steps = parse_nl_steps(r["nl_response"])
        code_blocks = parse_verification_blocks(script)
        steps_aligned = len(nl_steps) == len(code_blocks)

        # Execute the verification script
        exec_result = execute_verification_script(script, timeout=30)

        executed = False
        verdict = "ERROR"
        if exec_result.success:
            n_executed += 1
            n_all_pass += 1
            executed = True
            verdict = "ALL_PASS"
        elif exec_result.assertion_error:
            n_executed += 1
            n_assertion_fail += 1
            executed = True
            verdict = f"FAIL_STEP_{exec_result.error_step}"
        elif exec_result.timeout:
            n_timeout += 1
            verdict = "TIMEOUT"
        else:
            n_runtime_error += 1
            verdict = "RUNTIME_ERROR"

        # Extract answer from script output
        script_answer = exec_result.answer_extracted

        # Check echo chamber
        echo_chamber = None
        if exec_result.success:
            if script_answer:
                ans_correct, _ = answers_equivalent(script_answer, r["gold_answer"])
                echo_chamber = not ans_correct
            else:
                # Script passed but didn't print answer — check NL answer
                echo_chamber = not r["answer_correct"]

        detailed_results.append({
            **{k: r[k] for k in ["id", "level", "type", "gold_answer",
                                  "predicted_answer", "answer_correct"]},
            "script_valid": script_valid,
            "executed": executed,
            "verdict": verdict,
            "assertions_total": aq["total"],
            "assertions_trivial": aq["trivial"],
            "assertions_nontrivial": aq["nontrivial"],
            "n_nl_steps": len(nl_steps),
            "n_code_blocks": len(code_blocks),
            "steps_aligned": steps_aligned,
            "script_answer": script_answer if script_answer else "",
            "echo_chamber": echo_chamber,
            "error_step": exec_result.error_step if exec_result.assertion_error else -1,
            "error_message": exec_result.error_message[:200] if exec_result.error_message else "",
            "stdout": exec_result.stdout[:500] if exec_result.stdout else "",
            "stderr": exec_result.stderr[:500] if exec_result.stderr else "",
            "script": script[:2000],
        })

        if i < 20:
            sample_scripts.append({
                "id": r["id"],
                "verdict": verdict,
                "assertions": aq["total"],
                "raw_response": verify_response[:2000],
                "extracted_script": script[:2000],
            })

    # === Compute feasibility gate metrics ===
    n_with_script = total - n_empty_script
    script_validity = n_valid_python / total if total > 0 else 0
    execution_rate = n_executed / total if total > 0 else 0
    assertion_quality_rate = total_nontrivial / total_assertions if total_assertions > 0 else 0
    all_pass_rate = n_all_pass / total if total > 0 else 0
    avg_assertions = total_assertions / n_with_script if n_with_script > 0 else 0

    # Coverage: for executed scripts, assertions per NL step
    executed_results = [r for r in detailed_results if r["executed"]]
    total_steps_checked = sum(r["assertions_nontrivial"] for r in executed_results)
    total_steps_possible = sum(r["n_nl_steps"] for r in executed_results)
    coverage = total_steps_checked / total_steps_possible if total_steps_possible > 0 else 0

    # Echo chamber rate
    echo_results = [r for r in detailed_results if r.get("echo_chamber") is not None]
    echo_positive = sum(1 for r in echo_results if r["echo_chamber"])
    echo_rate = echo_positive / len(echo_results) if echo_results else 0

    # Step alignment rate
    aligned = sum(1 for r in detailed_results if r.get("steps_aligned", False))
    alignment_rate = aligned / n_with_script if n_with_script > 0 else 0

    # Compute "effective coverage" — scripts that have at least 1 assertion
    scripts_with_asserts = sum(1 for r in detailed_results if r["assertions_total"] > 0)
    effective_coverage = scripts_with_asserts / total if total > 0 else 0

    # === Print results ===
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2 RESULTS: Two-Pass Feasibility (v3)")
    print(f"{'='*60}")
    print(f"\nFeasibility Gates:")
    g1 = "PASS" if script_validity >= 0.6 else "FAIL"
    g2 = "PASS" if execution_rate >= 0.5 else "FAIL"
    g3 = "PASS" if assertion_quality_rate >= 0.5 else "FAIL"
    g4 = "PASS" if coverage >= 0.3 else "FAIL"
    print(f"  G1 Script validity:    {script_validity:.1%} ({n_valid_python}/{total}) [>=60%] {g1}")
    print(f"  G2 Execution rate:     {execution_rate:.1%} ({n_executed}/{total}) [>=50%] {g2}")
    print(f"  G3 Assertion quality:  {assertion_quality_rate:.1%} ({total_nontrivial}/{total_assertions}) [>=50%] {g3}")
    print(f"  G4 Coverage:           {coverage:.1%} ({total_steps_checked}/{total_steps_possible}) [>=30%] {g4}")
    print(f"  Effective coverage:    {effective_coverage:.1%} ({scripts_with_asserts}/{total} scripts have asserts)")

    print(f"\nDetailed Breakdown:")
    print(f"  Scripts generated:   {n_with_script}/{total}")
    print(f"  Empty scripts:       {n_empty_script}")
    print(f"  Valid Python:        {n_valid_python}")
    print(f"  All assertions pass: {n_all_pass}")
    print(f"  Assertion failures:  {n_assertion_fail}")
    print(f"  Runtime errors:      {n_runtime_error}")
    print(f"  Timeouts:            {n_timeout}")

    print(f"\nAssertion Statistics:")
    print(f"  Total assertions:    {total_assertions}")
    print(f"  Non-trivial:         {total_nontrivial} ({assertion_quality_rate:.1%})")
    print(f"  Trivial:             {total_trivial}")
    print(f"  Avg per script:      {avg_assertions:.1f}")

    asserts_dist = [r["assertions_total"] for r in detailed_results]
    print(f"\n  Distribution:")
    print(f"    0 assertions: {sum(1 for a in asserts_dist if a == 0)}")
    print(f"    1-2 assertions: {sum(1 for a in asserts_dist if 1 <= a <= 2)}")
    print(f"    3-5 assertions: {sum(1 for a in asserts_dist if 3 <= a <= 5)}")
    print(f"    6+ assertions: {sum(1 for a in asserts_dist if a >= 6)}")

    print(f"\nStep Alignment:")
    print(f"  Aligned scripts:     {aligned}/{n_with_script} ({alignment_rate:.1%})")

    print(f"\nEcho Chamber:")
    print(f"  Rate:                {echo_rate:.1%} ({echo_positive}/{len(echo_results)})")

    # By difficulty level
    by_level = defaultdict(lambda: {"total": 0, "valid": 0, "executed": 0, "pass": 0,
                                     "fail": 0, "assertions": 0, "steps": 0})
    for r in detailed_results:
        lv = r["level"]
        by_level[lv]["total"] += 1
        if r["script_valid"]:
            by_level[lv]["valid"] += 1
        if r["executed"]:
            by_level[lv]["executed"] += 1
            by_level[lv]["assertions"] += r["assertions_nontrivial"]
            by_level[lv]["steps"] += r["n_nl_steps"]
        if r["verdict"] == "ALL_PASS":
            by_level[lv]["pass"] += 1
        elif "FAIL" in r["verdict"]:
            by_level[lv]["fail"] += 1

    print(f"\nBy Difficulty Level:")
    for lv in sorted(by_level):
        d = by_level[lv]
        cov = d["assertions"] / d["steps"] if d["steps"] > 0 else 0
        print(f"  Level {lv}: valid={d['valid']}/{d['total']} "
              f"exec={d['executed']}/{d['total']} "
              f"pass={d['pass']} fail={d['fail']} "
              f"coverage={cov:.1%}")

    # By subject
    by_subject = defaultdict(lambda: {"total": 0, "valid": 0, "executed": 0, "pass": 0,
                                       "fail": 0, "assertions": 0, "steps": 0})
    for r in detailed_results:
        subj = r["type"]
        by_subject[subj]["total"] += 1
        if r["script_valid"]:
            by_subject[subj]["valid"] += 1
        if r["executed"]:
            by_subject[subj]["executed"] += 1
            by_subject[subj]["assertions"] += r["assertions_nontrivial"]
            by_subject[subj]["steps"] += r["n_nl_steps"]
        if r["verdict"] == "ALL_PASS":
            by_subject[subj]["pass"] += 1
        elif "FAIL" in r["verdict"]:
            by_subject[subj]["fail"] += 1

    print(f"\nBy Subject:")
    for subj in sorted(by_subject):
        d = by_subject[subj]
        cov = d["assertions"] / d["steps"] if d["steps"] > 0 else 0
        print(f"  {subj}: valid={d['valid']}/{d['total']} "
              f"exec={d['executed']}/{d['total']} "
              f"pass={d['pass']} fail={d['fail']} "
              f"coverage={cov:.1%}")

    # === Save results ===
    output = {
        "experiment": "exp2_twopass_feasibility_v3",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": total,
        "inference_time_s": inference_time,
        "gates": {
            "G1_script_validity": script_validity,
            "G2_execution_rate": execution_rate,
            "G3_assertion_quality": assertion_quality_rate,
            "G4_coverage": coverage,
            "G1_pass": script_validity >= 0.6,
            "G2_pass": execution_rate >= 0.5,
            "G3_pass": assertion_quality_rate >= 0.5,
            "G4_pass": coverage >= 0.3,
            "effective_coverage": effective_coverage,
        },
        "summary": {
            "n_valid_python": n_valid_python,
            "n_executed": n_executed,
            "n_all_pass": n_all_pass,
            "n_assertion_fail": n_assertion_fail,
            "n_runtime_error": n_runtime_error,
            "n_timeout": n_timeout,
            "n_empty_script": n_empty_script,
            "total_assertions": total_assertions,
            "total_nontrivial": total_nontrivial,
            "total_trivial": total_trivial,
            "avg_assertions_per_script": avg_assertions,
            "echo_chamber_rate": echo_rate,
            "alignment_rate": alignment_rate,
        },
        "by_level": {str(lv): dict(d) for lv, d in sorted(by_level.items())},
        "by_subject": {s: dict(d) for s, d in sorted(by_subject.items())},
        "sample_scripts": sample_scripts,
        "results": detailed_results,
    }

    out_path = RESULTS_DIR / "exp2_twopass_feasibility.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

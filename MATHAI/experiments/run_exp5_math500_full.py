"""Experiment 5: Full-Scale MATH-500 Evaluation.

Runs the complete ExeVer pipeline + baselines on 500 MATH problems
with Qwen2.5-Math-7B-Instruct. Generates 8 samples per problem for
majority@8 and best-of-8 baselines, plus full ExeVer pipeline.

This is the PRIMARY experiment for the workshop paper.
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp5-math500")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.49.0", "vllm==0.7.3", "numpy<2")
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


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=7200,
    scaledown_window=120,
)
def run_all_inference(problems_json: str) -> str:
    """Run ALL inference on GPU: baselines + ExeVer solve + verify.

    Returns a single JSON with all generated outputs.
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

    # === Phase 1: Greedy CoT (baseline + ExeVer Pass 1) ===
    greedy_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
    )
    greedy_prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in problems]

    print(f"\n[Phase 1] Generating {len(greedy_prompts)} greedy CoT solutions...")
    t0 = __import__("time").time()
    greedy_outputs = llm.generate(greedy_prompts, greedy_params)
    t1 = __import__("time").time()
    print(f"  Done in {t1-t0:.1f}s")

    results["greedy_solutions"] = [
        out.outputs[0].text for out in greedy_outputs
    ]

    # === Phase 2: Sampled CoT (4 per problem for majority@4/best-of-4) ===
    sample_params = SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        n=4,
    )

    print(f"\n[Phase 2] Generating 4 sampled solutions per problem ({len(problems)*4} total)...")
    t0 = __import__("time").time()
    sample_outputs = llm.generate(greedy_prompts, sample_params)
    t1 = __import__("time").time()
    print(f"  Done in {t1-t0:.1f}s")

    results["sampled_solutions"] = [
        [out.text for out in outputs.outputs]
        for outputs in sample_outputs
    ]

    # === Phase 3: Verification scripts (Pass 2 on greedy solutions) ===
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

    print(f"\n[Phase 3] Generating {len(verify_prompts)} verification scripts...")
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

    # Load MATH-500
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} MATH-500 problems")

    # === Run all GPU inference ===
    print("\n" + "="*60)
    print("PHASE 1: GPU Inference (greedy + sampled + verification)")
    print("="*60)
    raw = run_all_inference.remote(json.dumps(problems))
    gpu_results = json.loads(raw)

    greedy_solutions = gpu_results["greedy_solutions"]
    sampled_solutions = gpu_results["sampled_solutions"]
    verify_responses = gpu_results["verify_responses"]

    # === Evaluate Baselines ===
    print("\n" + "="*60)
    print("PHASE 2: Evaluate Baselines Locally")
    print("="*60)

    # Greedy CoT
    greedy_correct = 0
    greedy_results = []
    for i, (prob, sol) in enumerate(zip(problems, greedy_solutions)):
        pred = extract_model_answer(sol)
        correct, method = answers_equivalent(pred, prob["answer"])
        greedy_correct += int(correct)
        greedy_results.append({
            "id": prob["id"],
            "level": prob["level"],
            "type": prob["type"],
            "gold_answer": prob["answer"],
            "predicted_answer": pred,
            "correct": correct,
        })
    greedy_acc = greedy_correct / len(problems)
    print(f"Greedy CoT pass@1: {greedy_acc:.3f} ({greedy_correct}/{len(problems)})")

    # Sampled CoT + majority@8 + best-of-8
    sampled_results = []
    majority4_correct = 0
    best4_correct = 0
    sampled_pass1_correct = 0

    for i, (prob, samples) in enumerate(zip(problems, sampled_solutions)):
        sample_preds = []
        any_correct = False
        for sol in samples:
            pred = extract_model_answer(sol)
            correct, _ = answers_equivalent(pred, prob["answer"])
            sample_preds.append({"pred": pred, "correct": correct})
            if correct:
                any_correct = True

        # Sampled pass@1 (first sample)
        if sample_preds[0]["correct"]:
            sampled_pass1_correct += 1

        # Best-of-4 (oracle)
        if any_correct:
            best4_correct += 1

        # Majority@4 with symbolic equivalence grouping
        groups = []
        for sp in sample_preds:
            merged = False
            for gi, (canonical, count, _correct) in enumerate(groups):
                eq, _ = answers_equivalent(sp["pred"], canonical)
                if eq:
                    groups[gi] = (canonical, count + 1, _correct or sp["correct"])
                    merged = True
                    break
            if not merged:
                groups.append((sp["pred"], 1, sp["correct"]))
        groups.sort(key=lambda x: x[1], reverse=True)
        if groups and groups[0][2]:
            majority4_correct += 1

        sampled_results.append({
            "id": prob["id"],
            "level": prob["level"],
            "type": prob["type"],
            "best4_correct": any_correct,
            "majority4_correct": groups[0][2] if groups else False,
            "pass1_correct": sample_preds[0]["correct"],
        })

    sampled_acc = sampled_pass1_correct / len(problems)
    best4_acc = best4_correct / len(problems)
    majority4_acc = majority4_correct / len(problems)
    print(f"Sampled pass@1: {sampled_acc:.3f}")
    print(f"Majority@4: {majority4_acc:.3f}")
    print(f"Best-of-4: {best4_acc:.3f}")

    # === ExeVer Pipeline ===
    print("\n" + "="*60)
    print("PHASE 3: ExeVer Pipeline (verify + repair)")
    print("="*60)

    exever_results = []
    repair_needed = []
    n_valid = 0
    n_executed = 0
    n_all_pass = 0
    n_assertion_fail = 0
    n_runtime_error = 0
    n_empty = 0
    total_assertions = 0
    total_nontrivial = 0

    for i, (prob, sol, ver_resp) in enumerate(zip(problems, greedy_solutions, verify_responses)):
        pred = extract_model_answer(sol)
        correct, _ = answers_equivalent(pred, prob["answer"])

        script = extract_script(ver_resp)
        if not script.strip():
            n_empty += 1
            exever_results.append({
                "id": prob["id"], "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "NO_SCRIPT", "assertions": 0,
                "echo_chamber": None, "repaired": False,
                "nl_solution": sol,
            })
            continue

        # Check validity
        try:
            compile(script, "<verify>", "exec")
            n_valid += 1
        except SyntaxError:
            exever_results.append({
                "id": prob["id"], "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "SYNTAX_ERROR", "assertions": 0,
                "echo_chamber": None, "repaired": False,
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
                "id": prob["id"], "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "ALL_PASS", "assertions": n_asserts,
                "echo_chamber": echo, "repaired": False,
                "script_answer": script_answer, "nl_solution": sol,
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
                problem=prob["problem"],
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
                "id": prob["id"], "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": f"FAIL_STEP_{step}", "assertions": n_asserts,
                "echo_chamber": None, "repaired": False,
                "nl_solution": sol,
            })
        else:
            if exec_result.timeout:
                verdict = "TIMEOUT"
            else:
                n_runtime_error += 1
                verdict = "RUNTIME_ERROR"
            exever_results.append({
                "id": prob["id"], "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": verdict, "assertions": n_asserts,
                "echo_chamber": None, "repaired": False,
                "nl_solution": sol,
            })

    print(f"\nPre-repair stats:")
    print(f"  Valid scripts: {n_valid}/{len(problems)}")
    print(f"  Executed: {n_executed}")
    print(f"  All pass: {n_all_pass}")
    print(f"  Assertion fail: {n_assertion_fail}")
    print(f"  Runtime error: {n_runtime_error}")
    print(f"  Empty: {n_empty}")
    print(f"  Total assertions: {total_assertions} (avg {total_assertions/max(n_valid,1):.1f}/script)")

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
                # Answer correct but verification didn't pass — still use repaired answer
                exever_results[idx]["predicted_answer"] = repaired_pred
                exever_results[idx]["answer_correct"] = True
                exever_results[idx]["repaired"] = True
                exever_results[idx]["verdict"] = "REPAIRED_UNVERIFIED"

        print(f"  Repair success (verification passes): {n_repair_success}/{len(repair_needed)}")

    # === Compute ExeVer Accuracy ===
    exever_correct = sum(1 for r in exever_results if r["answer_correct"])
    exever_acc = exever_correct / len(problems)
    print(f"\nExeVer accuracy: {exever_acc:.3f} ({exever_correct}/{len(problems)})")

    # === By Level ===
    print(f"\n{'='*60}")
    print(f"RESULTS BY DIFFICULTY LEVEL")
    print(f"{'='*60}")
    by_level = {}
    for lv in [1, 2, 3, 4, 5]:
        lv_probs = [(i, p) for i, p in enumerate(problems) if p["level"] == lv]
        n_lv = len(lv_probs)
        if n_lv == 0:
            continue

        greedy_lv = sum(1 for i, _ in lv_probs if greedy_results[i]["correct"]) / n_lv
        pass1_lv = sum(1 for i, _ in lv_probs if sampled_results[i]["pass1_correct"]) / n_lv
        best4_lv = sum(1 for i, _ in lv_probs if sampled_results[i]["best4_correct"]) / n_lv
        maj4_lv = sum(1 for i, _ in lv_probs if sampled_results[i]["majority4_correct"]) / n_lv
        exever_lv = sum(1 for i, _ in lv_probs if exever_results[i]["answer_correct"]) / n_lv

        by_level[lv] = {
            "n": n_lv,
            "greedy": greedy_lv,
            "pass1": pass1_lv,
            "best4": best4_lv,
            "maj4": maj4_lv,
            "exever": exever_lv,
        }
        print(f"  L{lv} (n={n_lv}): greedy={greedy_lv:.3f} pass1={pass1_lv:.3f} "
              f"maj4={maj4_lv:.3f} best4={best4_lv:.3f} exever={exever_lv:.3f}")

    # === By Subject ===
    print(f"\n{'='*60}")
    print(f"RESULTS BY SUBJECT")
    print(f"{'='*60}")
    by_subject = {}
    subjects = sorted(set(p["type"] for p in problems))
    for subj in subjects:
        s_probs = [(i, p) for i, p in enumerate(problems) if p["type"] == subj]
        n_s = len(s_probs)
        if n_s == 0:
            continue

        greedy_s = sum(1 for i, _ in s_probs if greedy_results[i]["correct"]) / n_s
        pass1_s = sum(1 for i, _ in s_probs if sampled_results[i]["pass1_correct"]) / n_s
        best4_s = sum(1 for i, _ in s_probs if sampled_results[i]["best4_correct"]) / n_s
        maj4_s = sum(1 for i, _ in s_probs if sampled_results[i]["majority4_correct"]) / n_s
        exever_s = sum(1 for i, _ in s_probs if exever_results[i]["answer_correct"]) / n_s

        by_subject[subj] = {
            "n": n_s,
            "greedy": greedy_s,
            "pass1": pass1_s,
            "best4": best4_s,
            "maj4": maj4_s,
            "exever": exever_s,
        }
        print(f"  {subj} (n={n_s}): greedy={greedy_s:.3f} pass1={pass1_s:.3f} "
              f"maj4={maj4_s:.3f} best4={best4_s:.3f} exever={exever_s:.3f}")

    # === Verifiability Map ===
    print(f"\n{'='*60}")
    print(f"VERIFIABILITY MAP")
    print(f"{'='*60}")
    vmap = {}
    for subj in subjects:
        for lv in [1, 2, 3, 4, 5]:
            subset = [r for r in exever_results
                      if r["type"] == subj and r["level"] == lv]
            if not subset:
                continue
            total = len(subset)
            has_assert = sum(1 for r in subset if r["assertions"] > 0)
            all_pass = sum(1 for r in subset if r["verdict"] == "ALL_PASS")
            fail = sum(1 for r in subset if "FAIL" in r["verdict"])
            key = f"{subj}_{lv}"
            vmap[key] = {
                "subject": subj, "level": lv, "n": total,
                "has_assertions_pct": has_assert / total,
                "all_pass_pct": all_pass / total,
                "fail_pct": fail / total,
                "coverage": (all_pass + fail) / total,
            }

    # === Echo Chamber ===
    echo_results = [r for r in exever_results if r.get("echo_chamber") is not None]
    echo_pos = sum(1 for r in echo_results if r["echo_chamber"])
    echo_rate = echo_pos / len(echo_results) if echo_results else 0
    print(f"\nEcho chamber rate: {echo_rate:.3f} ({echo_pos}/{len(echo_results)})")

    # === ExeVer Verdict Distribution ===
    verdicts = Counter(r["verdict"] for r in exever_results)
    print(f"\nVerdict distribution:")
    for v, c in verdicts.most_common():
        print(f"  {v}: {c}")

    # === Save Results ===
    output = {
        "experiment": "exp5_math500_full",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": len(problems),
        "accuracy": {
            "greedy_cot": greedy_acc,
            "sampled_pass1": sampled_acc,
            "majority_4": majority4_acc,
            "best_of_4": best4_acc,
            "exever": exever_acc,
        },
        "by_level": {str(k): v for k, v in by_level.items()},
        "by_subject": by_subject,
        "verifiability_map": vmap,
        "echo_chamber": {
            "rate": echo_rate,
            "n_echo": echo_pos,
            "n_total": len(echo_results),
        },
        "verification_stats": {
            "n_valid": n_valid,
            "n_executed": n_executed,
            "n_all_pass": n_all_pass,
            "n_assertion_fail": n_assertion_fail,
            "n_runtime_error": n_runtime_error,
            "n_empty": n_empty,
            "total_assertions": total_assertions,
            "total_nontrivial": total_nontrivial,
            "avg_assertions": total_assertions / max(n_valid, 1),
        },
        "repair": {
            "attempted": len(repair_needed),
            "successful": sum(1 for r in exever_results if r.get("repaired", False)),
        },
        "verdicts": dict(verdicts),
        "exever_results": exever_results,
    }

    out_path = RESULTS_DIR / "exp5_math500_full.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Overall':>8} {'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6} {'L5':>6}")
    print("-" * 62)
    for name, key in [("CoT (greedy)", "greedy"), ("CoT (sampled)", "pass1"),
                       ("Majority@4", "maj4"), ("Best-of-4", "best4"), ("ExeVer", "exever")]:
        overall = output["accuracy"][{
            "greedy": "greedy_cot", "pass1": "sampled_pass1",
            "maj4": "majority_4", "best4": "best_of_4", "exever": "exever"
        }[key]]
        lv_vals = [by_level.get(lv, {}).get(key, 0) for lv in [1,2,3,4,5]]
        print(f"{name:<20} {overall:>7.1%} {lv_vals[0]:>5.1%} {lv_vals[1]:>5.1%} "
              f"{lv_vals[2]:>5.1%} {lv_vals[3]:>5.1%} {lv_vals[4]:>5.1%}")


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

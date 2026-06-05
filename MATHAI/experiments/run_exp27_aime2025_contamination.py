"""Experiment 27: AIME 2025 contamination-clean replication.

Runs the full pipeline (ExeVer + SGRV + SC + verbalized) on AIME 2025 problems
released AFTER Qwen2.5-Math-7B's training cutoff, using Qwen2.5-7B-Instruct-Turbo
via Together API as both solver and verifier.

Addresses W13 from the NeurIPS review: Wu et al. 2025 (arXiv:2507.10532) showed
Qwen2.5-Math-7B memorizes 54.6% of MATH-500. AIME 2025 is the clean control.

Primary questions:
1. Does ExeVer's 13.8% FAR hold on AIME 2025, or was it inflated by memorization?
2. Does SGRV's 100% top-tier precision hold on AIME 2025?
3. Does self-consistency still outperform SGRV on global AUROC?

Output: results/exp27_aime2025_contamination.json
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval.answer_check import answers_equivalent, extract_model_answer
from src.pbt.pipeline import run_pbt
from src.exever.pipeline import run_exever

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key, timeout=120.0, max_retries=2)


def llm_call(prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    """Single LLM call with retries."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def llm_call_n(prompt: str, n: int = 4, max_tokens: int = 2048, temperature: float = 0.7):
    """N samples."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        return [c.message.content for c in resp.choices]
    except Exception:
        return [llm_call(prompt, max_tokens=max_tokens, temperature=temperature) for _ in range(n)]


SOLVE_PROMPT = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

## Step 2: [brief title]
[reasoning and computation for this step]

...continue for all steps...

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""

CONF_PROMPT = """You solved this problem:

Problem: {problem}

Your solution:
{solution}

On a scale of 0.0 to 1.0, how confident are you that your final answer is correct? Respond with ONLY a single number between 0.0 and 1.0, nothing else."""


def process_problem(prob):
    """Run full pipeline on one AIME problem."""
    t0 = time.time()

    # Generate 4 samples for self-consistency (sample0 is the "primary" solution)
    samples = llm_call_n(SOLVE_PROMPT.format(problem=prob["problem"]), n=4)

    # Answer check on first sample
    predicted = [extract_model_answer(s) for s in samples]
    correct = [answers_equivalent(p, prob["answer"])[0] for p in predicted]

    # Self-consistency
    groups = []
    for ans, ok in zip(predicted, correct):
        merged = False
        for gi, (canon, cnt, any_ok) in enumerate(groups):
            eq, _ = answers_equivalent(ans, canon)
            if eq:
                groups[gi] = (canon, cnt + 1, any_ok or ok)
                merged = True
                break
        if not merged:
            groups.append((ans, 1, ok))
    groups.sort(key=lambda x: x[1], reverse=True)
    sc_conf = groups[0][1] / len(samples)

    # Verbalized confidence
    verb_conf = 0.5
    try:
        raw = llm_call(CONF_PROMPT.format(problem=prob["problem"], solution=samples[0]), max_tokens=20, temperature=0.0)
        m = re.search(r"(\d+\.\d+|\d+)", raw.strip())
        if m:
            v = float(m.group(1))
            if v > 1.0:
                v = v / 100.0 if v <= 100 else 0.5
            verb_conf = v
    except Exception:
        pass

    # SGRV (property-based testing)
    pbt = run_pbt(
        problem=prob["problem"],
        solution=samples[0],
        gold_answer=prob["answer"],
        problem_id=prob["id"],
    )
    sgrv_conf = 1.0 if (pbt.all_tested_pass and pbt.n_testable > 0) else 0.3

    # ExeVer (same-model two-pass verification)
    # Pass 1 is already done (samples[0]); wrap it as a pre-computed solver
    # so run_exever uses our existing solution and only runs Pass 2 (verify).
    try:
        def solver_fn(_prompt):
            return samples[0]
        def verifier_fn(prompt):
            return llm_call(prompt, max_tokens=2048, temperature=0.0)

        exever = run_exever(
            problem=prob["problem"],
            gold_answer=prob["answer"],
            solver_fn=solver_fn,
            verifier_fn=verifier_fn,
            problem_id=prob["id"],
            max_repairs=0,
            max_backtrack=0,
            execution_timeout=30,
        )
        exever_allpass = bool(exever.execution_result and exever.execution_result.success)
        exever_script_len = len(exever.verification_script or "")
        exever_n_assertions = exever.n_assertions
    except Exception as e:
        exever_allpass = False
        exever_script_len = 0
        exever_n_assertions = 0
        print(f"  ExeVer error on {prob['id']}: {type(e).__name__}: {str(e)[:100]}", flush=True)

    return {
        "id": prob["id"],
        "type": prob.get("type", ""),
        "sample0_correct": correct[0],
        "sample0_answer": predicted[0],
        "gold_answer": prob["answer"],
        "sc_confidence": sc_conf,
        "sc_selected_correct": groups[0][2],
        "verb_confidence": verb_conf,
        "sgrv_confidence": sgrv_conf,
        "sgrv_all_pass": pbt.all_tested_pass and pbt.n_testable > 0,
        "sgrv_n_testable": pbt.n_testable,
        "exever_all_pass": exever_allpass,
        "exever_script_len": exever_script_len,
        "exever_n_assertions": exever_n_assertions,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    with open(RESULTS_DIR / "aime_2025.json") as f:
        problems = json.load(f)
    print(f"AIME 2025: {len(problems)} problems")
    print(f"Model: {MODEL}")
    print("=" * 60, flush=True)

    out_path = RESULTS_DIR / "exp27_aime2025_contamination.json"
    results = []
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            if isinstance(existing, dict) and "results" in existing:
                results = existing["results"]
            elif isinstance(existing, list):
                results = existing
            done_ids = set(r["id"] for r in results)
            print(f"  Resuming from {len(done_ids)} completed problems", flush=True)
            problems = [p for p in problems if p["id"] not in done_ids]
        except Exception:
            pass

    for i, prob in enumerate(problems):
        try:
            r = process_problem(prob)
            results.append(r)
        except Exception as e:
            print(f"  [{i+1}] {prob['id']}: ERROR {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue

        if (i + 1) % 1 == 0:
            acc = sum(1 for x in results if x["sample0_correct"]) / len(results)
            print(f"  [{len(results)}/30] {prob['id']} correct={r['sample0_correct']} "
                  f"sgrv={r['sgrv_confidence']} exever={'PASS' if r['exever_all_pass'] else 'FAIL'} "
                  f"({r['elapsed_sec']}s, acc={acc:.2f})", flush=True)
            # Save after every problem (n=30 is small, resilience is cheap)
            with open(out_path, "w") as f:
                json.dump({"model": MODEL, "benchmark": "AIME 2025", "n": len(results), "results": results}, f, indent=2)

    # Final save
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "benchmark": "AIME 2025", "n": len(results), "results": results}, f, indent=2)

    # Quick summary
    n = len(results)
    acc = sum(1 for r in results if r["sample0_correct"]) / n
    sgrv_pass = [r for r in results if r["sgrv_all_pass"]]
    sgrv_pass_correct = sum(1 for r in sgrv_pass if r["sample0_correct"])
    exever_pass = [r for r in results if r["exever_all_pass"]]
    exever_pass_correct = sum(1 for r in exever_pass if r["sample0_correct"])

    print()
    print("=" * 60)
    print(f"AIME 2025 CONTAMINATION-CLEAN RESULTS (n={n})")
    print("=" * 60)
    print(f"Baseline accuracy: {acc:.3f} ({int(acc*n)}/{n})")
    print()
    print(f"SGRV:")
    print(f"  top tier (all_pass): {len(sgrv_pass)}/{n} = {len(sgrv_pass)/n:.1%} coverage")
    if sgrv_pass:
        print(f"  top tier accuracy: {sgrv_pass_correct}/{len(sgrv_pass)} = {sgrv_pass_correct/len(sgrv_pass):.3f}")
    print()
    print(f"ExeVer (same-model):")
    print(f"  all_pass: {len(exever_pass)}/{n} = {len(exever_pass)/n:.1%}")
    if exever_pass:
        far = 1 - (exever_pass_correct / len(exever_pass))
        print(f"  FAR: {far:.3f} ({len(exever_pass) - exever_pass_correct}/{len(exever_pass)} wrong)")
    print()
    print("Compare to MATH-500:")
    print("  MATH-500 baseline acc (n=175):      0.766")
    print("  MATH-500 SGRV top-tier coverage:     33.7%")
    print("  MATH-500 SGRV top-tier accuracy:     1.000")
    print("  MATH-500 ExeVer FAR (Qwen-Math-7B):  0.138")


if __name__ == "__main__":
    main()

"""Experiment 11: LLM-as-Judge Baseline (B8).

Tests whether a strong LLM judge can replace code execution for
step-level verification. Uses Qwen2.5-Math-7B-Instruct as both
solver and judge.

Variants:
  1. Greedy CoT + LLM judge evaluation (step-level correctness)
  2. Judge-based rejection sampling: generate 4 solutions, judge scores
     each, pick the highest-scored one

Key question: Is code execution necessary, or can a neural judge
do the same job?
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import modal

RESULTS_DIR = Path("results")

app = modal.App("exever-exp11-llm-judge")

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

JUDGE_PROMPT = """Review the following math solution step by step. For each step, determine if it is correct.

Solution:
{solution}

For each step, write:
Step k: CORRECT or INCORRECT (with brief explanation if incorrect)

After reviewing all steps, give an overall score from 0-10 and state whether the final answer is likely correct.

Overall score: X/10
Final answer likely correct: YES/NO"""


def parse_judge_output(judge_text: str) -> dict:
    """Parse structured fields from the LLM judge response.

    Returns dict with:
      - score: int 0-10 (or -1 if unparseable)
      - likely_correct: bool or None
      - step_judgments: list of (step_num, "CORRECT"|"INCORRECT")
    """
    result: dict = {
        "score": -1,
        "likely_correct": None,
        "step_judgments": [],
    }

    # Extract overall score
    score_match = re.search(r"Overall score:\s*(\d+)\s*/\s*10", judge_text)
    if score_match:
        result["score"] = int(score_match.group(1))

    # Extract likely correct verdict
    likely_match = re.search(
        r"Final answer likely correct:\s*(YES|NO)",
        judge_text,
        re.IGNORECASE,
    )
    if likely_match:
        result["likely_correct"] = likely_match.group(1).upper() == "YES"

    # Extract per-step judgments
    step_pattern = re.compile(
        r"Step\s+(\d+)\s*:\s*(CORRECT|INCORRECT)", re.IGNORECASE
    )
    for m in step_pattern.finditer(judge_text):
        step_num = int(m.group(1))
        verdict = m.group(2).upper()
        result["step_judgments"].append((step_num, verdict))

    return result


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={"/models": model_volume},
    timeout=7200,
    scaledown_window=120,
)
def run_judge_inference(problems_json: str) -> str:
    """Generate greedy + sampled solutions, then run LLM judge on all of them.

    Pipeline:
      Phase 1: Greedy CoT (1 per problem)
      Phase 2: Sampled CoT (4 per problem)
      Phase 3: LLM judge on greedy solutions
      Phase 4: LLM judge on each of the 4 sampled solutions
    """
    import json as _json
    from vllm import LLM, SamplingParams

    problems = _json.loads(problems_json)

    print("Loading Qwen2.5-Math-7B-Instruct on H100...")
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

    # === Phase 1: Greedy CoT ===
    greedy_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
    )
    solve_prompts = [SOLVE_PROMPT.format(problem=p["problem"]) for p in problems]

    print(f"\n[Phase 1] Generating {len(solve_prompts)} greedy CoT solutions...")
    t0 = __import__("time").time()
    greedy_outputs = llm.generate(solve_prompts, greedy_params)
    t1 = __import__("time").time()
    print(f"  Done in {t1 - t0:.1f}s")

    results["greedy_solutions"] = [
        out.outputs[0].text for out in greedy_outputs
    ]

    # === Phase 2: Sampled CoT (4 per problem) ===
    sample_params = SamplingParams(
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95,
        n=4,
    )

    print(f"\n[Phase 2] Generating 4 sampled solutions per problem "
          f"({len(problems) * 4} total)...")
    t0 = __import__("time").time()
    sample_outputs = llm.generate(solve_prompts, sample_params)
    t1 = __import__("time").time()
    print(f"  Done in {t1 - t0:.1f}s")

    results["sampled_solutions"] = [
        [out.text for out in outputs.outputs]
        for outputs in sample_outputs
    ]

    # === Phase 3: Judge greedy solutions ===
    judge_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
    )
    greedy_judge_prompts = [
        JUDGE_PROMPT.format(solution=sol)
        for sol in results["greedy_solutions"]
    ]

    print(f"\n[Phase 3] Judging {len(greedy_judge_prompts)} greedy solutions...")
    t0 = __import__("time").time()
    greedy_judge_outputs = llm.generate(greedy_judge_prompts, judge_params)
    t1 = __import__("time").time()
    print(f"  Done in {t1 - t0:.1f}s")

    results["greedy_judgments"] = [
        out.outputs[0].text for out in greedy_judge_outputs
    ]

    # === Phase 4: Judge each sampled solution ===
    # Flatten: build one prompt per (problem, sample) pair
    sampled_judge_prompts = []
    for samples in results["sampled_solutions"]:
        for sol in samples:
            sampled_judge_prompts.append(JUDGE_PROMPT.format(solution=sol))

    print(f"\n[Phase 4] Judging {len(sampled_judge_prompts)} sampled solutions...")
    t0 = __import__("time").time()
    sampled_judge_outputs = llm.generate(sampled_judge_prompts, judge_params)
    t1 = __import__("time").time()
    print(f"  Done in {t1 - t0:.1f}s")

    # Re-nest into [problem][sample] structure
    flat_judgments = [out.outputs[0].text for out in sampled_judge_outputs]
    results["sampled_judgments"] = []
    idx = 0
    for samples in results["sampled_solutions"]:
        batch = flat_judgments[idx : idx + len(samples)]
        results["sampled_judgments"].append(batch)
        idx += len(samples)

    print("\nAll GPU inference complete. Returning results.")
    return _json.dumps(results)


@app.local_entrypoint()
def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer

    # Load MATH-500
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} MATH-500 problems")

    # === Run GPU inference ===
    print("\n" + "=" * 60)
    print("PHASE 1: GPU Inference (solve + judge)")
    print("=" * 60)
    raw = run_judge_inference.remote(json.dumps(problems))
    gpu_results = json.loads(raw)

    greedy_solutions = gpu_results["greedy_solutions"]
    sampled_solutions = gpu_results["sampled_solutions"]
    greedy_judgments = gpu_results["greedy_judgments"]
    sampled_judgments = gpu_results["sampled_judgments"]

    # === Evaluate greedy CoT baseline ===
    print("\n" + "=" * 60)
    print("PHASE 2: Evaluate Results")
    print("=" * 60)

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

    # === Parse greedy judge outputs ===
    greedy_judge_results = []
    judge_agrees_with_gold = 0
    judge_says_yes = 0
    judge_says_no = 0
    n_parseable = 0

    for i, (prob, sol, judgment) in enumerate(
        zip(problems, greedy_solutions, greedy_judgments)
    ):
        parsed = parse_judge_output(judgment)
        actually_correct = greedy_results[i]["correct"]

        if parsed["likely_correct"] is not None:
            n_parseable += 1
            if parsed["likely_correct"]:
                judge_says_yes += 1
            else:
                judge_says_no += 1
            if parsed["likely_correct"] == actually_correct:
                judge_agrees_with_gold += 1

        greedy_judge_results.append({
            "id": prob["id"],
            "score": parsed["score"],
            "likely_correct": parsed["likely_correct"],
            "n_steps_judged": len(parsed["step_judgments"]),
            "n_incorrect_steps": sum(
                1 for _, v in parsed["step_judgments"] if v == "INCORRECT"
            ),
            "actually_correct": actually_correct,
        })

    judge_agreement = judge_agrees_with_gold / max(n_parseable, 1)
    print(f"\nJudge agreement with gold: {judge_agreement:.3f} "
          f"({judge_agrees_with_gold}/{n_parseable})")
    print(f"  Judge says YES: {judge_says_yes}, NO: {judge_says_no}")

    # === Evaluate sampled solutions ===
    # For each problem, compute: majority@4, best-of-4, judge-selected
    majority4_correct = 0
    best4_correct = 0
    judge_selected_correct = 0
    sampled_results = []

    for i, (prob, samples, sample_judges) in enumerate(
        zip(problems, sampled_solutions, sampled_judgments)
    ):
        sample_preds = []
        any_correct = False
        for sol in samples:
            pred = extract_model_answer(sol)
            correct, _ = answers_equivalent(pred, prob["answer"])
            sample_preds.append({"pred": pred, "correct": correct})
            if correct:
                any_correct = True

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
        majority_correct = groups[0][2] if groups else False
        if majority_correct:
            majority4_correct += 1

        # Judge-selected: pick sample with highest judge score
        judge_scores = []
        for jtext in sample_judges:
            parsed = parse_judge_output(jtext)
            judge_scores.append(parsed["score"])

        best_judge_idx = 0
        best_judge_score = -1
        for j_idx, sc in enumerate(judge_scores):
            if sc > best_judge_score:
                best_judge_score = sc
                best_judge_idx = j_idx

        judge_pick_correct = sample_preds[best_judge_idx]["correct"]
        if judge_pick_correct:
            judge_selected_correct += 1

        sampled_results.append({
            "id": prob["id"],
            "level": prob["level"],
            "type": prob["type"],
            "best4_correct": any_correct,
            "majority4_correct": majority_correct,
            "judge_selected_correct": judge_pick_correct,
            "judge_scores": judge_scores,
            "judge_selected_idx": best_judge_idx,
        })

    majority4_acc = majority4_correct / len(problems)
    best4_acc = best4_correct / len(problems)
    judge_selected_acc = judge_selected_correct / len(problems)

    print(f"\nMajority@4: {majority4_acc:.3f} ({majority4_correct}/{len(problems)})")
    print(f"Best-of-4 (oracle): {best4_acc:.3f} ({best4_correct}/{len(problems)})")
    print(f"Judge-selected@4: {judge_selected_acc:.3f} "
          f"({judge_selected_correct}/{len(problems)})")

    # === Judge precision / recall on greedy solutions ===
    # Treat judge "likely_correct: YES" as a positive prediction
    tp = sum(1 for r in greedy_judge_results
             if r["likely_correct"] is True and r["actually_correct"])
    fp = sum(1 for r in greedy_judge_results
             if r["likely_correct"] is True and not r["actually_correct"])
    fn = sum(1 for r in greedy_judge_results
             if r["likely_correct"] is False and r["actually_correct"])
    tn = sum(1 for r in greedy_judge_results
             if r["likely_correct"] is False and not r["actually_correct"])

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"\nJudge as classifier (greedy solutions):")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

    # === Step-level analysis ===
    total_steps_judged = sum(r["n_steps_judged"] for r in greedy_judge_results)
    total_incorrect_flagged = sum(
        r["n_incorrect_steps"] for r in greedy_judge_results
    )
    flagged_rate = total_incorrect_flagged / max(total_steps_judged, 1)
    print(f"\nStep-level: {total_steps_judged} steps judged, "
          f"{total_incorrect_flagged} flagged INCORRECT "
          f"({flagged_rate:.1%})")

    # === By level ===
    print(f"\n{'=' * 60}")
    print("RESULTS BY DIFFICULTY LEVEL")
    print(f"{'=' * 60}")
    by_level = {}
    for lv in [1, 2, 3, 4, 5]:
        lv_probs = [(i, p) for i, p in enumerate(problems) if p["level"] == lv]
        n_lv = len(lv_probs)
        if n_lv == 0:
            continue
        greedy_lv = sum(
            1 for i, _ in lv_probs if greedy_results[i]["correct"]
        ) / n_lv
        maj4_lv = sum(
            1 for i, _ in lv_probs if sampled_results[i]["majority4_correct"]
        ) / n_lv
        best4_lv = sum(
            1 for i, _ in lv_probs if sampled_results[i]["best4_correct"]
        ) / n_lv
        judge_lv = sum(
            1 for i, _ in lv_probs
            if sampled_results[i]["judge_selected_correct"]
        ) / n_lv

        by_level[lv] = {
            "n": n_lv,
            "greedy": greedy_lv,
            "majority4": maj4_lv,
            "best4": best4_lv,
            "judge_selected": judge_lv,
        }
        print(f"  L{lv} (n={n_lv}): greedy={greedy_lv:.3f} "
              f"maj4={maj4_lv:.3f} best4={best4_lv:.3f} "
              f"judge={judge_lv:.3f}")

    # === By subject ===
    print(f"\n{'=' * 60}")
    print("RESULTS BY SUBJECT")
    print(f"{'=' * 60}")
    by_subject = {}
    subjects = sorted(set(p["type"] for p in problems))
    for subj in subjects:
        s_probs = [(i, p) for i, p in enumerate(problems) if p["type"] == subj]
        n_s = len(s_probs)
        if n_s == 0:
            continue
        greedy_s = sum(
            1 for i, _ in s_probs if greedy_results[i]["correct"]
        ) / n_s
        maj4_s = sum(
            1 for i, _ in s_probs if sampled_results[i]["majority4_correct"]
        ) / n_s
        best4_s = sum(
            1 for i, _ in s_probs if sampled_results[i]["best4_correct"]
        ) / n_s
        judge_s = sum(
            1 for i, _ in s_probs
            if sampled_results[i]["judge_selected_correct"]
        ) / n_s

        by_subject[subj] = {
            "n": n_s,
            "greedy": greedy_s,
            "majority4": maj4_s,
            "best4": best4_s,
            "judge_selected": judge_s,
        }
        print(f"  {subj} (n={n_s}): greedy={greedy_s:.3f} "
              f"maj4={maj4_s:.3f} best4={best4_s:.3f} "
              f"judge={judge_s:.3f}")

    # === Score distribution ===
    score_dist = Counter()
    for r in greedy_judge_results:
        if r["score"] >= 0:
            score_dist[r["score"]] += 1
    print(f"\nJudge score distribution (greedy):")
    for s in sorted(score_dist.keys()):
        print(f"  {s}/10: {score_dist[s]}")

    # === Save results ===
    output = {
        "experiment": "exp11_llm_judge_baseline",
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "n_problems": len(problems),
        "accuracy": {
            "greedy_cot": greedy_acc,
            "majority_4": majority4_acc,
            "best_of_4": best4_acc,
            "judge_selected_4": judge_selected_acc,
        },
        "judge_quality": {
            "agreement_with_gold": judge_agreement,
            "n_parseable": n_parseable,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "step_level": {
            "total_steps_judged": total_steps_judged,
            "total_incorrect_flagged": total_incorrect_flagged,
            "flagged_rate": flagged_rate,
        },
        "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
        "by_level": {str(k): v for k, v in by_level.items()},
        "by_subject": by_subject,
        "greedy_judge_results": greedy_judge_results,
        "sampled_results": sampled_results,
    }

    out_path = RESULTS_DIR / "exp11_llm_judge.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # === Summary table ===
    print(f"\n{'=' * 60}")
    print("SUMMARY TABLE")
    print(f"{'=' * 60}")
    print(f"{'Method':<22} {'Overall':>8} "
          f"{'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6} {'L5':>6}")
    print("-" * 64)
    for name, key in [
        ("CoT (greedy)", "greedy"),
        ("Majority@4", "majority4"),
        ("Best-of-4", "best4"),
        ("Judge-selected@4", "judge_selected"),
    ]:
        overall = output["accuracy"][{
            "greedy": "greedy_cot",
            "majority4": "majority_4",
            "best4": "best_of_4",
            "judge_selected": "judge_selected_4",
        }[key]]
        lv_vals = [by_level.get(lv, {}).get(key, 0) for lv in [1, 2, 3, 4, 5]]
        print(f"{name:<22} {overall:>7.1%} "
              f"{lv_vals[0]:>5.1%} {lv_vals[1]:>5.1%} "
              f"{lv_vals[2]:>5.1%} {lv_vals[3]:>5.1%} {lv_vals[4]:>5.1%}")

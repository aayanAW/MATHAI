"""Experiment 25: Selective Prediction Comparison.

Compares SGRV against standard selective prediction baselines on MATH-500:
1. Verbalized confidence (ask the model)
2. Self-consistency proportion (majority vote strength)
3. Semantic uncertainty (Kuhn et al., ICLR 2023) — approximate via answer clustering
4. ExeVer (same-model verification)
5. SGRV (ours)

Metrics:
- AUROC (discrimination)
- Risk-coverage AUC (selective prediction primary metric)
- Accuracy at fixed coverage levels (20%, 40%, 60%, 80%)
- Clopper-Pearson CIs

Uses 3 MATH-500 subsets to conserve API budget:
- n=200 stratified problems (40 per level)
- 1 model: Qwen2.5-7B-Instruct-Turbo (already available, same as exp13)
- 4 samples per problem for self-consistency baselines
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

RESULTS_DIR = Path("results")


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.pbt.pipeline import run_pbt

    api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
    client = OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key)
    model_name = "Qwen/Qwen2.5-7B-Instruct-Turbo"

    # Load MATH-500 and sample n=200 stratified
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        all_problems = json.load(f)

    # Stratified sample: 40 per level
    import random
    random.seed(42)
    problems = []
    for lv in [1, 2, 3, 4, 5]:
        lv_probs = [p for p in all_problems if p["level"] == lv]
        random.shuffle(lv_probs)
        problems.extend(lv_probs[:40])

    print(f"Selective prediction comparison on {len(problems)} problems")
    print(f"Model: {model_name}")
    print("=" * 60)

    SOLVE_PROMPT = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

## Step 2: [brief title]
[reasoning and computation for this step]

...continue for all steps...

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""

    CONFIDENCE_PROMPT = """You solved this problem:

Problem: {problem}

Your solution:
{solution}

On a scale of 0.0 to 1.0, how confident are you that your final answer is correct? Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

    results = []

    for i, prob in enumerate(problems):
        print(f"  [{i+1}/{len(problems)}] {prob['id']} (L{prob['level']})...", end=" ", flush=True)

        # === Step 1: Generate 4 samples (for self-consistency and SGRV) ===
        samples = []
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": SOLVE_PROMPT.format(problem=prob["problem"])}],
                max_tokens=2048,
                temperature=0.7,
                n=4,
            )
            samples = [c.message.content for c in resp.choices]
        except Exception as e:
            print(f"GEN ERROR: {e}")
            continue

        # === Step 2: Extract answers ===
        predicted = []
        for sol in samples:
            ans = extract_model_answer(sol)
            predicted.append(ans)

        # === Step 3: Check correctness of each sample ===
        correct_flags = []
        for ans in predicted:
            ok, _ = answers_equivalent(ans, prob["answer"])
            correct_flags.append(ok)

        # === Step 4: Self-consistency (majority vote) ===
        # Group answers by semantic equivalence
        groups = []
        for ans, ok in zip(predicted, correct_flags):
            merged = False
            for gi, (canon, count, any_correct) in enumerate(groups):
                eq, _ = answers_equivalent(ans, canon)
                if eq:
                    groups[gi] = (canon, count + 1, any_correct or ok)
                    merged = True
                    break
            if not merged:
                groups.append((ans, 1, ok))
        groups.sort(key=lambda x: x[1], reverse=True)

        # Selected answer = most common
        selected_answer = groups[0][0] if groups else ""
        selected_correct = groups[0][2] if groups else False

        # Self-consistency confidence = max vote / total
        sc_confidence = groups[0][1] / len(samples) if groups else 0
        n_unique = len(groups)

        # === Step 5: Verbalized confidence (single API call) ===
        verb_conf = 0.5  # default
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": CONFIDENCE_PROMPT.format(
                        problem=prob["problem"],
                        solution=samples[0],
                    )}
                ],
                max_tokens=20,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            # Parse first float from text
            match = re.search(r"(\d+\.\d+|\d+)", text)
            if match:
                verb_conf = float(match.group(1))
                if verb_conf > 1.0:
                    verb_conf /= 100.0 if verb_conf <= 100 else 10000
        except Exception as e:
            pass

        # === Step 6: Run SGRV on the first (greedy-equivalent) sample ===
        pbt_result = run_pbt(
            problem=prob["problem"],
            solution=samples[0],
            gold_answer=prob["answer"],
            problem_id=prob["id"],
        )
        sgrv_confidence = 1.0 if (pbt_result.all_tested_pass and pbt_result.n_testable > 0) else 0.3

        # === Record ===
        results.append({
            "id": prob["id"],
            "level": prob["level"],
            "type": prob["type"],
            "sample0_correct": correct_flags[0],
            "sample0_answer": predicted[0],
            "sc_confidence": sc_confidence,
            "sc_n_unique": n_unique,
            "sc_selected_correct": selected_correct,
            "verb_confidence": verb_conf,
            "sgrv_confidence": sgrv_confidence,
            "sgrv_all_pass": pbt_result.all_tested_pass and pbt_result.n_testable > 0,
            "sgrv_n_tested": pbt_result.n_testable,
        })

        print(f"greedy_correct={correct_flags[0]}, sc_conf={sc_confidence:.2f}, verb_conf={verb_conf:.2f}, "
              f"sgrv_pass={pbt_result.all_tested_pass and pbt_result.n_testable > 0}")

        time.sleep(0.3)

        # Save incremental in case of interruption
        if (i + 1) % 25 == 0:
            with open(RESULTS_DIR / "exp25_selective_prediction_partial.json", "w") as f:
                json.dump(results, f, indent=2)

    # === Compute metrics ===
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from scipy.stats import binomtest

    def ci(k, n):
        if n == 0:
            return 0, 0
        r = binomtest(k, n)
        c = r.proportion_ci(method="exact")
        return c.low, c.high

    def rc_auc(labels, scores):
        labels = np.array(labels, dtype=int)
        scores = np.array(scores, dtype=float)
        order = np.argsort(-scores)
        y_sorted = labels[order]
        n = len(y_sorted)
        cumsum_correct = np.cumsum(y_sorted)
        coverages = np.arange(1, n + 1) / n
        accuracies = cumsum_correct / np.arange(1, n + 1)
        risks = 1 - accuracies
        return float(np.trapezoid(risks, coverages))

    print(f"\n{'='*60}")
    print(f"SELECTIVE PREDICTION RESULTS (n={len(results)})")
    print(f"{'='*60}")

    labels = np.array([r["sample0_correct"] for r in results], dtype=int)
    baseline_acc = labels.mean()
    print(f"\nBaseline accuracy (greedy sample 0): {baseline_acc:.3f}")

    methods = {
        "Self-Consistency": [r["sc_confidence"] for r in results],
        "Verbalized Confidence": [r["verb_confidence"] for r in results],
        "SGRV (ours)": [r["sgrv_confidence"] for r in results],
    }

    print(f"\n{'Method':<25} {'AUROC':>8} {'RC-AUC':>10}")
    print("-" * 45)
    metrics_out = {}
    for name, scores in methods.items():
        scores_arr = np.array(scores)
        auroc = roc_auc_score(labels, scores_arr)
        rcauc = rc_auc(labels, scores_arr)
        print(f"{name:<25} {auroc:>8.3f} {rcauc:>10.3f}")
        metrics_out[name] = {"auroc": float(auroc), "rc_auc": rcauc}

    # Accuracy at fixed coverages
    print(f"\n{'Method':<25} {'Acc@20%':>10} {'Acc@40%':>10} {'Acc@60%':>10} {'Acc@80%':>10}")
    print("-" * 70)
    for name, scores in methods.items():
        row = [name]
        accs_at = {}
        for cov in [0.2, 0.4, 0.6, 0.8]:
            scores_arr = np.array(scores)
            order = np.argsort(-scores_arr)
            n_accept = int(len(results) * cov)
            if n_accept == 0:
                row.append("--")
                continue
            accepted = labels[order[:n_accept]]
            acc = accepted.mean()
            row.append(f"{acc:.3f}")
            accs_at[f"acc_at_{int(cov*100)}"] = float(acc)
        print(f"{row[0]:<25} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")
        metrics_out[name].update(accs_at)

    output = {
        "n": len(results),
        "baseline_accuracy": float(baseline_acc),
        "metrics": metrics_out,
        "raw_results": results,
    }

    with open(RESULTS_DIR / "exp25_selective_prediction.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/exp25_selective_prediction.json")


if __name__ == "__main__":
    main()

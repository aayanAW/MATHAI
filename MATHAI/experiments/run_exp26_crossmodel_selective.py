"""Experiment 26: Cross-model selective prediction.

Replicates exp25's selective prediction comparison on 3 model families:
- Qwen2.5-7B-Instruct-Turbo (primary, already in exp25)
- Llama-3.3-70B-Instruct-Turbo (cross-family, larger scale)
- DeepSeek-V3 (different architecture, MoE)

For each model:
- Generate solutions on 100 stratified MATH-500 problems (20 per level)
- Compare 3 baselines: verbalized confidence, self-consistency, SGRV
- Report AUROC and risk-coverage AUC

Total: ~600 API calls per model × 3 = ~1800 calls
Estimated cost: ~$80 (DeepSeek-V3 is more expensive than Qwen-7B)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

RESULTS_DIR = Path("results")

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
]


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.eval.answer_check import answers_equivalent, extract_model_answer
    from src.pbt.pipeline import run_pbt

    api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
    # Set per-request timeout so a single hung call doesn't stall the whole run
    client = OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key, timeout=90.0, max_retries=2)

    # Stratified subset: 10 per level = 50 problems per model (budget-conscious)
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        all_problems = json.load(f)

    import random
    random.seed(42)
    problems = []
    for lv in [1, 2, 3, 4, 5]:
        lv_probs = [p for p in all_problems if p["level"] == lv]
        random.shuffle(lv_probs)
        problems.extend(lv_probs[:10])

    print(f"Cross-model selective prediction on {len(problems)} problems", flush=True)
    print(f"Models: {MODELS}", flush=True)
    print("=" * 60, flush=True)

    # Load existing results so we can skip completed models
    existing = {}
    out_path = RESULTS_DIR / "exp26_crossmodel_selective.json"
    if out_path.exists():
        try:
            with open(out_path) as f:
                prior = json.load(f)
            prior_raw = prior.get("raw_results", prior) if isinstance(prior, dict) else prior
            if isinstance(prior_raw, dict):
                for mk, mv in prior_raw.items():
                    if isinstance(mv, list) and len(mv) >= 50:
                        existing[mk] = mv
                        print(f"  Skipping {mk}: already have {len(mv)} results", flush=True)
        except Exception as e:
            print(f"  Could not load existing results: {e}", flush=True)

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

    all_results = dict(existing)

    for model_name in MODELS:
        if model_name in existing:
            continue
        print(f"\n--- {model_name} ---", flush=True)
        results = []

        for i, prob in enumerate(problems):
            try:
                # Generate 4 samples
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": SOLVE_PROMPT.format(problem=prob["problem"])}],
                    max_tokens=2048,
                    temperature=0.7,
                    n=4,
                )
                samples = [c.message.content for c in resp.choices]
            except Exception as e:
                # Fall back to single generation if n>1 not supported
                try:
                    samples = []
                    for _ in range(4):
                        r = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": SOLVE_PROMPT.format(problem=prob["problem"])}],
                            max_tokens=2048,
                            temperature=0.7,
                        )
                        samples.append(r.choices[0].message.content)
                except Exception as e2:
                    print(f"  [{i+1}/{len(problems)}] {prob['id']}: ERROR {e2}", flush=True)
                    continue

            # Extract + check answers
            predicted = [extract_model_answer(s) for s in samples]
            correct = [answers_equivalent(p, prob["answer"])[0] for p in predicted]

            # Self-consistency
            groups = []
            for ans, ok in zip(predicted, correct):
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
            sc_conf = groups[0][1] / len(samples)

            # Verbalized confidence
            verb_conf = 0.5
            try:
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": CONFIDENCE_PROMPT.format(
                        problem=prob["problem"], solution=samples[0])}],
                    max_tokens=20,
                    temperature=0.0,
                )
                text = r.choices[0].message.content.strip()
                m = re.search(r"(\d+\.\d+|\d+)", text)
                if m:
                    verb_conf = float(m.group(1))
                    if verb_conf > 1.0:
                        verb_conf /= 100.0 if verb_conf <= 100 else 10000
            except Exception:
                pass

            # SGRV on first sample
            pbt = run_pbt(
                problem=prob["problem"],
                solution=samples[0],
                gold_answer=prob["answer"],
                problem_id=prob["id"],
            )
            sgrv_conf = 1.0 if (pbt.all_tested_pass and pbt.n_testable > 0) else 0.3

            results.append({
                "id": prob["id"],
                "level": prob["level"],
                "sample0_correct": correct[0],
                "sc_confidence": sc_conf,
                "verb_confidence": verb_conf,
                "sgrv_confidence": sgrv_conf,
                "sgrv_all_pass": pbt.all_tested_pass and pbt.n_testable > 0,
                "n_unique_answers": len(groups),
            })

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(problems)}] processed (acc={sum(r['sample0_correct'] for r in results)/len(results):.2f})", flush=True)
                # Save after each 5 for resilience
                all_results[model_name] = list(results)
                with open(RESULTS_DIR / "exp26_crossmodel_selective.json", "w") as f:
                    json.dump(all_results, f, indent=2)
            time.sleep(0.2)

        all_results[model_name] = results
        # Save incrementally
        with open(RESULTS_DIR / "exp26_crossmodel_selective.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved {model_name}: {len(results)} results", flush=True)

    # === Compute metrics per model ===
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from scipy.stats import binomtest

    def rc_auc(labels, scores):
        labels = np.array(labels, dtype=int)
        scores = np.array(scores, dtype=float)
        order = np.argsort(-scores)
        y_sorted = labels[order]
        n = len(y_sorted)
        cumsum = np.cumsum(y_sorted)
        coverages = np.arange(1, n + 1) / n
        risks = 1 - cumsum / np.arange(1, n + 1)
        return float(np.trapezoid(risks, coverages))

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL SELECTIVE PREDICTION RESULTS")
    print(f"{'='*70}")

    summary = {}
    for model_name, results in all_results.items():
        if not results:
            continue
        print(f"\n--- {model_name.split('/')[-1]} (n={len(results)}) ---")
        labels = np.array([r["sample0_correct"] for r in results], dtype=int)
        baseline_acc = labels.mean()
        print(f"  Baseline accuracy: {baseline_acc:.3f}")

        model_summary = {"n": len(results), "baseline_accuracy": float(baseline_acc), "methods": {}}
        for method, key in [("Self-Consistency", "sc_confidence"),
                             ("Verbalized Conf", "verb_confidence"),
                             ("SGRV (ours)", "sgrv_confidence")]:
            scores = np.array([r[key] for r in results])
            try:
                auroc = roc_auc_score(labels, scores)
            except ValueError:
                auroc = 0.5
            rcauc = rc_auc(labels, scores)
            print(f"  {method:<20} AUROC={auroc:.3f}  RC-AUC={rcauc:.3f}")
            model_summary["methods"][method] = {
                "auroc": float(auroc),
                "rc_auc": rcauc,
            }
        summary[model_name] = model_summary

    output = {"summary": summary, "raw_results": all_results}
    with open(RESULTS_DIR / "exp26_crossmodel_selective.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/exp26_crossmodel_selective.json")


if __name__ == "__main__":
    main()

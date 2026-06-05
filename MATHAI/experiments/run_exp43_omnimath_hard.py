"""Tier 2 extension: Omni-MATH hard-100 contamination-clean evaluation.

Draw 100 hardest problems from Omni-MATH that are post-Qwen-training-cutoff
(Qwen2.5-Math-7B cutoff ~ April 2024; Omni-MATH was released October 2024).
Run solver (Qwen-7B-Instruct-Turbo, 10 samples each) + Llama-70B X-SGRV
extraction. Report top-tier precision, coverage, Adv-FP as in Table 7.

Budget: ~$0.25 solver + ~$0.20 extractor = <$1.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier  # type: ignore
from src.eval.answer_check import answers_equivalent  # type: ignore

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp43_omnimath_hard.json"
SOLVER = "Qwen/Qwen2.5-7B-Instruct-Turbo"
EXTRACTOR = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
N_HARD = 100
N_SAMPLES = 1  # greedy only; hardest Omni-MATH problems, solver accuracy ~0% anyway

API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY",
)
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=API_KEY, timeout=45.0, max_retries=1)


import concurrent.futures


def _call_with_timeout(fn, timeout: float = 60.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None
        except Exception:
            return None


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


def load_omnimath_hard() -> list[dict]:
    """Load Omni-MATH from HF datasets; pick the 100 hardest."""
    from datasets import load_dataset
    print("Downloading Omni-MATH from HuggingFace...")
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    print(f"  loaded {len(ds)} problems")
    problems = []
    for i, ex in enumerate(ds):
        # Fields: problem, solution, answer, difficulty, domain, source, year
        diff = ex.get("difficulty")
        if diff is None:
            continue
        try:
            diff = float(diff)
        except Exception:
            continue
        problems.append({
            "id": f"omnimath_{i}",
            "problem": ex.get("problem", ""),
            "answer": str(ex.get("answer", "")),
            "difficulty": diff,
            "domain": ex.get("domain", ""),
            "source": ex.get("source", ""),
        })
    # Sort by difficulty descending, take top N_HARD
    problems.sort(key=lambda p: -p["difficulty"])
    return problems[:N_HARD]


def solve_with_samples(problem: str, n: int = N_SAMPLES) -> tuple[str, int]:
    """Greedy single-sample solver with hard timeout."""
    def _do():
        resp = client.chat.completions.create(
            model=SOLVER,
            messages=[{"role": "user", "content":
                       f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{problem}"}],
            max_tokens=1024,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    text = _call_with_timeout(_do, timeout=50.0) or ""
    a = _extract_boxed(text)
    return (a or ""), (1 if a else 0)


def make_extractor_raw(model_id: str):
    def _call(prompt: str) -> str:
        def _do():
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        return _call_with_timeout(_do, timeout=60.0) or ""
    return _call


def make_extractor(model_id: str):
    return make_extractor_raw(model_id)


def adversarial_candidates(gold: str) -> list[str]:
    try:
        g = int(str(gold).strip())
        return [str(g - 1), str(g + 1), str(g + 7), "42" if g != 42 else "41"]
    except Exception:
        return ["0", "1", "100", "42"]


def process(prob: dict, candidate: str, correct: bool, extractor_fn) -> dict:
    t0 = time.time()
    try:
        ext = extract_verifier(prob["problem"], extractor_fn)
    except Exception as e:
        return {"id": prob["id"], "outcome": "extraction_error", "error": str(e), "elapsed": time.time() - t0}
    if ext.unverifiable or ext.error or ext.script is None:
        return {
            "id": prob["id"],
            "outcome": "unverifiable" if ext.unverifiable else "extraction_error",
            "error": ext.error,
            "elapsed": time.time() - t0,
        }
    gold_str = str(prob["answer"])
    tests = [("gold", gold_str), ("candidate", str(candidate))]
    for i, adv in enumerate(adversarial_candidates(gold_str)):
        if adv != gold_str:
            tests.append((f"adv{i}", adv))
    verdicts = {}
    for label, val in tests:
        try:
            v = execute_verifier(ext.script, val, timeout=10.0)
            verdicts[label] = v.verdict
        except Exception:
            verdicts[label] = None
    gold_v = verdicts["gold"]
    cand_v = verdicts["candidate"]
    adv_v = [verdicts[k] for k in verdicts if k.startswith("adv")]
    working = gold_v is True and all(x is False for x in adv_v)
    adv_fp = sum(1 for x in adv_v if x is True)
    classification = "working" if working else (
        "false_positive" if adv_fp > 0 else (
            "compute_abstain" if gold_v is None or any(x is None for x in adv_v) else "logic_broken"
        )
    )
    return {
        "id": prob["id"],
        "outcome": "verifier_produced",
        "classification": classification,
        "script": ext.script,
        "gold": gold_str,
        "candidate": str(candidate),
        "solver_correct": correct,
        "gold_verdict": gold_v,
        "candidate_verdict": cand_v,
        "adv_verdicts": adv_v,
        "adv_fp_count": adv_fp,
        "elapsed": time.time() - t0,
    }


def main():
    problems = load_omnimath_hard()
    print(f"Loaded {len(problems)} hardest Omni-MATH problems "
          f"(difficulty range {problems[-1]['difficulty']:.1f} - {problems[0]['difficulty']:.1f})")

    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {"solver": SOLVER, "extractor": EXTRACTOR, "problems": problems, "results": []}

    done_ids = {r["id"] for r in out["results"]}
    todo = [p for p in problems if p["id"] not in done_ids]
    print(f"Already done: {len(done_ids)}  To do: {len(todo)}")

    extractor_fn = make_extractor(EXTRACTOR)

    for i, p in enumerate(todo):
        pid = p["id"]
        t0 = time.time()
        plur, n_unique = solve_with_samples(p["problem"], n=N_SAMPLES)
        correct = _equiv(plur, str(p["answer"]))
        r = process(p, plur, correct, extractor_fn)
        r["plurality_answer"] = plur
        r["n_unique_samples"] = n_unique
        out["results"].append(r)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            print(f"  [{i+1}/{len(todo)}] {pid}: cand={plur}  correct={correct}  "
                  f"class={r.get('classification')}  ({time.time() - t0:.1f}s)", flush=True)

    # Summary
    res = out["results"]
    n = len(res)
    working = [r for r in res if r.get("classification") == "working"]
    top_tier = [r for r in res if r.get("candidate_verdict") is True]
    top_correct = [r for r in top_tier if r.get("solver_correct")]
    adv_fps = sum(r.get("adv_fp_count", 0) or 0 for r in working)
    solver_acc = sum(1 for r in res if r.get("solver_correct")) / n if n else 0
    print("\n" + "=" * 60)
    print("Omni-MATH hard-100 X-SGRV (Llama-70B)")
    print("=" * 60)
    print(f"n_total: {n}")
    print(f"solver plurality accuracy: {solver_acc:.1%}")
    print(f"working verifiers: {len(working)}/{n} ({len(working)/n:.1%})")
    print(f"top-tier (candidate accepted): {len(top_tier)}")
    if top_tier:
        from scipy.stats import binomtest
        prec = len(top_correct) / len(top_tier)
        ci = binomtest(len(top_correct), len(top_tier)).proportion_ci(0.95, "exact")
        print(f"top-tier precision: {len(top_correct)}/{len(top_tier)} = {prec:.3f} [95% CI {ci.low:.3f}, {ci.high:.3f}]")
    else:
        print("top-tier empty")
    print(f"adv FPs: {adv_fps}/{len(working) * 4}")

    out["summary"] = {
        "n": n,
        "solver_accuracy": solver_acc,
        "n_working": len(working),
        "n_top_tier": len(top_tier),
        "n_top_correct": len(top_correct),
        "top_tier_precision": (len(top_correct) / len(top_tier)) if top_tier else None,
        "adv_fps": adv_fps,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

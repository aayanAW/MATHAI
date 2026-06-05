"""Tier 2: MATH-175 -> MATH-500 scale-up.

For the 325 MATH-500 problems NOT in the stratified n=175 sample:
1. Solve with Qwen2.5-7B-Instruct-Turbo greedy (one call each).
2. Extract a Llama-3.3-70B verifier script.
3. Extract a DeepSeek-V3 verifier script.
4. Execute each verifier against gold, solver candidate, and 4 adversarial
   gold-relative probes (±1, +7, fixed).
5. Save all 500 rows (175 from exp31/32 merged with the new 325).

Budget: ~$2 (~325 × (solver + Llama + DeepSeek calls), no bulk tokens).
Time: ~1-2 hours wall time sequentially.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import concurrent.futures

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier  # type: ignore
from src.eval.answer_check import answers_equivalent  # type: ignore


def _call_with_timeout(fn, timeout: float = 60.0):
    """Run a no-arg callable with a hard timeout. Returns None on timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None
        except Exception:
            return None

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT_LLAMA = RESULTS / "exp37_xsgrv_math500_llama70b.json"
OUT_DEEPSEEK = RESULTS / "exp37_xsgrv_math500_deepseek.json"
SOLVER = "Qwen/Qwen2.5-7B-Instruct-Turbo"
LLAMA = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEEPSEEK = "deepseek-ai/DeepSeek-V3"

API_KEY = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=API_KEY, timeout=60.0, max_retries=1)


def _equiv(a, b):
    r = answers_equivalent(a, b)
    return bool(r[0]) if isinstance(r, tuple) else bool(r)


def _extract_boxed(text: str) -> str:
    """Extract content of the last \\boxed{...} with balanced braces."""
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


def solve_candidate(problem: str) -> str:
    def _do():
        resp = client.chat.completions.create(
            model=SOLVER,
            messages=[{"role": "user", "content":
                       f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\n{problem}"}],
            max_tokens=1024,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    text = _call_with_timeout(_do, timeout=60.0) or ""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def make_extractor(model_id: str):
    def _call(prompt):
        def _do():
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
            return resp.choices[0].message.content
        return _call_with_timeout(_do, timeout=90.0) or ""
    return _call


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
        return {"id": prob["id"], "outcome": "unverifiable" if ext.unverifiable else "extraction_error",
                "error": ext.error, "elapsed": time.time() - t0}
    gold_str = str(prob["answer"])
    tests = [("gold", gold_str), ("candidate", str(candidate))]
    for i, adv in enumerate(adversarial_candidates(gold_str)):
        if adv != gold_str:
            tests.append((f"adv{i}", adv))
    verdicts = {}
    for label, val in tests:
        try:
            v = execute_verifier(ext.script, val, timeout=10.0)
            verdicts[label] = {"value": val, "verdict": v.verdict, "error": v.error}
        except Exception as e:
            verdicts[label] = {"value": val, "verdict": None, "error": str(e)}
    gold_v = verdicts["gold"]["verdict"]
    cand_v = verdicts["candidate"]["verdict"]
    adv_v = [verdicts[k]["verdict"] for k in verdicts if k.startswith("adv")]
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
    # Load all 500
    with open(RESULTS / "math_test_sample_500.json") as f:
        all_math = json.load(f)
    # Exclude the 175 in exp25 (already done in exp31/exp32)
    with open(RESULTS / "exp25_selective_prediction.json") as f:
        exp25_ids = {r["id"] for r in json.load(f)}
    missing = [p for p in all_math if p["id"] not in exp25_ids]
    print(f"To process: {len(missing)} new problems out of {len(all_math)} total")

    # Load existing exp37 (resume if present)
    llama_results = []
    deepseek_results = []
    if OUT_LLAMA.exists():
        llama_results = json.load(open(OUT_LLAMA)).get("results", [])
    if OUT_DEEPSEEK.exists():
        deepseek_results = json.load(open(OUT_DEEPSEEK)).get("results", [])

    llama_done = {r["id"] for r in llama_results}
    deepseek_done = {r["id"] for r in deepseek_results}

    # Solve each missing problem once, then run both extractors
    llama_fn = make_extractor(LLAMA)
    deepseek_fn = make_extractor(DEEPSEEK)

    candidate_cache_path = RESULTS / "exp37_candidates_cache.json"
    candidate_cache = {}
    if candidate_cache_path.exists():
        candidate_cache = json.load(open(candidate_cache_path))

    for i, prob in enumerate(missing):
        pid = prob["id"]
        need_llama = pid not in llama_done
        need_ds = pid not in deepseek_done
        if not need_llama and not need_ds:
            continue

        # Solve if not cached
        if pid in candidate_cache:
            cand = candidate_cache[pid]["candidate"]
            correct = candidate_cache[pid]["correct"]
        else:
            try:
                cand = solve_candidate(prob["problem"])
            except Exception as e:
                print(f"  [{i+1}/{len(missing)}] {pid}: SOLVER ERR {type(e).__name__}")
                continue
            correct = _equiv(cand, str(prob["answer"]))
            candidate_cache[pid] = {"candidate": cand, "correct": correct}
            with open(candidate_cache_path, "w") as f:
                json.dump(candidate_cache, f, indent=2, default=str)

        if need_llama:
            r = process(prob, cand, correct, llama_fn)
            llama_results.append(r)
            with open(OUT_LLAMA, "w") as f:
                json.dump({"extractor": LLAMA, "results": llama_results}, f, indent=2, default=str)

        if need_ds:
            r = process(prob, cand, correct, deepseek_fn)
            deepseek_results.append(r)
            with open(OUT_DEEPSEEK, "w") as f:
                json.dump({"extractor": DEEPSEEK, "results": deepseek_results}, f, indent=2, default=str)

        if (i + 1) % 10 == 0 or i == len(missing) - 1:
            print(f"  [{i+1}/{len(missing)}] {pid}: cand={cand}, correct={correct}", flush=True)


if __name__ == "__main__":
    main()

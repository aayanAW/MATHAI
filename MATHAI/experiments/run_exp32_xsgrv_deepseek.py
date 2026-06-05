"""Experiment 32: X-SGRV with DeepSeek-V3 as a second extractor.

Mirrors exp30/31 structure but uses DeepSeek-V3 instead of Llama-3.3-70B.
Runs on the same 175 MATH-500 problems (from exp25) + 30 AIME 2025 problems.
DeepSeek-V3 is also cross-family relative to the Qwen-7B solver.

The point is NOT that DeepSeek-V3 is better than Llama-70B — it is that a
second independent cross-family extractor enables a consensus mechanism:
we only trust a verifier when BOTH Llama-70B and DeepSeek-V3 produce
working verifiers that agree on the candidate.

Consensus computation happens in analysis/xsgrv_consensus.py (separate script),
which reads both exp31/exp30 (Llama) JSONs and exp32 (DeepSeek) JSONs.

Known issue: DeepSeek-V3 via Together occasionally SSL-hangs (observed in
exp26). Mitigation: `timeout=120, max_retries=1` per API call. If a single
problem exceeds ~2 min in the runner, kill -9 the process; the incremental
per-problem save preserves progress.

Budget: ~$3 (DeepSeek is ~4× more expensive per token than Llama on Together)
Time:   ~60 min wall clock at ~15s/problem
"""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier

EXTRACTOR_MODEL = "deepseek-ai/DeepSeek-V3"
RESULTS_DIR = Path(__file__).parent.parent / "results"

api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=1,
)


def make_extractor():
    def _call(prompt):
        resp = client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content
    return _call


def adversarial_candidates(gold_answer: str) -> list[str]:
    """Generate 4 adversarial wrong candidates near the gold answer."""
    try:
        g = int(str(gold_answer).strip())
        return [str(g - 1), str(g + 1), str(g + 7), "42" if g != 42 else "41"]
    except Exception:
        return ["0", "1", "100", "42"]


def process_problem(prob: dict, candidate: str, solver_correct: bool, gold: str, extractor_fn) -> dict:
    t0 = time.time()
    ext = extract_verifier(prob["problem"], extractor_fn)
    ext_time = time.time() - t0

    if ext.unverifiable:
        return {"id": prob["id"], "outcome": "unverifiable", "script": None, "elapsed": ext_time}
    if ext.error or ext.script is None:
        return {"id": prob["id"], "outcome": "extraction_error", "script": None,
                "error": ext.error, "elapsed": ext_time}

    gold_str = str(gold)
    tests = [("gold", gold_str), ("candidate", str(candidate))]
    for i, adv in enumerate(adversarial_candidates(gold)):
        if adv != gold_str:
            tests.append((f"adv{i}", adv))

    test_results = {}
    for label, val in tests:
        ver = execute_verifier(ext.script, val, timeout=10.0)
        test_results[label] = {
            "value": val,
            "verdict": ver.verdict,
            "error": ver.error,
            "exec_time": ver.execution_time,
        }

    gold_verdict = test_results["gold"]["verdict"]
    cand_verdict = test_results["candidate"]["verdict"]
    adv_verdicts = [test_results[k]["verdict"] for k in test_results if k.startswith("adv")]

    working = gold_verdict is True and all(v is False for v in adv_verdicts)
    adv_fp = sum(1 for v in adv_verdicts if v is True)
    compute_abstain = gold_verdict is None or any(v is None for v in adv_verdicts)
    logic_broken = gold_verdict is False and adv_fp == 0

    if working:
        classification = "working"
    elif adv_fp > 0:
        classification = "false_positive"
    elif compute_abstain:
        classification = "compute_abstain"
    elif logic_broken:
        classification = "logic_broken"
    else:
        classification = "other"

    return {
        "id": prob["id"],
        "outcome": "verifier_produced",
        "classification": classification,
        "script_chars": len(ext.script),
        "script": ext.script,
        "gold": gold_str,
        "candidate": str(candidate),
        "solver_correct": solver_correct,
        "gold_verdict": gold_verdict,
        "candidate_verdict": cand_verdict,
        "adv_verdicts": adv_verdicts,
        "adv_fp_count": adv_fp,
        "elapsed": time.time() - t0,
    }


def run_condition(problems: list[dict], candidate_fn, out_path: Path, label: str):
    results: list[dict] = []
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            if "results" in existing:
                results = existing["results"]
                print(f"Resuming from {len(results)} existing {label} results")
        except Exception as e:
            print(f"Warning: could not load {out_path}: {e}")

    done_ids = {r["id"] for r in results}
    to_do = [p for p in problems if p["id"] not in done_ids]
    print(f"\n=== {label} ({EXTRACTOR_MODEL}) ===")
    print(f"  Already done: {len(done_ids)}  To do: {len(to_do)}", flush=True)

    extractor_fn = make_extractor()
    for i, prob in enumerate(to_do):
        candidate, solver_correct, gold = candidate_fn(prob)
        r = process_problem(prob, candidate, solver_correct, gold, extractor_fn)
        results.append(r)

        if r["outcome"] == "verifier_produced":
            print(
                f"  [{len(results)}/{len(problems)}] {prob['id']}: {r['classification']} "
                f"(gold={r['gold_verdict']}, cand={r['candidate_verdict']}, "
                f"adv_fp={r['adv_fp_count']}, {r['elapsed']:.1f}s)",
                flush=True,
            )
        else:
            print(f"  [{len(results)}/{len(problems)}] {prob['id']}: {r['outcome']} ({r['elapsed']:.1f}s)",
                  flush=True)

        with open(out_path, "w") as f:
            json.dump({"extractor": EXTRACTOR_MODEL, "results": results}, f, indent=2, default=str)

    return results


def summarize(results: list[dict], label: str) -> None:
    n = len(results)
    produced = sum(1 for r in results if r.get("outcome") == "verifier_produced")
    working = sum(1 for r in results if r.get("classification") == "working")
    fp = sum(1 for r in results if r.get("classification") == "false_positive")
    top_tier = [r for r in results if r.get("candidate_verdict") is True]
    top_correct = sum(1 for r in top_tier if r.get("solver_correct"))
    print(f"\n{label}  n={n}  produced={produced}  working={working}  fp={fp}  "
          f"top_tier={len(top_tier)}"
          f"{'' if not top_tier else f' precision={top_correct}/{len(top_tier)}={top_correct/len(top_tier):.4f}'}")


def main():
    # ------- AIME 2025 setup (same as exp30) -------
    aime = json.load(open(RESULTS_DIR / "aime_2025.json"))
    exp27 = json.load(open(RESULTS_DIR / "exp27_aime2025_contamination.json"))
    exp27_by_id = {r["id"]: r for r in exp27["results"]}

    def aime_candidate(prob):
        er = exp27_by_id[prob["id"]]
        return er["sample0_answer"], er["sample0_correct"], prob["answer"]

    # ------- MATH-500 n=175 setup (same as exp31) -------
    with open(RESULTS_DIR / "exp25_selective_prediction.json") as f:
        exp25 = json.load(f)
    exp25_by_id = {r["id"]: r for r in exp25}
    with open(RESULTS_DIR / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    math_by_id = {p["id"]: p for p in math_all}
    all_ids = [r["id"] for r in exp25]
    math_175_probs = [math_by_id[i] for i in all_ids if i in math_by_id]
    assert len(math_175_probs) == 175, f"Expected 175, got {len(math_175_probs)}"

    def math_candidate(prob):
        er = exp25_by_id[prob["id"]]
        return er["sample0_answer"], er["sample0_correct"], prob["answer"]

    aime_out = RESULTS_DIR / "exp32_aime_deepseek.json"
    math_out = RESULTS_DIR / "exp32_math175_deepseek.json"

    # Run AIME first (smaller, cheaper — fail fast if DeepSeek hangs)
    r_aime = run_condition(aime, aime_candidate, aime_out, "AIME 2025 × DeepSeek-V3")
    summarize(r_aime, "AIME 2025 × DeepSeek-V3")

    r_math = run_condition(math_175_probs, math_candidate, math_out, "MATH-500 n=175 × DeepSeek-V3")
    summarize(r_math, "MATH-500 n=175 × DeepSeek-V3")


if __name__ == "__main__":
    main()

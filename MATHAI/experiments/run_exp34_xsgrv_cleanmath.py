"""Experiment 34: X-SGRV on the contamination-clean MathArena combo (n=125).

Replaces the handoff's "LiveMathBench" task with MathArena-sourced contests
(HMMT Feb 2025, BRUMO 2025, SMT 2025, APEX 2025). LiveMathBench is gated;
MathArena is open and covers the same post-Qwen-cutoff contamination-clean
range. Total problems: 125.

Pipeline (3 phases, incremental saves after every problem):
  Phase 1 — Generate Qwen2.5-7B-Instruct-Turbo solver answers (T=0.0)
            Cached to: results/cleanmath_solver_qwen7b.json
  Phase 2 — Run X-SGRV with Llama-3.3-70B cross-family extractor
            Saved to: results/exp34_cleanmath_llama70b.json
  Phase 3 — Summary (coverage, working rate, top-tier precision, adversarial FP)

Budget:
  Solver: 125 × 1k tok × $0.20/M = ~$0.03
  Extractor: 125 × 2k tok × $0.88/M = ~$0.22
  Total: ~$0.25
Time: ~20 min wall clock
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier
from src.eval.answer_check import answers_equivalent


def _equiv(a: str, b: str) -> bool:
    """answers_equivalent returns (bool, reason) — unpack carefully."""
    try:
        result = answers_equivalent(str(a), str(b))
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)
    except Exception:
        return str(a).strip() == str(b).strip()

SOLVER_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
EXTRACTOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

RESULTS_DIR = Path(__file__).parent.parent / "results"
INPUT_PATH = RESULTS_DIR / "cleanmath_combo.json"
SOLVER_PATH = RESULTS_DIR / "cleanmath_solver_qwen7b.json"
OUT_PATH = RESULTS_DIR / "exp34_cleanmath_llama70b.json"

api_key = os.environ.get("TOGETHER_API_KEY", "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY")
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=2,
)


SOLVER_PROMPT = """Solve the following math problem step by step. At the end of your solution, write your final answer inside \\boxed{{}}.

Problem: {problem}

Solution:"""


def extract_boxed(text: str) -> str | None:
    """Extract the content of the LAST \\boxed{...} in the text."""
    # Find all \boxed{...} matches, matching nested braces
    matches = []
    i = 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 1
        j = idx + len("\\boxed{")
        start = j
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[start: j - 1].strip())
        i = j
    return matches[-1] if matches else None


def solve_problem(problem_text: str) -> tuple[str, str]:
    """Return (raw_response, extracted_boxed_answer)."""
    resp = client.chat.completions.create(
        model=SOLVER_MODEL,
        messages=[{"role": "user", "content": SOLVER_PROMPT.format(problem=problem_text)}],
        max_tokens=1536,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    boxed = extract_boxed(raw) or ""
    return raw, boxed


def phase1_generate_solver_answers(problems: list[dict]) -> dict[str, dict]:
    """Generate (or resume) Qwen-7B solver answers for each problem."""
    solver_cache: dict[str, dict] = {}
    if SOLVER_PATH.exists():
        try:
            solver_cache = json.load(open(SOLVER_PATH))
            print(f"[Phase 1] Resumed {len(solver_cache)} cached solver answers")
        except Exception:
            pass

    to_solve = [p for p in problems if p["id"] not in solver_cache]
    print(f"[Phase 1] Solving {len(to_solve)} new problems with {SOLVER_MODEL}")

    for i, prob in enumerate(to_solve):
        t0 = time.time()
        try:
            raw, boxed = solve_problem(prob["problem"])
            gold = str(prob["answer"]).strip()
            correct = _equiv(boxed, gold)
            solver_cache[prob["id"]] = {
                "id": prob["id"],
                "raw": raw,
                "boxed_answer": boxed,
                "gold": gold,
                "correct": correct,
                "elapsed": time.time() - t0,
            }
            marker = "✓" if correct else "✗"
            print(f"  [{len(solver_cache)}/{len(problems)}] {prob['id']}: {marker} "
                  f"boxed={boxed[:30]!r} gold={gold[:30]!r} ({time.time() - t0:.1f}s)",
                  flush=True)
        except Exception as e:
            print(f"  [{len(solver_cache)+1}/{len(problems)}] {prob['id']}: ERROR {type(e).__name__}: {e}",
                  flush=True)
            solver_cache[prob["id"]] = {
                "id": prob["id"],
                "raw": None,
                "boxed_answer": "",
                "gold": str(prob["answer"]).strip(),
                "correct": False,
                "error": f"{type(e).__name__}: {e}",
                "elapsed": time.time() - t0,
            }

        with open(SOLVER_PATH, "w") as f:
            json.dump(solver_cache, f, indent=2, default=str)

    correct_count = sum(1 for v in solver_cache.values() if v.get("correct"))
    print(f"[Phase 1] Done. Solver accuracy: {correct_count}/{len(solver_cache)} = {correct_count/len(solver_cache)*100:.1f}%")
    return solver_cache


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
        test_results[label] = {"value": val, "verdict": ver.verdict, "error": ver.error,
                               "exec_time": ver.execution_time}

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


def phase2_run_xsgrv(problems: list[dict], solver_cache: dict[str, dict]) -> list[dict]:
    results: list[dict] = []
    if OUT_PATH.exists():
        try:
            existing = json.load(open(OUT_PATH))
            if "results" in existing:
                results = existing["results"]
                print(f"[Phase 2] Resumed {len(results)} existing X-SGRV results")
        except Exception:
            pass

    done_ids = {r["id"] for r in results}
    to_do = [p for p in problems if p["id"] not in done_ids]
    print(f"[Phase 2] X-SGRV on {len(to_do)} new problems with {EXTRACTOR_MODEL}")

    extractor_fn = make_extractor()
    for prob in to_do:
        sc = solver_cache.get(prob["id"])
        if sc is None:
            print(f"  [{len(results)}/{len(problems)}] {prob['id']}: SKIP (no solver answer)")
            continue
        candidate = sc["boxed_answer"]
        gold = prob["answer"]
        # Always re-compute correctness from boxed + gold (ignore possibly stale
        # cached `correct` field that may have used string equality).
        solver_correct = _equiv(candidate, gold)

        r = process_problem(prob, candidate, solver_correct, gold, extractor_fn)
        r["contest"] = prob.get("type")
        results.append(r)

        if r["outcome"] == "verifier_produced":
            print(f"  [{len(results)}/{len(problems)}] {prob['id']}: {r['classification']} "
                  f"(gold={r['gold_verdict']}, cand={r['candidate_verdict']}, "
                  f"adv_fp={r['adv_fp_count']}, {r['elapsed']:.1f}s)", flush=True)
        else:
            print(f"  [{len(results)}/{len(problems)}] {prob['id']}: {r['outcome']} ({r['elapsed']:.1f}s)",
                  flush=True)

        with open(OUT_PATH, "w") as f:
            json.dump({"extractor": EXTRACTOR_MODEL, "results": results}, f, indent=2, default=str)

    return results


def summarize(results: list[dict], label: str) -> None:
    n = len(results)
    unver = sum(1 for r in results if r.get("outcome") == "unverifiable")
    ext_err = sum(1 for r in results if r.get("outcome") == "extraction_error")
    produced = sum(1 for r in results if r.get("outcome") == "verifier_produced")
    working = sum(1 for r in results if r.get("classification") == "working")
    compute_abstain = sum(1 for r in results if r.get("classification") == "compute_abstain")
    logic_broken = sum(1 for r in results if r.get("classification") == "logic_broken")
    fp = sum(1 for r in results if r.get("classification") == "false_positive")

    total_adv = 0
    adv_fp_total = 0
    for r in results:
        if r.get("outcome") != "verifier_produced":
            continue
        for v in (r.get("adv_verdicts") or []):
            if v is not None:
                total_adv += 1
                if v is True:
                    adv_fp_total += 1

    top_tier = [r for r in results if r.get("candidate_verdict") is True]
    top_correct = sum(1 for r in top_tier if r.get("solver_correct"))

    solver_correct = sum(1 for r in results if r.get("solver_correct"))
    print(f"\n{'=' * 60}\n{label}  (n={n})\n{'=' * 60}")
    print(f"Solver baseline accuracy: {solver_correct}/{n} = {solver_correct/n*100:.1f}%")
    print(f"Extraction: UNVERIFIABLE={unver}  error={ext_err}  produced={produced}")
    print(f"Classification: working={working} ({100*working/n:.1f}%)  "
          f"broken={logic_broken}  abstain={compute_abstain}  FP={fp}")
    if total_adv:
        print(f"Adv FP rate: {adv_fp_total}/{total_adv} = {adv_fp_total/total_adv*100:.2f}%")
    print(f"Top tier: {len(top_tier)}/{n} = {len(top_tier)/n*100:.1f}%")
    if top_tier:
        print(f"Top-tier precision: {top_correct}/{len(top_tier)} = {top_correct/len(top_tier):.4f}")
        from scipy.stats import binomtest
        res = binomtest(top_correct, len(top_tier))
        ci = res.proportion_ci(confidence_level=0.95, method="exact")
        print(f"Top-tier precision 95% CI: [{ci.low:.4f}, {ci.high:.4f}]")

    # Per-contest breakdown
    from collections import defaultdict
    per_contest = defaultdict(lambda: {"n": 0, "working": 0, "top_tier": 0, "top_correct": 0})
    for r in results:
        c = r.get("contest", "?")
        per_contest[c]["n"] += 1
        if r.get("classification") == "working":
            per_contest[c]["working"] += 1
        if r.get("candidate_verdict") is True:
            per_contest[c]["top_tier"] += 1
            if r.get("solver_correct"):
                per_contest[c]["top_correct"] += 1
    print("\nPer-contest breakdown:")
    for c, s in sorted(per_contest.items()):
        prec = f"{s['top_correct']}/{s['top_tier']}" if s["top_tier"] else "n/a"
        print(f"  {c:20s} n={s['n']:3d} working={s['working']:3d} top_tier={s['top_tier']:3d} prec={prec}")


def main() -> None:
    with open(INPUT_PATH) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems from {INPUT_PATH.name}")

    solver_cache = phase1_generate_solver_answers(problems)
    results = phase2_run_xsgrv(problems, solver_cache)
    summarize(results, f"CleanMath Combo × {EXTRACTOR_MODEL}")
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()

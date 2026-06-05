"""OpenAI gpt-oss-120b as a third X-SGRV extractor.

Enables 3-way consensus across Llama-3.3-70B (Meta) + DeepSeek-V3 (DeepSeek AI)
+ gpt-oss-120b (OpenAI open-weights). This is a genuinely cross-family test
across the three largest LLM labs.

For each of 330 problems (MATH-175 + AIME + CleanMath), we:
  1. Have gpt-oss-120b emit a SymPy verify(answer) function from the problem
     statement using the same prompt as the Llama/DeepSeek extractors.
  2. Execute the script against the solver's candidate, the gold answer, and
     4 adversarial probes.
  3. Classify the verifier as working / false_positive / compute_abstain /
     logic_broken and compute top-tier precision.

Budget: ~330 problems * ~2k tokens * $0.15/M = ~$1. No extra compute.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier  # type: ignore

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp46_gptoss_extractor.json"
EXTRACTOR = "openai/gpt-oss-120b"

API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "tgp_v1_NjxsBA_J8FWOkIY0JJngkB9YKYbeyG0exLQRj8p37kY",
)
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=API_KEY, timeout=120.0, max_retries=1)


def _call_timeout(fn, timeout=110.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return None


def make_extractor_fn():
    def _call(prompt):
        def _do():
            resp = client.chat.completions.create(
                model=EXTRACTOR,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        return _call_timeout(_do, timeout=110.0) or ""
    return _call


def adversarial_candidates(gold):
    try:
        g = int(str(gold).strip())
        return [str(g - 1), str(g + 1), str(g + 7), "42" if g != 42 else "41"]
    except Exception:
        return ["0", "1", "100", "42"]


def load_benchmarks_with_candidates():
    """Return list of {id, bench, problem, gold, candidate, solver_correct}."""
    out = []
    # MATH-175 (use exp25 candidates)
    with open(RESULTS / "exp25_selective_prediction.json") as f:
        exp25 = json.load(f)
    exp25_by_id = {r["id"]: r for r in exp25}
    with open(RESULTS / "math_test_sample_500.json") as f:
        math_all = json.load(f)
    for p in math_all:
        pid = p["id"]
        if pid in exp25_by_id:
            e = exp25_by_id[pid]
            out.append({
                "id": pid, "bench": "math175", "problem": p["problem"],
                "gold": str(p["answer"]),
                "candidate": e["sample0_answer"],
                "solver_correct": e["sample0_correct"],
            })

    # AIME 2025 — use exp27 solver candidates if available
    try:
        exp27 = json.load(open(RESULTS / "exp27_aime2025_contamination.json"))
        aime_cands = {r["id"]: {"candidate": r.get("candidate") or r.get("sample0_answer"),
                                "correct": r.get("sample0_correct") or r.get("solver_correct")}
                      for r in exp27.get("results", exp27)}
    except Exception:
        aime_cands = {}
    with open(RESULTS / "aime_2025.json") as f:
        aime = json.load(f)
    for p in aime:
        pid = p["id"]
        c = aime_cands.get(pid, {})
        out.append({
            "id": pid, "bench": "aime", "problem": p["problem"],
            "gold": str(p["answer"]),
            "candidate": c.get("candidate", ""),
            "solver_correct": c.get("correct", False),
        })

    # CleanMath — use exp34 solver candidates
    try:
        exp34 = json.load(open(RESULTS / "exp34_cleanmath_llama70b.json"))
        cm_cands = {r["id"]: {"candidate": r.get("candidate"), "correct": r.get("solver_correct")}
                    for r in exp34.get("results", [])}
    except Exception:
        cm_cands = {}
    with open(RESULTS / "cleanmath_combo.json") as f:
        cm = json.load(f)
    for p in cm:
        pid = p["id"]
        c = cm_cands.get(pid, {})
        out.append({
            "id": pid, "bench": "cleanmath", "problem": p["problem"],
            "gold": str(p["answer"]),
            "candidate": c.get("candidate", ""),
            "solver_correct": c.get("correct", False),
        })
    return out


def process(p, extractor_fn):
    t0 = time.time()
    try:
        ext = extract_verifier(p["problem"], extractor_fn)
    except Exception as e:
        return {"id": p["id"], "bench": p["bench"], "outcome": "extraction_error",
                "error": str(e)[:200], "elapsed": time.time() - t0}
    if ext.unverifiable:
        return {"id": p["id"], "bench": p["bench"], "outcome": "unverifiable",
                "elapsed": time.time() - t0}
    if ext.error or ext.script is None:
        return {"id": p["id"], "bench": p["bench"], "outcome": "extraction_error",
                "error": ext.error, "elapsed": time.time() - t0}

    gold_str = str(p["gold"])
    cand = str(p.get("candidate") or "")
    tests = [("gold", gold_str), ("candidate", cand)]
    for i, adv in enumerate(adversarial_candidates(gold_str)):
        if adv != gold_str:
            tests.append((f"adv{i}", adv))

    verdicts = {}
    for label, val in tests:
        try:
            verdicts[label] = execute_verifier(ext.script, val, timeout=10.0).verdict
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
        "id": p["id"], "bench": p["bench"], "outcome": "verifier_produced",
        "classification": classification,
        "script": ext.script,
        "gold": gold_str, "candidate": cand, "solver_correct": p["solver_correct"],
        "gold_verdict": gold_v, "candidate_verdict": cand_v,
        "adv_verdicts": adv_v, "adv_fp_count": adv_fp,
        "elapsed": time.time() - t0,
    }


def main():
    if OUT.exists():
        out = json.load(open(OUT))
    else:
        out = {"extractor": EXTRACTOR, "results": []}
    done = {r["id"] for r in out["results"]}

    probs = load_benchmarks_with_candidates()
    todo = [p for p in probs if p["id"] not in done]
    print(f"Total: {len(probs)}, todo: {len(todo)}", flush=True)

    ext_fn = make_extractor_fn()
    for i, p in enumerate(todo):
        r = process(p, ext_fn)
        out["results"].append(r)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=str)
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            print(f"  [{i+1}/{len(todo)}] {r['bench']}/{r['id']}: {r.get('outcome','?')}/{r.get('classification','?')} ({r['elapsed']:.1f}s)", flush=True)

    # Summary
    from scipy.stats import binomtest
    print("\n" + "=" * 70)
    print(f"gpt-oss-120b extractor summary")
    print("=" * 70)
    for bench in ["math175", "aime", "cleanmath"]:
        rows = [r for r in out["results"] if r.get("bench") == bench]
        if not rows:
            continue
        n = len(rows)
        working = [r for r in rows if r.get("classification") == "working"]
        tier = [r for r in rows if r.get("candidate_verdict") is True]
        tier_correct = sum(1 for r in tier if r.get("solver_correct"))
        adv_fp = sum(r.get("adv_fp_count", 0) or 0 for r in rows)
        print(f"\n[{bench}] n={n}")
        print(f"  working={len(working)} ({len(working)/n:.1%}), top-tier={len(tier)}, adv_fp={adv_fp}")
        if tier:
            ci = binomtest(tier_correct, len(tier)).proportion_ci(0.95, "exact")
            print(f"  precision: {tier_correct}/{len(tier)} = {tier_correct/len(tier):.3f} [{ci.low:.3f}, {ci.high:.3f}]")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

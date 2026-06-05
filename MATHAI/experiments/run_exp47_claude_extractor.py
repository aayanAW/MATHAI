"""Anthropic Claude Sonnet 4.6 as a fourth X-SGRV extractor.

Enables 4-way cross-family consensus:
  Meta (Llama-3.3-70B) + DeepSeek (V3) + OpenAI (gpt-oss-120b) + Anthropic (Claude Sonnet 4.6).

For each of 330 problems (MATH-175 + AIME + CleanMath) we ask Claude to emit
a Python SymPy verify(answer) function from the problem statement and execute
it against the solver's candidate, the gold answer, and 4 adversarial probes.

Budget: ~330 problems * ~3k tokens combined * ($3/M in + $15/M out) / 2 ~ $10-15.
Reads ANTHROPIC_API_KEY from env.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.xsgrv.extractor import extract_verifier, execute_verifier  # type: ignore

RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
OUT = RESULTS / "exp47_claude_extractor.json"
EXTRACTOR = "claude-sonnet-4-6"

if "ANTHROPIC_API_KEY" not in os.environ:
    raise SystemExit("ERROR: set ANTHROPIC_API_KEY in env before running")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def make_extractor_fn():
    def _call(prompt):
        try:
            r = client.messages.create(
                model=EXTRACTOR,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in r.content if hasattr(b, "text"))
        except Exception as e:
            print(f"  [claude err] {type(e).__name__}: {str(e)[:120]}", flush=True)
            return ""
    return _call


def adversarial_candidates(gold):
    try:
        g = int(str(gold).strip())
        return [str(g - 1), str(g + 1), str(g + 7), "42" if g != 42 else "41"]
    except Exception:
        return ["0", "1", "100", "42"]


def load_benchmarks_with_candidates():
    out = []
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

    try:
        exp27 = json.load(open(RESULTS / "exp27_aime2025_contamination.json"))
        aime_cands = {r["id"]: {"candidate": r.get("candidate") or r.get("sample0_answer"),
                                "correct": r.get("sample0_correct") or r.get("solver_correct")}
                      for r in exp27.get("results", exp27 if isinstance(exp27, list) else [])}
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

    from scipy.stats import binomtest
    print("\n" + "=" * 70)
    print(f"Claude Sonnet 4.6 extractor summary")
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
        print(f"\n[{bench}] n={n}, working={len(working)} ({len(working)/n:.1%}), top-tier={len(tier)}, adv_fp={adv_fp}")
        if tier:
            ci = binomtest(tier_correct, len(tier)).proportion_ci(0.95, "exact")
            print(f"  precision: {tier_correct}/{len(tier)} = {tier_correct/len(tier):.3f} [{ci.low:.3f}, {ci.high:.3f}]")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

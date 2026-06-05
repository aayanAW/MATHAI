"""Run the four NEW extractors on the same 330-problem set as cached
Group B (E05/E06/E07/E09).

New extractors (added to ARCHITECTURE_PIVOT_A §5.2):
  - E02 Llama-4-Maverick (Together AI)
  - E04 DeepSeek-R1-Distill-70B (Together AI)
  - E10 Qwen2.5-72B (Together AI, substituted for Qwen2.5-Math-72B which
        is not on Together)
  - E12 Gemini-2.5-Pro (Google)

Cost guard: hard cap of ~4 × 330 = 1,320 LLM calls. At ~$0.10/call worst
case → max ~$130. With Gemini 2.5 Pro at low rates and Together's open
models cheaper, expected total ~$30-50.

Persists per-problem progress to disk after every call so any
interruption is resumable. Use `make new-extractors` to drive.

Output: results/exp51_llama4maverick_extractor.json (and similar)
written into /Users/aayanalwani/MATHAI/MATHAI/results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Load .env if present (without leaking to global env logs)
ENV_PATH = HERE.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from verifyensemble.extractors.api_wrappers import (
    EXTRACTOR_REGISTRY,
    make_extractor_call,
)
from verifyensemble.extractors.parser import extract_verifier
from verifyensemble.sandbox.adversarial import deployment_time_filter
from verifyensemble.sandbox.classify import classify as gold_classify
from verifyensemble.sandbox.executor import execute_verifier
from verifyensemble.utils.io import save_artifact

MATHAI_RESULTS = HERE.parent / ".." / "MATHAI" / "results"

# Anchor: pull candidates + gold from the existing exp46 cache (gpt-oss
# extractor) since it has the same 330 problems we want to evaluate on.
ANCHOR_CACHE = MATHAI_RESULTS / "exp46_gptoss_extractor.json"
MATH_PROBLEMS = MATHAI_RESULTS / "math_test_sample_500.json"
AIME_PROBLEMS = MATHAI_RESULTS / "aime_2025.json"
CLEANMATH_PROBLEMS = MATHAI_RESULTS / "cleanmath_combo.json"

OUT_FILES = {
    "E02_llama_4_maverick":        "exp51_llama4maverick_extractor.json",
    "E04_deepseek_r1_distill_70B": "exp52_deepseekr1distill_extractor.json",
    "E10_qwen2_5_72B":             "exp53_qwen72b_extractor.json",
    "E12_gemini_2_5_flash":        "exp54_gemini25flash_extractor.json",
    "E08S_gpt_4o":                 "exp55_gpt4o_extractor.json",
    "E04S_gpt_4_1":                "exp56_gpt41_extractor.json",
    "E10S_gpt_5":                  "exp57_gpt5_extractor.json",
    "E01A_claude_opus_4_7":        "exp58_opus47_extractor.json",
    "E02A_claude_opus_4_6":        "exp59_opus46_extractor.json",
    "E03A_claude_haiku_4_5":       "exp61_haiku45_extractor.json",
    "E13_llama_3_3_70B":           "exp62_llama33_70b_extractor.json",
    "E14_qwen_3_235B":             "exp63_qwen3_235b_extractor.json",
}


def load_problem_texts() -> dict[str, dict]:
    """Build {problem_id: {problem, answer, bench}} by merging all sources."""
    texts: dict[str, dict] = {}
    if MATH_PROBLEMS.exists():
        for p in json.load(open(MATH_PROBLEMS)):
            texts[p["id"]] = {"problem": p["problem"],
                              "gold": str(p.get("answer", "")),
                              "bench": "math175"}
    if AIME_PROBLEMS.exists():
        aime = json.load(open(AIME_PROBLEMS))
        for p in (aime if isinstance(aime, list) else aime.get("problems", [])):
            pid = p.get("id") or p.get("problem_id")
            texts[pid] = {"problem": p.get("problem") or p.get("question") or "",
                          "gold": str(p.get("answer", p.get("gold", ""))),
                          "bench": "aime"}
    if CLEANMATH_PROBLEMS.exists():
        cm = json.load(open(CLEANMATH_PROBLEMS))
        for p in (cm if isinstance(cm, list) else cm.get("problems", [])):
            pid = p.get("id") or p.get("problem_id")
            texts[pid] = {"problem": p.get("problem") or p.get("question") or "",
                          "gold": str(p.get("answer", p.get("gold", ""))),
                          "bench": "cleanmath"}
    return texts


def load_anchor() -> tuple[list[dict], dict[str, dict]]:
    """Return (anchor_records, problem_texts) and pick the 330-problem set."""
    anchor = json.load(open(ANCHOR_CACHE))
    recs = anchor["results"] if isinstance(anchor, dict) else anchor
    texts = load_problem_texts()
    return recs, texts


def classify_record(script: str | None, gold: str, candidate: str
                    ) -> dict:
    """Run gold + candidate + adversarial probes; return record fragment."""
    if not script:
        return {
            "classification": "UNVERIFIABLE",
            "script": None,
            "gold_verdict": None,
            "candidate_verdict": None,
            "adv_verdicts": [],
            "adv_fp_count": 0,
            "outcome": "abstained_or_no_script",
        }
    # Gold + candidate verdicts
    g = execute_verifier(script, gold, timeout=8.0)
    c = execute_verifier(script, candidate, timeout=8.0)
    # Offline classification (uses gold)
    classification = gold_classify(script, gold, timeout=8.0)
    # Deployment-time adversarial filter (uses only candidate)
    broken, adv_results = deployment_time_filter(script, candidate, timeout=8.0)
    adv_verdicts = [r["verdict"] for r in adv_results]
    adv_fp = sum(1 for v in adv_verdicts if v is True)
    return {
        "classification": classification,
        "script": script,
        "gold_verdict": g.verdict,
        "candidate_verdict": c.verdict,
        "adv_verdicts": adv_verdicts,
        "adv_broken": broken,
        "adv_fp_count": adv_fp,
        "outcome": "verifier_produced",
    }


def run_one_extractor(eid: str, anchor_recs: list[dict],
                      texts: dict[str, dict], limit: int | None) -> None:
    out_path = MATHAI_RESULTS / OUT_FILES[eid]
    # Resumable: load existing partial
    done_records: dict[str, dict] = {}
    if out_path.exists():
        try:
            prev = json.load(open(out_path))
            for r in prev.get("results", []):
                done_records[r["id"]] = r
        except Exception:
            done_records = {}

    call = make_extractor_call(eid)
    cfg = EXTRACTOR_REGISTRY[eid]
    print(f"\n[{eid}] model={cfg['model']}  resumable from "
          f"{len(done_records)}/{len(anchor_recs)} cached records")

    new_count = 0
    t0 = time.time()
    for i, anchor_rec in enumerate(anchor_recs):
        pid = anchor_rec["id"]
        if pid in done_records:
            continue
        if limit is not None and new_count >= limit:
            break
        bench = anchor_rec.get("bench", texts.get(pid, {}).get("bench", "?"))
        problem_text = texts.get(pid, {}).get("problem", "")
        if not problem_text:
            continue  # cannot run without problem text
        gold = str(anchor_rec.get("gold", ""))
        candidate = str(anchor_rec.get("candidate", ""))

        # Extract verifier
        try:
            ex = extract_verifier(problem_text, call)
        except Exception as e:
            ex = type("Ex", (), {"script": None, "unverifiable": False,
                                  "raw_response": "", "error": str(e)})

        if ex.unverifiable:
            record_frag = {
                "classification": "UNVERIFIABLE",
                "script": None, "gold_verdict": None,
                "candidate_verdict": None, "adv_verdicts": [],
                "adv_fp_count": 0,
                "outcome": "explicit_unverifiable",
            }
        elif ex.script is None:
            record_frag = {
                "classification": "parse_error",
                "script": None, "gold_verdict": None,
                "candidate_verdict": None, "adv_verdicts": [],
                "adv_fp_count": 0,
                "outcome": "parse_error",
            }
        else:
            record_frag = classify_record(ex.script, gold, candidate)

        record = {
            "id": pid,
            "bench": bench,
            "gold": gold,
            "candidate": candidate,
            "solver_correct": bool(anchor_rec.get("solver_correct")),
            **record_frag,
            "elapsed": time.time() - t0,
        }
        done_records[pid] = record
        new_count += 1

        if new_count % 5 == 0:
            save_artifact({"extractor": eid,
                           "model": cfg["model"],
                           "results": list(done_records.values())}, out_path)
            avg = (time.time() - t0) / max(new_count, 1)
            est_left = avg * (len(anchor_recs) - len(done_records))
            print(f"  [{eid}] {len(done_records)}/{len(anchor_recs)}  "
                  f"avg={avg:.1f}s/prob  est_left={est_left/60:.1f}min")

    save_artifact({"extractor": eid,
                   "model": cfg["model"],
                   "results": list(done_records.values())}, out_path)
    print(f"  [{eid}] done. Wrote {out_path} ({len(done_records)} records).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap problems per extractor for smoke-test")
    ap.add_argument("--extractors", nargs="*", default=list(OUT_FILES.keys()),
                    help="subset of extractor ids")
    args = ap.parse_args()

    anchor_recs, texts = load_anchor()
    print(f"anchor records: {len(anchor_recs)}  problem texts: {len(texts)}")

    for eid in args.extractors:
        try:
            run_one_extractor(eid, anchor_recs, texts, args.limit)
        except Exception as e:
            print(f"  [{eid}] ERROR: {e}")


if __name__ == "__main__":
    main()

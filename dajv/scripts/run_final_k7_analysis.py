"""Final batch analysis: 7-extractor full scale-up (4 cached + 3 new OpenAI).

Run AFTER gpt-5 extractor cache reaches 330 records.

Pipeline:
  1. Verify cache completeness.
  2. Re-run hybrid_modality with k=7.
  3. Re-run h6_significance with 5 within-OpenAI pairs (C(5,2)=10).
  4. Re-run 7-extractor aggregation.
  5. Re-render figures.
  6. Recompute headline_metrics.

Exits with code 1 if any cache is short.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAJV = HERE.parent
RESULTS = DAJV / ".." / "MATHAI" / "results"

REQUIRED_CACHES = [
    ("E05_gpt_oss_120B", "exp46_gptoss_extractor.json"),
    ("E06_gpt_5_mini",   "exp50_gpt5_extractor.json"),
    ("E07_claude_4_6",   "exp47_claude_extractor.json"),
    ("E09_qwen3_coder",  "exp48_qwen3coder_extractor.json"),
    ("E08S_gpt_4o",      "exp55_gpt4o_extractor.json"),
    ("E04S_gpt_4_1",     "exp56_gpt41_extractor.json"),
    ("E10S_gpt_5",       "exp57_gpt5_extractor.json"),
    ("E01A_claude_opus_4_7", "exp58_opus47_extractor.json"),
    ("E02A_claude_opus_4_6", "exp59_opus46_extractor.json"),
    ("E03A_claude_haiku_4_5", "exp61_haiku45_extractor.json"),
    ("E13_llama_3_3_70B", "exp62_llama33_70b_extractor.json"),
    ("E14_qwen_3_235B", "exp63_qwen3_235b_extractor.json"),
]


def _run(cmd: list[str]) -> bool:
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(DAJV))
    return r.returncode == 0


def main() -> None:
    ok = True
    for eid, fname in REQUIRED_CACHES:
        p = RESULTS / fname
        if not p.exists():
            print(f"  [{eid}] MISSING")
            ok = False
            continue
        try:
            d = json.load(open(p))
            n = len(d.get("results", []))
            status = "OK" if n >= 300 else f"PARTIAL ({n})"
            print(f"  [{eid}] {status}")
            if n < 300:
                ok = False
        except Exception as e:
            print(f"  [{eid}] LOAD ERROR: {e}")
            ok = False
    if not ok:
        print("\nNot all caches complete. Exiting.")
        sys.exit(1)

    # Run analyses
    scripts = [
        ["python3", "scripts/run_hybrid_modality.py"],
        ["python3", "scripts/run_h6_significance.py", "--min-records", "300"],
        ["python3", "scripts/run_8extractor_aggregation.py",
         "--min-records", "300", "--out", "aggregation_7extractor_final.json"],
        ["python3", "scripts/run_hybrid_modality_figure.py"],
        ["python3", "scripts/run_h6_figure.py"],
        ["python3", "scripts/run_headline_metrics.py"],
    ]
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DAJV)
    for cmd in scripts:
        print(f"\n$ PYTHONPATH={DAJV} {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=str(DAJV), env=env)
        if r.returncode != 0:
            print(f"  step failed: {cmd}")
            sys.exit(1)
    print("\nFinal k=7 analysis complete.")


if __name__ == "__main__":
    main()

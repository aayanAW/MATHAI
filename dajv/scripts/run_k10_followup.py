"""Run the full k=10 ensemble experiment suite once all Anthropic
extractor caches reach 330 records.

Polls for cache completion and exits with code 1 if any are still
partial.  Use ``--wait`` to block until they finish (with a timeout).

Sequence:
  1. Verify caches: E01A (Opus 4.7), E02A (Opus 4.6), E03A (Haiku 4.5).
  2. Run xmod_dajv at k=10 (all variants including BSIA suite).
  3. Run BSIA nested-CV sensitivity sweep at k=10.
  4. Run H6 significance test (n_within_anth = 6 pairs, total 16 within).
  5. Run cross_modality_heatmap (k=10 figure).
  6. Run hybrid_modality (cross-modality numbers at k=10).
  7. Re-run headline_metrics.

Saves: artifacts/k10_followup_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAJV = HERE.parent
RESULTS = DAJV / ".." / "MATHAI" / "results"

REQUIRED = [
    ("E01A_claude_opus_4_7", "exp58_opus47_extractor.json"),
    ("E02A_claude_opus_4_6", "exp59_opus46_extractor.json"),
    ("E03A_claude_haiku_4_5", "exp61_haiku45_extractor.json"),
]


def cache_status() -> dict[str, int]:
    """Return per-extractor record counts."""
    out = {}
    for eid, fname in REQUIRED:
        p = RESULTS / fname
        if not p.exists():
            out[eid] = 0
            continue
        try:
            out[eid] = len(json.load(open(p)).get("results", []))
        except Exception:
            out[eid] = -1
    return out


def all_complete(min_records: int = 300) -> bool:
    s = cache_status()
    return all(v >= min_records for v in s.values())


def run_one(cmd: list[str], env: dict | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(DAJV), env=env)
    return r.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true",
                    help="Poll until all caches are complete.")
    ap.add_argument("--timeout-sec", type=int, default=3600,
                    help="Max wait time when --wait set.")
    ap.add_argument("--min-records", type=int, default=300)
    args = ap.parse_args()

    if args.wait:
        deadline = time.time() + args.timeout_sec
        while time.time() < deadline:
            s = cache_status()
            done = all(v >= args.min_records for v in s.values())
            print(f"[{time.strftime('%H:%M:%S')}] status: " +
                  ", ".join(f"{k}={v}" for k, v in s.items()))
            if done:
                break
            time.sleep(60)
        if not all_complete(args.min_records):
            print("Timeout waiting for caches.")
            sys.exit(1)

    if not all_complete(args.min_records):
        print("Caches not complete:")
        print(json.dumps(cache_status(), indent=2))
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(DAJV)

    sequence = [
        ["python3", "scripts/run_xmod_dajv.py", "--out", "xmod_dajv_k10.json"],
        ["python3", "scripts/run_bsia_sensitivity.py"],
        ["python3", "scripts/run_h6_significance.py"],
        ["python3", "scripts/run_cross_modality_heatmap.py"],
        ["python3", "scripts/run_hybrid_modality.py"],
        ["python3", "scripts/run_8extractor_aggregation.py",
         "--out", "aggregation_k10.json"],
        ["python3", "scripts/run_headline_metrics.py"],
        ["python3", "scripts/run_figures.py"],
        ["python3", "scripts/run_h6_figure.py"],
        ["python3", "scripts/run_hybrid_modality_figure.py"],
    ]

    summary = {"caches": cache_status(), "steps": []}
    for cmd in sequence:
        rc = run_one(cmd, env=env)
        summary["steps"].append({"cmd": " ".join(cmd), "rc": rc})
        if rc != 0:
            print(f"!! step failed: {' '.join(cmd)}")

    artifacts_dir = DAJV / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    with (artifacts_dir / "k10_followup_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote artifacts/k10_followup_summary.json")


if __name__ == "__main__":
    main()

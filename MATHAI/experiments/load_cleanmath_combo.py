"""Load a contamination-clean math benchmark combo from the MathArena HF org.

MathArena hosts unified HF datasets for recent math contests with numeric
answers. All contests below were held AFTER Qwen2.5-Math-7B's training cutoff
(~May 2024) so they are contamination-clean per Wu et al. 2025.

We try each candidate dataset; misses are logged and skipped so one broken
dataset doesn't kill the whole load. The loader normalizes all entries into
the same schema used by aime_2025.json:

    {"id": "<prefix>_<idx>", "problem": str, "answer": str,
     "level": "olympiad", "type": "<contest>", "source_dataset": str}

Output: results/cleanmath_combo.json
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_PATH = RESULTS_DIR / "cleanmath_combo.json"

# Candidate MathArena datasets. Each entry: (hf_id, id_prefix, contest_tag).
# All are post-May 2024 and therefore contamination-clean relative to
# Qwen2.5-Math-7B's training cutoff.
CANDIDATE_DATASETS = [
    ("MathArena/hmmt_feb_2025", "hmmt_feb_2025", "HMMT Feb 2025"),
    ("MathArena/hmmt_nov_2024", "hmmt_nov_2024", "HMMT Nov 2024"),
    ("MathArena/brumo_2025",    "brumo_2025",    "BRUMO 2025"),
    ("MathArena/smt_2025",      "smt_2025",      "SMT 2025"),
    ("MathArena/apex_2025",     "apex_2025",     "APEX 2025"),
]


def normalize_row(row: dict, prefix: str, idx: int, tag: str, hf_id: str) -> dict | None:
    """Extract problem and answer from an HF row into our unified schema.

    MathArena datasets vary slightly in column names. We check a short list
    of known aliases and return None if the row can't be parsed.
    """
    problem_keys = ["problem", "question", "prompt"]
    answer_keys = ["answer", "final_answer", "gold_answer", "solution_answer"]

    problem = None
    for k in problem_keys:
        if k in row and row[k] is not None and str(row[k]).strip():
            problem = str(row[k]).strip()
            break
    if not problem:
        return None

    answer = None
    for k in answer_keys:
        if k in row and row[k] is not None and str(row[k]).strip():
            answer = str(row[k]).strip()
            break
    if not answer:
        return None

    return {
        "id": f"{prefix}_{idx + 1}",
        "problem": problem,
        "answer": answer,
        "level": "olympiad",
        "type": tag,
        "source_dataset": hf_id,
    }


def try_load(hf_id: str) -> list[dict] | None:
    """Try to load an HF dataset. Returns rows (as a list of dicts) or None."""
    try:
        ds = load_dataset(hf_id, split="train")
    except Exception:
        try:
            ds = load_dataset(hf_id, split="test")
        except Exception as e:
            print(f"  FAIL {hf_id}: {type(e).__name__}: {str(e)[:200]}")
            return None
    rows = [dict(r) for r in ds]
    return rows


def main() -> None:
    combined: list[dict] = []
    report: list[dict] = []

    for hf_id, prefix, tag in CANDIDATE_DATASETS:
        print(f"\n>>> Trying {hf_id}")
        rows = try_load(hf_id)
        if rows is None:
            report.append({"dataset": hf_id, "status": "failed", "n": 0})
            continue

        parsed = []
        for i, row in enumerate(rows):
            norm = normalize_row(row, prefix, i, tag, hf_id)
            if norm is not None:
                parsed.append(norm)

        print(f"  OK: {len(parsed)}/{len(rows)} rows parsed from {hf_id}")
        if parsed and len(parsed) > 0:
            print(f"  First problem preview: {parsed[0]['problem'][:120]}...")
            print(f"  First answer: {parsed[0]['answer']}")

        combined.extend(parsed)
        report.append({"dataset": hf_id, "status": "ok", "n": len(parsed), "total_rows": len(rows)})

    print(f"\n{'=' * 60}")
    print(f"Combined total: {len(combined)} problems from {sum(1 for r in report if r['status'] == 'ok')} datasets")
    print(f"{'=' * 60}")
    for r in report:
        marker = "OK" if r["status"] == "ok" else "FAIL"
        print(f"  [{marker}] {r['dataset']}: {r['n']} problems")

    if not combined:
        print("\nNo datasets loaded. Aborting.")
        sys.exit(1)

    with open(OUT_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()

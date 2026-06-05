"""Lean 4 compile-and-typecheck for the 270 Lean samples in exp52 via Modal.

For each generated Lean theorem+proof from Claude Sonnet 4.6, GPT-5-mini, and
DeepSeek-V3, write it to a file inside a lightweight Lean 4 + Mathlib project
on Modal, attempt compilation with `lake env lean`, and record whether the
proof type-checks.

Launch:
    modal run experiments/run_exp53_lean_compile.py::run_all
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "mathai-lean-compile"
app = modal.App(APP_NAME)

# Lean 4 image with Mathlib cache pulled.
lean_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential", "wget")
    .run_commands(
        # Install elan (Lean toolchain manager)
        "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.14.0",
    )
    .run_commands(
        # Put elan/lean on PATH and create a Mathlib-dependent project.
        "/root/.elan/bin/lake --version",
        "mkdir -p /workdir",
        # Lake "math" template creates a project with Mathlib as a dependency.
        "cd /workdir && /root/.elan/bin/lake +leanprover/lean4:v4.14.0 new mathai_test math",
    )
    .run_commands(
        # Pull the Mathlib-compiled cache.
        "cd /workdir/mathai_test && /root/.elan/bin/lake update || true",
        "cd /workdir/mathai_test && /root/.elan/bin/lake exe cache get || true",
        "cd /workdir/mathai_test && /root/.elan/bin/lake build MathaiTest || true",
    )
    .env({"PATH": "/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)


@app.function(image=lean_image, timeout=60 * 60 * 2, cpu=4.0, memory=8192)
def compile_one(source_code: str, extractor: str, problem_id: str) -> dict:
    """Write Lean source to a file in the Mathlib project and try to compile."""
    import subprocess
    import tempfile
    import os
    import time

    # Build a minimal file with a Mathlib import.
    prelude = "import Mathlib\n\nopen Classical in\n"
    src = prelude + source_code.strip() + "\n"

    workdir = "/workdir/mathai_test"
    lean_file = os.path.join(workdir, "MathaiTest", "Trial.lean")
    os.makedirs(os.path.dirname(lean_file), exist_ok=True)
    with open(lean_file, "w") as f:
        f.write(src)

    t0 = time.time()
    try:
        result = subprocess.run(
            ["/root/.elan/bin/lake", "env", "lean", lean_file],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        ok = result.returncode == 0
        stderr = (result.stderr or "")[-800:]
        stdout = (result.stdout or "")[-400:]
    except subprocess.TimeoutExpired:
        ok = False
        stderr = "TIMEOUT(120s)"
        stdout = ""
    except Exception as e:
        ok = False
        stderr = f"EXC: {type(e).__name__}: {e}"[:400]
        stdout = ""
    return {
        "extractor": extractor,
        "problem_id": problem_id,
        "compile_ok": ok,
        "stderr_tail": stderr,
        "stdout_tail": stdout,
        "elapsed": time.time() - t0,
    }


@app.local_entrypoint()
def run_all():
    RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")
    SRC = RESULTS / "exp52_lean_headtohead.json"
    OUT = RESULTS / "exp53_lean_compile.json"

    data = json.load(open(SRC))
    # Only try rows that actually emitted Lean source
    rows_to_try = [r for r in data["results"] if r.get("lean_src") and r.get("verdict") == "ok"]
    print(f"Total Lean samples to compile-check: {len(rows_to_try)}")

    if OUT.exists():
        out = json.load(open(OUT))
        done = {(r["extractor"], r["problem_id"]) for r in out["results"]}
    else:
        out = {"results": []}
        done = set()

    todo = [r for r in rows_to_try if (r["extractor"], r["id"]) not in done]
    print(f"Already done: {len(done)}, Todo: {len(todo)}")

    # Run in small parallel batches
    args_list = [(r["lean_src"], r["extractor"], r["id"]) for r in todo]

    n_done = 0
    for result in compile_one.starmap(args_list, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"  [err] {type(result).__name__}: {result}")
            continue
        out["results"].append(result)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  [{n_done}/{len(todo)}] ok={sum(1 for r in out['results'] if r.get('compile_ok'))}")
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2, default=str)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Summary
    from collections import Counter
    print("\n=== Lean compile-check summary ===")
    for ext in ["claude-sonnet-4-6", "gpt-5-mini", "deepseek-v3"]:
        erows = [r for r in out["results"] if r["extractor"] == ext]
        n = len(erows)
        ok = sum(1 for r in erows if r.get("compile_ok"))
        print(f"  {ext}: {ok}/{n} = {ok/n:.1%} compile + typecheck")
    print(f"\nSaved -> {OUT}")

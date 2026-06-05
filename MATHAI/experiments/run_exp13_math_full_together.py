"""Experiment 13 (Together AI): Full MATH Test Set Evaluation (5000 problems).

Headline experiment for the paper. Runs the complete ExeVer pipeline +
baselines on the FULL MATH test set (~5000 problems across 7 subjects)
with Qwen2.5-Math-7B-Instruct via Together AI's API.

No Modal dependency -- runs locally with async API calls to Together AI
and local subprocess execution for verification scripts.

Usage:
    # Full run (greedy + ExeVer + 4 samples per problem)
    python experiments/run_exp13_math_full_together.py

    # Skip sampled solutions (greedy + ExeVer only)
    python experiments/run_exp13_math_full_together.py --no-samples

    # Custom batch/concurrency
    python experiments/run_exp13_math_full_together.py --batch-size 500 --max-concurrent 50

Metrics (by level and by subject):
- Greedy CoT accuracy
- ExeVer accuracy (with repair)
- Majority@4, Best@4 (if --no-samples not set)
- Verification coverage, echo chamber rate
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup -- allow importing from src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.answer_check import answers_equivalent, extract_model_answer
from src.exever.executor import execute_verification_script
from src.exever.step_parser import extract_assertions, parse_nl_steps

# ---------------------------------------------------------------------------
# Together AI config
# ---------------------------------------------------------------------------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
BASE_URL = "https://api.together.xyz/v1"
MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "exp13_checkpoint.json"
OUTPUT_PATH = RESULTS_DIR / "exp13_math_full.json"

# ---------------------------------------------------------------------------
# Prompts (identical to exp5)
# ---------------------------------------------------------------------------
SOLVE_PROMPT = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

## Step 2: [brief title]
[reasoning and computation for this step]

...continue for all steps...

At the end, state your final answer as: The answer is \\boxed{{answer}}.

Problem: {problem}"""

VERIFY_PROMPT = """Below is a step-by-step math solution. Write a Python/SymPy script that checks each step.

IMPORTANT: Your script MUST use assert statements to verify each step.

Example of what I want:
```python
from sympy import *
x = symbols('x')
# === STEP 1 ===
eq = x**2 - 5*x + 6
# === STEP 2 ===
claimed = (x-2)*(x-3)
assert expand(claimed - eq) == 0, "FAIL:Step 2: factoring"
# === STEP 3 ===
assert eq.subs(x, 2) == 0, "FAIL:Step 3: x=2 not root"
assert eq.subs(x, 3) == 0, "FAIL:Step 3: x=3 not root"
# === STEP 4 ===
assert 2 + 3 == 5, "FAIL:Step 4: sum"
print("ANSWER:", 5)
```

Rules:
- Start with: from sympy import *
- For each step write: # === STEP k === then assert statements
- Every step MUST have at least one assert
- Use expand() not simplify()
- End with: print("ANSWER:", answer)
- Output ONLY the Python code, nothing else.

Solution to verify:
{solution}

Write ONLY the Python verification script inside a ```python code block."""

REPAIR_PROMPT = """You are solving the following math problem. Your previous solution had an error at Step {step_num}.

Problem: {problem}

Verified correct steps (keep these exactly):
{verified_prefix}

The error was at Step {step_num}:
{failed_step}

Error message from verification:
{error_message}

Please provide:
1. A corrected Step {step_num} and all subsequent steps
2. Format each step as "## Step k: [title]" followed by reasoning
3. State your final answer as: The answer is \\boxed{{answer}}.

Continue from Step {step_num}:"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exp13")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_script(response: str) -> str:
    """Extract Python script from model response."""
    if "```" not in response:
        lines = response.split("\n")
        code_lines: list[str] = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                if stripped.startswith(
                    ("from ", "import ", "#", "x ", "x=", "eq",
                     "def ", "class ", "print(")
                ) or stripped == "":
                    started = True
                    code_lines.append(line)
            else:
                code_lines.append(line)
        return "\n".join(code_lines).strip()

    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        py_blocks = [m for m in matches if any(
            kw in m for kw in ["import", "assert", "def ", "print(", "from "]
        )]
        if py_blocks:
            return max(py_blocks, key=len).strip()
    return ""


def _is_trivial_assert(a: str) -> bool:
    body = a.replace("assert ", "", 1).strip()
    if body.startswith("True"):
        return True
    if "==" in body:
        parts = body.split("==", 1)
        if parts[0].strip() == parts[1].strip().split(",")[0].strip():
            return True
    if body.startswith("isinstance("):
        return True
    return False


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint from disk, or return empty structure."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {
        "greedy_solutions": {},     # id -> solution text
        "verify_responses": {},     # id -> verify response text
        "repair_solutions": {},     # id -> repair solution text
        "repair_verify": {},        # id -> repair verify response text
        "sampled_solutions": {},    # id -> [list of 4 solutions]
        "model_used": None,
    }


def save_checkpoint(ckpt: Dict[str, Any]) -> None:
    """Save checkpoint to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f)


# ---------------------------------------------------------------------------
# Async API client with rate-limiting + retries
# ---------------------------------------------------------------------------
class TogetherClient:
    """Async wrapper around Together AI's OpenAI-compatible API."""

    def __init__(self, model: str, max_concurrent: int = 50):
        from openai import AsyncOpenAI

        if not TOGETHER_API_KEY:
            raise RuntimeError(
                "TOGETHER_API_KEY environment variable is not set. "
                "Export it before running: export TOGETHER_API_KEY=<your-key>"
            )
        self.client = AsyncOpenAI(
            api_key=TOGETHER_API_KEY,
            base_url=BASE_URL,
        )
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._request_count = 0

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        n: int = 1,
    ) -> List[str]:
        """Generate completions with retry + backoff.

        Returns list of n completion strings.
        """
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    self._request_count += 1
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop or [],
                        n=n,
                    )
                    return [choice.message.content or "" for choice in response.choices]
            except Exception as e:
                err_str = str(e).lower()
                # Rate limit or server error -- retry with backoff
                if any(kw in err_str for kw in ["rate", "429", "500", "502", "503"]):
                    delay = base_delay * (2 ** attempt)
                    log.warning(
                        "API error (attempt %d/%d): %s -- retrying in %.1fs",
                        attempt + 1, max_retries, str(e)[:120], delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error("Non-retryable API error: %s", str(e)[:200])
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries")

    async def check_model_available(self) -> str:
        """Check if primary model is available, fall back if needed.

        Returns the model name that was successfully used.
        """
        try:
            result = await self.generate(
                "What is 2+2? Answer with just the number.",
                max_tokens=16,
            )
            if result and result[0].strip():
                log.info("Model %s is available", self.model)
                return self.model
        except Exception as e:
            log.warning("Primary model %s unavailable: %s", self.model, str(e)[:100])

        # Try fallback
        log.info("Trying fallback model %s ...", FALLBACK_MODEL)
        self.model = FALLBACK_MODEL
        try:
            result = await self.generate(
                "What is 2+2? Answer with just the number.",
                max_tokens=16,
            )
            if result and result[0].strip():
                log.info("Fallback model %s is available", self.model)
                return self.model
        except Exception as e:
            raise RuntimeError(
                f"Neither {MODEL} nor {FALLBACK_MODEL} is available on Together AI. "
                f"Error: {e}"
            )
        return self.model


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------
async def phase_greedy(
    client: TogetherClient,
    problems: List[Dict],
    ckpt: Dict[str, Any],
    batch_size: int,
) -> Dict[str, str]:
    """Phase 1: Generate greedy CoT solutions (temperature=0)."""
    solutions = dict(ckpt.get("greedy_solutions", {}))
    todo = [p for p in problems if p["id"] not in solutions]

    if not todo:
        log.info("[Greedy] All %d problems already in checkpoint, skipping", len(problems))
        return solutions

    log.info("[Greedy] Generating for %d problems (%d cached)", len(todo), len(solutions))

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        log.info(
            "  Batch %d-%d / %d ...",
            batch_start, batch_start + len(batch), len(todo),
        )

        tasks = [
            client.generate(
                SOLVE_PROMPT.format(problem=p["problem"]),
                temperature=0.0,
                max_tokens=2048,
            )
            for p in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for p, res in zip(batch, results):
            if isinstance(res, Exception):
                log.error("  Failed for %s: %s", p["id"], str(res)[:100])
                solutions[p["id"]] = ""
            else:
                solutions[p["id"]] = res[0]

        # Checkpoint after each batch
        ckpt["greedy_solutions"] = solutions
        save_checkpoint(ckpt)
        log.info("  Checkpoint saved (%d greedy solutions total)", len(solutions))

    return solutions


async def phase_verify(
    client: TogetherClient,
    problems: List[Dict],
    greedy_solutions: Dict[str, str],
    ckpt: Dict[str, Any],
    batch_size: int,
) -> Dict[str, str]:
    """Phase 2: Generate verification scripts for greedy solutions."""
    responses = dict(ckpt.get("verify_responses", {}))
    todo = [p for p in problems if p["id"] not in responses and greedy_solutions.get(p["id"])]

    if not todo:
        log.info("[Verify] All problems already in checkpoint, skipping")
        return responses

    log.info("[Verify] Generating scripts for %d problems (%d cached)", len(todo), len(responses))

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        log.info(
            "  Batch %d-%d / %d ...",
            batch_start, batch_start + len(batch), len(todo),
        )

        tasks = [
            client.generate(
                VERIFY_PROMPT.format(solution=greedy_solutions[p["id"]]),
                temperature=0.0,
                max_tokens=2048,
                # no stop token for chat API — extract code from response
            )
            for p in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for p, res in zip(batch, results):
            if isinstance(res, Exception):
                log.error("  Failed for %s: %s", p["id"], str(res)[:100])
                responses[p["id"]] = ""
            else:
                responses[p["id"]] = res[0]

        ckpt["verify_responses"] = responses
        save_checkpoint(ckpt)
        log.info("  Checkpoint saved (%d verify responses total)", len(responses))

    return responses


async def phase_repair(
    client: TogetherClient,
    problems: List[Dict],
    repair_inputs: List[Dict[str, Any]],
    ckpt: Dict[str, Any],
    batch_size: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Phase 3: Generate repairs + re-verification for failed problems."""
    repair_solutions = dict(ckpt.get("repair_solutions", {}))
    repair_verify = dict(ckpt.get("repair_verify", {}))
    todo = [ri for ri in repair_inputs if ri["id"] not in repair_solutions]

    if not todo:
        log.info("[Repair] All repairs already in checkpoint, skipping")
        return repair_solutions, repair_verify

    log.info("[Repair] Generating repairs for %d problems (%d cached)", len(todo), len(repair_solutions))

    # Step 1: Generate repaired solutions
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        log.info("  Repair batch %d-%d / %d ...", batch_start, batch_start + len(batch), len(todo))

        tasks = [
            client.generate(ri["repair_prompt"], temperature=0.0, max_tokens=2048)
            for ri in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ri, res in zip(batch, results):
            if isinstance(res, Exception):
                log.error("  Repair failed for %s: %s", ri["id"], str(res)[:100])
                repair_solutions[ri["id"]] = ""
            else:
                repair_solutions[ri["id"]] = res[0]

        ckpt["repair_solutions"] = repair_solutions
        save_checkpoint(ckpt)

    # Step 2: Generate re-verification scripts for repaired solutions
    todo_verify = [
        ri for ri in repair_inputs
        if ri["id"] not in repair_verify and repair_solutions.get(ri["id"])
    ]

    for batch_start in range(0, len(todo_verify), batch_size):
        batch = todo_verify[batch_start : batch_start + batch_size]
        log.info(
            "  Re-verify batch %d-%d / %d ...",
            batch_start, batch_start + len(batch), len(todo_verify),
        )

        tasks = [
            client.generate(
                VERIFY_PROMPT.format(solution=repair_solutions[ri["id"]]),
                temperature=0.0,
                max_tokens=2048,
                # no stop token for chat API — extract code from response
            )
            for ri in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ri, res in zip(batch, results):
            if isinstance(res, Exception):
                repair_verify[ri["id"]] = ""
            else:
                repair_verify[ri["id"]] = res[0]

        ckpt["repair_verify"] = repair_verify
        save_checkpoint(ckpt)

    return repair_solutions, repair_verify


async def phase_samples(
    client: TogetherClient,
    problems: List[Dict],
    ckpt: Dict[str, Any],
    batch_size: int,
) -> Dict[str, List[str]]:
    """Phase 4 (optional): Generate 4 sampled solutions per problem."""
    samples = dict(ckpt.get("sampled_solutions", {}))
    todo = [p for p in problems if p["id"] not in samples]

    if not todo:
        log.info("[Samples] All %d problems already in checkpoint, skipping", len(problems))
        return samples

    log.info("[Samples] Generating 4 samples each for %d problems (%d cached)", len(todo), len(samples))

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        log.info(
            "  Batch %d-%d / %d ...",
            batch_start, batch_start + len(batch), len(todo),
        )

        # Together AI supports n>1 for some models; if not, fall back to
        # issuing 4 separate requests per problem.
        tasks = []
        for p in batch:
            prompt = SOLVE_PROMPT.format(problem=p["problem"])
            try:
                tasks.append(
                    client.generate(prompt, temperature=0.7, max_tokens=2048, n=4)
                )
            except Exception:
                # Fallback: 4 separate calls
                for _ in range(4):
                    tasks.append(
                        client.generate(prompt, temperature=0.7, max_tokens=2048, n=1)
                    )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # If n=4 worked, each result is a list of 4 strings.
        # If we fell back, every 4 results belong to one problem.
        idx = 0
        for p in batch:
            res = results[idx]
            if isinstance(res, Exception):
                log.error("  Sample failed for %s: %s", p["id"], str(res)[:100])
                samples[p["id"]] = [""] * 4
                idx += 1
            elif isinstance(res, list) and len(res) == 4:
                samples[p["id"]] = res
                idx += 1
            else:
                # Fallback mode: collect next 4
                collected: list[str] = []
                for _ in range(4):
                    r = results[idx]
                    idx += 1
                    if isinstance(r, Exception):
                        collected.append("")
                    elif isinstance(r, list):
                        collected.append(r[0] if r else "")
                    else:
                        collected.append("")
                samples[p["id"]] = collected

        ckpt["sampled_solutions"] = samples
        save_checkpoint(ckpt)
        log.info("  Checkpoint saved (%d sampled sets total)", len(samples))

    return samples


# ---------------------------------------------------------------------------
# Local execution & evaluation
# ---------------------------------------------------------------------------
def run_exever_locally(
    problems: List[Dict],
    greedy_solutions: Dict[str, str],
    verify_responses: Dict[str, str],
) -> Tuple[List[Dict], List[Dict]]:
    """Execute verification scripts locally and collect results.

    Returns:
        (exever_results, repair_inputs)
    """
    exever_results: List[Dict] = []
    repair_inputs: List[Dict] = []

    n_valid = 0
    n_executed = 0
    n_all_pass = 0
    n_assertion_fail = 0
    n_runtime_error = 0
    n_empty = 0
    total_assertions = 0
    total_nontrivial = 0

    for i, prob in enumerate(problems):
        pid = prob["id"]
        sol = greedy_solutions.get(pid, "")
        ver_resp = verify_responses.get(pid, "")

        pred = extract_model_answer(sol)
        correct, _ = answers_equivalent(pred, prob["answer"])

        script = extract_script(ver_resp)
        if not script.strip():
            n_empty += 1
            exever_results.append({
                "id": pid, "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "NO_SCRIPT", "assertions": 0,
                "echo_chamber": None, "repaired": False,
                "nl_solution": sol,
            })
            continue

        # Check syntax
        try:
            compile(script, "<verify>", "exec")
            n_valid += 1
        except SyntaxError:
            exever_results.append({
                "id": pid, "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "SYNTAX_ERROR", "assertions": 0,
                "echo_chamber": None, "repaired": False,
                "nl_solution": sol,
            })
            continue

        assertions = extract_assertions(script)
        n_asserts = len(assertions)
        total_assertions += n_asserts
        nontrivial = sum(1 for a in assertions if not _is_trivial_assert(a))
        total_nontrivial += nontrivial

        exec_result = execute_verification_script(script, timeout=30)

        if exec_result.success:
            n_executed += 1
            n_all_pass += 1
            script_answer = exec_result.answer_extracted or ""
            echo = None
            if script_answer:
                ans_ok, _ = answers_equivalent(script_answer, prob["answer"])
                echo = not ans_ok
            else:
                echo = not correct
            exever_results.append({
                "id": pid, "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": "ALL_PASS", "assertions": n_asserts,
                "echo_chamber": echo, "repaired": False,
                "script_answer": script_answer, "nl_solution": sol,
            })
        elif exec_result.assertion_error:
            n_executed += 1
            n_assertion_fail += 1
            step = exec_result.error_step
            error_msg = exec_result.error_message

            # Build repair prompt
            nl_steps = parse_nl_steps(sol)
            step_idx = step if step >= 0 else 0
            verified_prefix = "\n\n".join(
                f"## Step {j+1}: {s}" for j, s in enumerate(nl_steps[:step_idx])
            )
            failed_step = nl_steps[step_idx] if step_idx < len(nl_steps) else ""

            repair_prompt = REPAIR_PROMPT.format(
                problem=prob["problem"],
                verified_prefix=verified_prefix,
                failed_step=failed_step,
                step_num=step_idx + 1,
                error_message=error_msg[:300],
            )
            repair_inputs.append({
                "id": pid,
                "idx": i,
                "repair_prompt": repair_prompt,
                "original_pred": pred,
                "original_correct": correct,
            })
            exever_results.append({
                "id": pid, "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": f"FAIL_STEP_{step}", "assertions": n_asserts,
                "echo_chamber": None, "repaired": False,
                "nl_solution": sol,
            })
        else:
            if exec_result.timeout:
                verdict = "TIMEOUT"
            else:
                n_runtime_error += 1
                verdict = "RUNTIME_ERROR"
            exever_results.append({
                "id": pid, "level": prob["level"], "type": prob["type"],
                "gold_answer": prob["answer"], "predicted_answer": pred,
                "answer_correct": correct,
                "verdict": verdict, "assertions": n_asserts,
                "echo_chamber": None, "repaired": False,
                "nl_solution": sol,
            })

        # Progress
        if (i + 1) % 500 == 0:
            log.info("  Executed %d / %d verification scripts", i + 1, len(problems))

    stats = {
        "n_valid": n_valid,
        "n_executed": n_executed,
        "n_all_pass": n_all_pass,
        "n_assertion_fail": n_assertion_fail,
        "n_runtime_error": n_runtime_error,
        "n_empty": n_empty,
        "total_assertions": total_assertions,
        "total_nontrivial": total_nontrivial,
    }
    log.info(
        "Verification stats: valid=%d executed=%d pass=%d fail=%d error=%d empty=%d",
        n_valid, n_executed, n_all_pass, n_assertion_fail, n_runtime_error, n_empty,
    )

    return exever_results, repair_inputs, stats  # type: ignore[return-value]


def apply_repairs(
    problems: List[Dict],
    exever_results: List[Dict],
    repair_inputs: List[Dict],
    repair_solutions: Dict[str, str],
    repair_verify: Dict[str, str],
) -> int:
    """Apply repair results back into exever_results. Returns count of successful repairs."""
    # Build index from problem id to exever_results position
    id_to_idx = {r["id"]: i for i, r in enumerate(exever_results)}

    n_repair_success = 0
    for ri in repair_inputs:
        pid = ri["id"]
        idx = id_to_idx.get(pid)
        if idx is None:
            continue

        prob = next((p for p in problems if p["id"] == pid), None)
        if prob is None:
            continue

        repaired_sol = repair_solutions.get(pid, "")
        reverify_resp = repair_verify.get(pid, "")

        repaired_pred = extract_model_answer(repaired_sol)
        repaired_correct, _ = answers_equivalent(repaired_pred, prob["answer"])

        # Check re-verification
        reverify_script = extract_script(reverify_resp)
        repair_pass = False
        if reverify_script.strip():
            try:
                compile(reverify_script, "<reverify>", "exec")
                rexec = execute_verification_script(reverify_script, timeout=30)
                if rexec.success:
                    repair_pass = True
                    n_repair_success += 1
            except SyntaxError:
                pass

        if repair_pass and repaired_correct:
            exever_results[idx]["verdict"] = "REPAIRED"
            exever_results[idx]["predicted_answer"] = repaired_pred
            exever_results[idx]["answer_correct"] = True
            exever_results[idx]["repaired"] = True
        elif repaired_correct:
            exever_results[idx]["predicted_answer"] = repaired_pred
            exever_results[idx]["answer_correct"] = True
            exever_results[idx]["repaired"] = True
            exever_results[idx]["verdict"] = "REPAIRED_UNVERIFIED"

    return n_repair_success


def evaluate_baselines(
    problems: List[Dict],
    greedy_solutions: Dict[str, str],
    sampled_solutions: Optional[Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Evaluate greedy CoT and sampled baselines."""
    greedy_results = []
    greedy_correct = 0

    for prob in problems:
        sol = greedy_solutions.get(prob["id"], "")
        pred = extract_model_answer(sol)
        correct, method = answers_equivalent(pred, prob["answer"])
        greedy_correct += int(correct)
        greedy_results.append({
            "id": prob["id"],
            "level": prob["level"],
            "type": prob["type"],
            "gold_answer": prob["answer"],
            "predicted_answer": pred,
            "correct": correct,
        })

    greedy_acc = greedy_correct / len(problems)
    log.info("Greedy CoT pass@1: %.3f (%d/%d)", greedy_acc, greedy_correct, len(problems))

    # Sampled baselines
    sampled_results = []
    majority4_correct = 0
    best4_correct = 0
    sampled_pass1_correct = 0

    if sampled_solutions:
        for prob in problems:
            pid = prob["id"]
            samples = sampled_solutions.get(pid, [])
            if not samples:
                sampled_results.append({
                    "id": pid, "level": prob["level"], "type": prob["type"],
                    "best4_correct": False, "majority4_correct": False,
                    "pass1_correct": False,
                })
                continue

            sample_preds = []
            any_correct = False
            for sol in samples:
                pred = extract_model_answer(sol)
                correct, _ = answers_equivalent(pred, prob["answer"])
                sample_preds.append({"pred": pred, "correct": correct})
                if correct:
                    any_correct = True

            if sample_preds and sample_preds[0]["correct"]:
                sampled_pass1_correct += 1

            if any_correct:
                best4_correct += 1

            # Majority@4 with symbolic equivalence grouping
            groups: list[tuple] = []
            for sp in sample_preds:
                merged = False
                for gi, (canonical, count, _correct) in enumerate(groups):
                    eq, _ = answers_equivalent(sp["pred"], canonical)
                    if eq:
                        groups[gi] = (canonical, count + 1, _correct or sp["correct"])
                        merged = True
                        break
                if not merged:
                    groups.append((sp["pred"], 1, sp["correct"]))
            groups.sort(key=lambda x: x[1], reverse=True)
            if groups and groups[0][2]:
                majority4_correct += 1

            sampled_results.append({
                "id": pid, "level": prob["level"], "type": prob["type"],
                "best4_correct": any_correct,
                "majority4_correct": groups[0][2] if groups else False,
                "pass1_correct": sample_preds[0]["correct"] if sample_preds else False,
            })

        sampled_acc = sampled_pass1_correct / len(problems)
        best4_acc = best4_correct / len(problems)
        majority4_acc = majority4_correct / len(problems)
        log.info("Sampled pass@1: %.3f", sampled_acc)
        log.info("Majority@4: %.3f", majority4_acc)
        log.info("Best-of-4: %.3f", best4_acc)
    else:
        sampled_acc = None
        best4_acc = None
        majority4_acc = None

    return {
        "greedy_results": greedy_results,
        "sampled_results": sampled_results,
        "greedy_acc": greedy_acc,
        "sampled_acc": sampled_acc,
        "majority4_acc": majority4_acc,
        "best4_acc": best4_acc,
    }


def compute_breakdowns(
    problems: List[Dict],
    greedy_results: List[Dict],
    sampled_results: List[Dict],
    exever_results: List[Dict],
    has_samples: bool,
) -> Dict[str, Any]:
    """Compute by-level and by-subject breakdowns."""
    # Build index
    id_to_greedy = {r["id"]: r for r in greedy_results}
    id_to_sampled = {r["id"]: r for r in sampled_results} if sampled_results else {}
    id_to_exever = {r["id"]: r for r in exever_results}

    # By level
    by_level: Dict[str, Any] = {}
    for lv in [1, 2, 3, 4, 5]:
        lv_probs = [p for p in problems if p["level"] == lv]
        n_lv = len(lv_probs)
        if n_lv == 0:
            continue

        greedy_lv = sum(1 for p in lv_probs if id_to_greedy.get(p["id"], {}).get("correct", False)) / n_lv
        exever_lv = sum(1 for p in lv_probs if id_to_exever.get(p["id"], {}).get("answer_correct", False)) / n_lv

        entry: Dict[str, Any] = {"n": n_lv, "greedy": greedy_lv, "exever": exever_lv}
        if has_samples and id_to_sampled:
            entry["pass1"] = sum(1 for p in lv_probs if id_to_sampled.get(p["id"], {}).get("pass1_correct", False)) / n_lv
            entry["best4"] = sum(1 for p in lv_probs if id_to_sampled.get(p["id"], {}).get("best4_correct", False)) / n_lv
            entry["maj4"] = sum(1 for p in lv_probs if id_to_sampled.get(p["id"], {}).get("majority4_correct", False)) / n_lv

        by_level[str(lv)] = entry

    # By subject
    by_subject: Dict[str, Any] = {}
    subjects = sorted(set(p["type"] for p in problems))
    for subj in subjects:
        s_probs = [p for p in problems if p["type"] == subj]
        n_s = len(s_probs)
        if n_s == 0:
            continue

        greedy_s = sum(1 for p in s_probs if id_to_greedy.get(p["id"], {}).get("correct", False)) / n_s
        exever_s = sum(1 for p in s_probs if id_to_exever.get(p["id"], {}).get("answer_correct", False)) / n_s

        entry = {"n": n_s, "greedy": greedy_s, "exever": exever_s}
        if has_samples and id_to_sampled:
            entry["pass1"] = sum(1 for p in s_probs if id_to_sampled.get(p["id"], {}).get("pass1_correct", False)) / n_s
            entry["best4"] = sum(1 for p in s_probs if id_to_sampled.get(p["id"], {}).get("best4_correct", False)) / n_s
            entry["maj4"] = sum(1 for p in s_probs if id_to_sampled.get(p["id"], {}).get("majority4_correct", False)) / n_s

        by_subject[subj] = entry

    # Verifiability map
    vmap: Dict[str, Any] = {}
    for subj in subjects:
        for lv in [1, 2, 3, 4, 5]:
            subset = [r for r in exever_results if r["type"] == subj and r["level"] == lv]
            if not subset:
                continue
            total = len(subset)
            has_assert = sum(1 for r in subset if r["assertions"] > 0)
            all_pass = sum(1 for r in subset if r["verdict"] == "ALL_PASS")
            fail = sum(1 for r in subset if "FAIL" in r["verdict"])
            vmap[f"{subj}_{lv}"] = {
                "subject": subj, "level": lv, "n": total,
                "has_assertions_pct": has_assert / total,
                "all_pass_pct": all_pass / total,
                "fail_pct": fail / total,
                "coverage": (all_pass + fail) / total,
            }

    return {"by_level": by_level, "by_subject": by_subject, "verifiability_map": vmap}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(args: argparse.Namespace) -> None:
    # Load problems
    data_path = RESULTS_DIR / "math_test_full.json"
    with open(data_path) as f:
        problems = json.load(f)
    log.info("Loaded %d problems from %s", len(problems), data_path)

    # Load checkpoint
    ckpt = load_checkpoint()
    log.info(
        "Checkpoint: %d greedy, %d verify, %d repair, %d samples",
        len(ckpt.get("greedy_solutions", {})),
        len(ckpt.get("verify_responses", {})),
        len(ckpt.get("repair_solutions", {})),
        len(ckpt.get("sampled_solutions", {})),
    )

    # Init client
    client = TogetherClient(model=MODEL, max_concurrent=args.max_concurrent)
    model_used = await client.check_model_available()
    ckpt["model_used"] = model_used

    # ---- Phase 1: Greedy CoT ----
    t0 = time.time()
    greedy_solutions = await phase_greedy(client, problems, ckpt, args.batch_size)
    t1 = time.time()
    log.info("Phase 1 (greedy) done in %.1fs", t1 - t0)

    # ---- Phase 2: Verification scripts ----
    t0 = time.time()
    verify_responses = await phase_verify(client, problems, greedy_solutions, ckpt, args.batch_size)
    t1 = time.time()
    log.info("Phase 2 (verify) done in %.1fs", t1 - t0)

    # ---- Local execution & evaluation ----
    log.info("Running local verification execution ...")
    t0 = time.time()
    exever_results, repair_inputs, verify_stats = run_exever_locally(
        problems, greedy_solutions, verify_responses
    )
    t1 = time.time()
    log.info("Local execution done in %.1fs (%d need repair)", t1 - t0, len(repair_inputs))

    # ---- Phase 3: Repair ----
    if repair_inputs:
        t0 = time.time()
        repair_solutions, repair_verify = await phase_repair(
            client, problems, repair_inputs, ckpt, args.batch_size,
        )
        t1 = time.time()
        log.info("Phase 3 (repair) done in %.1fs", t1 - t0)

        log.info("Applying repairs locally ...")
        n_repair_ok = apply_repairs(
            problems, exever_results, repair_inputs, repair_solutions, repair_verify,
        )
        log.info("Repair success: %d / %d", n_repair_ok, len(repair_inputs))
    else:
        log.info("No repairs needed")

    # ---- Phase 4: Sampled solutions (optional) ----
    sampled_solutions: Optional[Dict[str, List[str]]] = None
    if not args.no_samples:
        t0 = time.time()
        sampled_solutions = await phase_samples(client, problems, ckpt, args.batch_size)
        t1 = time.time()
        log.info("Phase 4 (samples) done in %.1fs", t1 - t0)

    # ---- Evaluation ----
    log.info("=" * 60)
    log.info("EVALUATION")
    log.info("=" * 60)

    baseline_data = evaluate_baselines(problems, greedy_solutions, sampled_solutions)

    exever_correct = sum(1 for r in exever_results if r["answer_correct"])
    exever_acc = exever_correct / len(problems)
    log.info("ExeVer accuracy: %.3f (%d/%d)", exever_acc, exever_correct, len(problems))

    # Echo chamber
    echo_results = [r for r in exever_results if r.get("echo_chamber") is not None]
    echo_pos = sum(1 for r in echo_results if r["echo_chamber"])
    echo_rate = echo_pos / len(echo_results) if echo_results else 0
    log.info("Echo chamber rate: %.3f (%d/%d)", echo_rate, echo_pos, len(echo_results))

    # Verdict distribution
    verdicts = Counter(r["verdict"] for r in exever_results)
    log.info("Verdict distribution:")
    for v, c in verdicts.most_common():
        log.info("  %s: %d", v, c)

    # Breakdowns
    has_samples = sampled_solutions is not None
    breakdowns = compute_breakdowns(
        problems,
        baseline_data["greedy_results"],
        baseline_data["sampled_results"],
        exever_results,
        has_samples,
    )

    # Print by-level
    print(f"\n{'='*60}")
    print("RESULTS BY DIFFICULTY LEVEL")
    print(f"{'='*60}")
    for lv in ["1", "2", "3", "4", "5"]:
        if lv in breakdowns["by_level"]:
            d = breakdowns["by_level"][lv]
            parts = [f"greedy={d['greedy']:.3f}", f"exever={d['exever']:.3f}"]
            if has_samples:
                parts.extend([
                    f"pass1={d.get('pass1',0):.3f}",
                    f"maj4={d.get('maj4',0):.3f}",
                    f"best4={d.get('best4',0):.3f}",
                ])
            print(f"  L{lv} (n={d['n']}): {' '.join(parts)}")

    # Print by-subject
    print(f"\n{'='*60}")
    print("RESULTS BY SUBJECT")
    print(f"{'='*60}")
    for subj, d in sorted(breakdowns["by_subject"].items()):
        parts = [f"greedy={d['greedy']:.3f}", f"exever={d['exever']:.3f}"]
        if has_samples:
            parts.extend([
                f"pass1={d.get('pass1',0):.3f}",
                f"maj4={d.get('maj4',0):.3f}",
                f"best4={d.get('best4',0):.3f}",
            ])
        print(f"  {subj} (n={d['n']}): {' '.join(parts)}")

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    header_parts = [f"{'Method':<20}", f"{'Overall':>8}"]
    for lv in [1, 2, 3, 4, 5]:
        header_parts.append(f"{'L'+str(lv):>6}")
    print(" ".join(header_parts))
    print("-" * 62)

    rows = [("CoT (greedy)", "greedy", baseline_data["greedy_acc"])]
    if has_samples:
        rows.extend([
            ("CoT (sampled)", "pass1", baseline_data["sampled_acc"]),
            ("Majority@4", "maj4", baseline_data["majority4_acc"]),
            ("Best-of-4", "best4", baseline_data["best4_acc"]),
        ])
    rows.append(("ExeVer", "exever", exever_acc))

    for name, key, overall in rows:
        parts = [f"{name:<20}", f"{overall:>7.1%}"]
        for lv in ["1", "2", "3", "4", "5"]:
            val = breakdowns["by_level"].get(lv, {}).get(key, 0)
            parts.append(f"{val:>5.1%}")
        print(" ".join(parts))

    # ---- Save results ----
    accuracy_dict: Dict[str, Any] = {
        "greedy_cot": baseline_data["greedy_acc"],
        "exever": exever_acc,
    }
    if has_samples:
        accuracy_dict["sampled_pass1"] = baseline_data["sampled_acc"]
        accuracy_dict["majority_4"] = baseline_data["majority4_acc"]
        accuracy_dict["best_of_4"] = baseline_data["best4_acc"]

    output = {
        "experiment": "exp13_math_full_together",
        "model": model_used,
        "n_problems": len(problems),
        "accuracy": accuracy_dict,
        "by_level": breakdowns["by_level"],
        "by_subject": breakdowns["by_subject"],
        "verifiability_map": breakdowns["verifiability_map"],
        "echo_chamber": {
            "rate": echo_rate,
            "n_echo": echo_pos,
            "n_total": len(echo_results),
        },
        "verification_stats": {
            **verify_stats,
            "avg_assertions": verify_stats["total_assertions"] / max(verify_stats["n_valid"], 1),
        },
        "repair": {
            "attempted": len(repair_inputs),
            "successful": sum(1 for r in exever_results if r.get("repaired", False)),
        },
        "verdicts": dict(verdicts),
        "exever_results": exever_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved results to %s", OUTPUT_PATH)

    # Clean up checkpoint on successful completion
    log.info("Run complete. Checkpoint kept at %s for reference.", CHECKPOINT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp13: Full MATH test set evaluation via Together AI",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip generating sampled solutions (greedy + ExeVer only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Checkpoint every N problems (default: 500)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Max concurrent API requests (default: 50)",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

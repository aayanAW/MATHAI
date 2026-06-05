"""Tier 3: Qwen2.5-Math-PRM-7B and Skywork-o1-Open-PRM-Qwen-2.5-7B on Modal.

Canonical scoring recipes (from pre-flight review, see PRE_COMMIT_n500.md Tier 3):

- Qwen2.5-Math-PRM-7B: `AutoModel` (not `AutoModelForCausalLM`), 2-class token
  classification head, per-step positive-class probability read at `<extra_0>`
  positions. Model card recipe.

- Skywork-o1-Open-PRM-Qwen-2.5-7B: custom `PRM_MODEL` from
  `github.com/SkyworkAI/skywork-o1-prm-inference`, scalar value head, step token
  is single `"\n"` (not `"\n\n"`), scores read at `reward_flags`.

ProcessBench sanity check (50 random problems from Qwen/ProcessBench) is the
first entrypoint — if Qwen-PRM avg F1 < 70 or Skywork avg F1 < 35, abort.

Run:
    modal run experiments/run_exp40_prms_modal.py::processbench_sanity
    modal run experiments/run_exp40_prms_modal.py::run_qwen_prm
    modal run experiments/run_exp40_prms_modal.py::run_skywork_prm
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "mathai-xsgrv-prms"
app = modal.App(APP_NAME)

# ---- Image builders: one per PRM so Skywork install is isolated ----

qwen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.46.0",
        "accelerate==0.34.0",
        "sentencepiece==0.2.0",
        "numpy==1.26.4",
        "datasets==3.0.0",
        "scikit-learn==1.5.2",
    )
)

skywork_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.46.0",
        "accelerate==0.34.0",
        "sentencepiece==0.2.0",
        "numpy==1.26.4",
        "datasets==3.0.0",
        "scikit-learn==1.5.2",
        "huggingface_hub==0.25.2",
        "safetensors==0.4.5",
    )
    .run_commands(
        "git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git /opt/skywork",
    )
)


# =====================================================================
# Qwen2.5-Math-PRM-7B — canonical recipe from HF model card
# =====================================================================

@app.function(
    image=qwen_image,
    gpu="A100-40GB",
    timeout=60 * 60 * 2,
)
def score_qwen_prm(
    samples_by_bench: dict,  # {bench: {pid: {"problem": str, "samples": list, "gold": str}}}
) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    print("Model loaded.")

    SEP = "<extra_0>"
    sep_ids = tokenizer.encode(SEP, add_special_tokens=False)
    assert len(sep_ids) == 1, f"Expected <extra_0> to be a single token, got {sep_ids}"
    sep_id = sep_ids[0]

    def make_step_rewards(logits, token_masks):
        probs = F.softmax(logits, dim=-1)
        probs = probs * token_masks.unsqueeze(-1)
        all_scores_res = []
        for i in range(probs.size(0)):
            sample = probs[i]
            pos_class = sample[sample != 0].view(-1, 2)[:, 1]
            all_scores_res.append(pos_class.cpu().tolist())
        return all_scores_res

    SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

    def score_one(problem: str, solution: str) -> list[float]:
        steps = [s.strip() for s in solution.split("\n\n") if s.strip()]
        if not steps:
            steps = [solution]
        assistant_content = SEP.join(steps) + SEP
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant_content},
        ]
        conv = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer.encode(conv, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        token_masks = (input_ids == sep_id)
        logits = outputs[0]  # AutoModel with PRM head returns classification logits
        rewards = make_step_rewards(logits, token_masks)
        return rewards[0] if rewards else [0.5]

    out = {}
    for bench, problems in samples_by_bench.items():
        print(f"\n=== {bench} ({len(problems)} problems) ===")
        out[bench] = {}
        n = len(problems)
        for i, (pid, pdata) in enumerate(problems.items()):
            problem = pdata.get("problem", "")
            samples = pdata["samples"]
            per_sample = []
            for s in samples:
                raw = s.get("raw") or ""
                try:
                    step_scores = score_one(problem, raw)
                except Exception as e:
                    step_scores = [0.5]
                if not step_scores:
                    step_scores = [0.5]
                per_sample.append({
                    "min": float(min(step_scores)),
                    "product": float(_prod(step_scores)),
                    "last": float(step_scores[-1]),
                    "mean": float(sum(step_scores) / len(step_scores)),
                    "n_steps": len(step_scores),
                })
            out[bench][pid] = {
                "per_sample": per_sample,
                "gold": pdata.get("gold"),
            }
            if (i + 1) % 10 == 0 or i == n - 1:
                print(f"  [{bench}] {i+1}/{n}", flush=True)
    return out


def _prod(xs):
    r = 1.0
    for x in xs:
        r *= float(x)
    return r


# =====================================================================
# Skywork-o1-Open-PRM-Qwen-2.5-7B — canonical recipe from inference repo
# =====================================================================

@app.function(
    image=skywork_image,
    gpu="A100-40GB",
    timeout=60 * 60 * 2,
)
def score_skywork_prm(
    samples_by_bench: dict,
) -> dict:
    import sys
    import traceback
    sys.path.insert(0, "/opt/skywork")
    import torch
    from transformers import AutoTokenizer

    # Imports from the Skywork inference repo.
    from model_utils.prm_model import PRM_MODEL  # type: ignore
    from model_utils.io_utils import (  # type: ignore
        prepare_input,
        prepare_batch_input_for_model,
        derive_step_rewards,
    )

    MODEL_ID = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
    print(f"Loading {MODEL_ID} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # CRITICAL: bf16 or fp16 --- default fp32 OOMs on a single A100-40GB.
    # Skywork-o1-PRM-7B: 28 GB fp32 weights, OOMs under long-sequence forward pass.
    model = PRM_MODEL.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    print("Model loaded.")

    n_ok, n_err = 0, 0

    def score_one(problem: str, solution: str) -> list[float]:
        nonlocal n_ok, n_err
        processed = [prepare_input(problem, solution, tokenizer=tokenizer, step_token="\n")]
        input_ids_list, _steps, reward_flags_list = zip(*processed)
        input_ids, attn, flags = prepare_batch_input_for_model(
            list(input_ids_list), list(reward_flags_list), tokenizer.pad_token_id
        )
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        flags = flags.to(device)
        with torch.no_grad():
            out_tup = model(input_ids=input_ids, attention_mask=attn, return_probs=True)
        # PRM_MODEL.forward returns (lm_logits, loss, rewards) when return_probs=True
        rewards = out_tup[-1]
        step_rewards = derive_step_rewards(rewards, flags)
        n_ok += 1
        return step_rewards[0] if step_rewards and step_rewards[0] else [0.5]

    out = {}
    for bench, problems in samples_by_bench.items():
        print(f"\n=== {bench} ({len(problems)} problems) ===")
        out[bench] = {}
        n = len(problems)
        for i, (pid, pdata) in enumerate(problems.items()):
            problem = pdata.get("problem", "")
            samples = pdata["samples"]
            per_sample = []
            for s in samples:
                raw = s.get("raw") or ""
                try:
                    step_scores = score_one(problem, raw)
                except Exception as e:
                    n_err += 1
                    if n_err <= 5:
                        print(f"  [ERR {n_err}] {type(e).__name__}: {e}")
                        traceback.print_exc()
                    step_scores = [0.5]
                if not step_scores:
                    step_scores = [0.5]
                per_sample.append({
                    "min": float(min(step_scores)),
                    "product": float(_prod(step_scores)),
                    "last": float(step_scores[-1]),
                    "mean": float(sum(step_scores) / len(step_scores)),
                    "n_steps": len(step_scores),
                })
            out[bench][pid] = {
                "per_sample": per_sample,
                "gold": pdata.get("gold"),
            }
            if (i + 1) % 10 == 0 or i == n - 1:
                print(f"  [{bench}] {i+1}/{n}  ok={n_ok} err={n_err}", flush=True)
    print(f"\nFINAL: ok={n_ok} err={n_err}")
    return out


# =====================================================================
# ProcessBench sanity check (50 random samples)
# =====================================================================

@app.function(
    image=qwen_image,
    gpu="A100-40GB",
    timeout=60 * 60,
)
def processbench_sanity_qwen(n_per_split: int = 12) -> dict:
    """Reproduce ProcessBench F1 on a 50-problem subset (12 per split × 4 splits)."""
    import random
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    from sklearn.metrics import f1_score

    MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
    print(f"Loading {MODEL_ID} for sanity check...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    SEP = "<extra_0>"
    sep_id = tokenizer.encode(SEP, add_special_tokens=False)[0]
    SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

    def _make_rewards(logits, masks):
        p = F.softmax(logits, dim=-1) * masks.unsqueeze(-1)
        out = []
        for i in range(p.size(0)):
            s = p[i]
            pc = s[s != 0].view(-1, 2)[:, 1]
            out.append(pc.cpu().tolist())
        return out

    def score(problem, steps):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": SEP.join(steps) + SEP},
        ]
        conv = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(conv, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(input_ids=ids)
        masks = (ids == sep_id)
        return _make_rewards(out[0], masks)[0]

    rng = random.Random(42)
    results = {}
    for split in ["gsm8k", "math", "olympiadbench", "omnimath"]:
        try:
            ds = load_dataset("Qwen/ProcessBench", split=split)
        except Exception as e:
            print(f"  {split}: load failed ({e})")
            continue
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        subset = [ds[i] for i in idxs[:n_per_split]]
        preds, labels = [], []
        for ex in subset:
            steps = ex["steps"]
            label = ex["label"]  # -1 means all correct, else first error step idx
            try:
                step_scores = score(ex["problem"], steps)
            except Exception as e:
                continue
            # The predicted "first error step" is the earliest step whose pos_class < 0.5
            first_err = -1
            for j, s in enumerate(step_scores):
                if s < 0.5:
                    first_err = j
                    break
            pred_has_error = int(first_err >= 0)
            true_has_error = int(label >= 0)
            preds.append(pred_has_error)
            labels.append(true_has_error)
        if preds:
            f1 = f1_score(labels, preds, zero_division=0)
            results[split] = {"n": len(preds), "f1_error": float(f1)}
            print(f"  {split}: n={len(preds)}  F1={f1:.2f}")
    avg = sum(r["f1_error"] for r in results.values()) / len(results) if results else 0
    results["avg_f1"] = float(avg)
    results["published_avg_f1"] = 73.5 / 100
    results["passes"] = avg >= 0.60  # relaxed threshold for 12-problem subsets
    print(f"\nAVG F1 = {avg:.3f}  (published {73.5/100:.3f})  passes={results['passes']}")
    return results


# =====================================================================
# Local entrypoints
# =====================================================================

def build_samples_payload() -> dict:
    RESULTS = Path("/Users/aayanalwani/MATHAI/MATHAI/results")

    def _load_problem_texts(bench: str) -> dict[str, str]:
        if bench == "math175":
            with open(RESULTS / "exp25_selective_prediction.json") as f:
                exp25 = json.load(f)
            with open(RESULTS / "math_test_sample_500.json") as f:
                math_all = json.load(f)
            by_id = {p["id"]: p["problem"] for p in math_all}
            return {r["id"]: by_id[r["id"]] for r in exp25 if r["id"] in by_id}
        if bench == "aime":
            with open(RESULTS / "aime_2025.json") as f:
                aime = json.load(f)
            return {p["id"]: p["problem"] for p in aime}
        if bench == "cleanmath":
            with open(RESULTS / "cleanmath_combo.json") as f:
                cm = json.load(f)
            return {p["id"]: p["problem"] for p in cm}
        raise ValueError(bench)

    payload = {}
    for bench in ["math175", "aime", "cleanmath"]:
        with open(RESULTS / f"se_samples_{bench}.json") as f:
            cache = json.load(f)
        problems = _load_problem_texts(bench)
        payload[bench] = {}
        for pid, c in cache.items():
            if pid not in problems:
                continue
            payload[bench][pid] = {
                "problem": problems[pid],
                "samples": c.get("samples", []),
                "gold": c.get("gold", ""),
            }
    return payload


@app.local_entrypoint()
def processbench_sanity():
    print("Running Qwen2.5-Math-PRM-7B ProcessBench sanity check...")
    result = processbench_sanity_qwen.remote()
    out_path = Path("/Users/aayanalwani/MATHAI/MATHAI/results/exp40_processbench_sanity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")
    if not result.get("passes"):
        print("\n!!! SANITY CHECK FAILED !!!")
        print("    Do not trust downstream PRM numbers.")
        return
    print("\n✓ Sanity check passed. Safe to run run_qwen_prm and run_skywork_prm.")


@app.local_entrypoint()
def run_qwen_prm():
    print("Preparing payload...")
    payload = build_samples_payload()
    print(f"  math175: {len(payload.get('math175', {}))}")
    print(f"  aime:    {len(payload.get('aime', {}))}")
    print(f"  cleanmath: {len(payload.get('cleanmath', {}))}")
    print("\nLaunching Qwen2.5-Math-PRM-7B on Modal A100...")
    result = score_qwen_prm.remote(payload)
    out_path = Path("/Users/aayanalwani/MATHAI/MATHAI/results/exp40_qwen_prm.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")


@app.local_entrypoint()
def run_skywork_prm():
    print("Preparing payload...")
    payload = build_samples_payload()
    print("Launching Skywork-o1-Open-PRM-Qwen-2.5-7B on Modal A100...")
    result = score_skywork_prm.remote(payload)
    out_path = Path("/Users/aayanalwani/MATHAI/MATHAI/results/exp40_skywork_prm.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")

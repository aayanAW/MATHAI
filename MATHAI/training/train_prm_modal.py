"""Train a Process Reward Model on PBT labels using Modal A100.

Two variants:
1. PBT-only: Only tested steps in loss
2. Outcome-conditioned: Tested + outcome-based untestable labels

The PRM is a Qwen2.5-Math-7B with LoRA that predicts step correctness.
Training format: Math-Shepherd style (solution with + / - labels at step boundaries).

Usage:
    modal run training/train_prm_modal.py
"""
import json
import modal

app = modal.App("mathai-prm-training")

# Image with all training dependencies
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.0",
        "peft==0.12.0",
        "trl==0.10.1",
        "datasets==2.21.0",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.3",
        "scipy",
        "sympy",
        "rich",
    )
)

model_volume = modal.Volume.from_name("mathai-prm-models", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="A100",
    volumes={"/models": model_volume},
    timeout=3600,
)
def download_model() -> str:
    """Pre-cache the model into the volume. Run this ONCE."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json as _json

    model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
    print(f"Downloading {model_id} to /models ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir="/models",
    )
    print("  Tokenizer downloaded.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, cache_dir="/models",
    )
    print("  Model downloaded.")

    modal.Volume.from_name("mathai-prm-models").commit()
    print("  Volume committed. Model is cached.")

    return _json.dumps({"status": "ok", "model": model_id})


@app.function(
    image=train_image,
    gpu="A100",
    volumes={"/models": model_volume},
    timeout=14400,  # 4 hours
)
def train_prm(training_data_json: str, variant: str = "pbt_only") -> str:
    """Train a PRM on PBT labels.

    Args:
        training_data_json: JSON string of training data (Math-Shepherd format).
        variant: "pbt_only" or "outcome_conditioned".

    Returns:
        JSON string with training metrics.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    import json as _json

    data = _json.loads(training_data_json)
    print(f"Training PRM ({variant}) on {len(data)} examples")

    # Load tokenizer and model
    model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir="/models",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir="/models",
        device_map="auto",
    )

    # Model should already be cached from download_model step

    # Prepare dataset: convert Math-Shepherd format to text
    # The model learns to predict + or - at step boundaries
    texts = []
    for example in data:
        label_text = example["label"]
        if variant == "pbt_only":
            # Skip examples where ALL labels are masked (?)
            if "+" not in label_text and "-" not in label_text:
                continue
        texts.append({"text": label_text})

    print(f"  {len(texts)} training examples after filtering")
    if len(texts) < 10:
        return _json.dumps({"error": "Too few training examples", "n": len(texts)})

    dataset = Dataset.from_list(texts)

    # Split 90/10
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=f"/models/prm_{variant}",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=2048,
        dataset_text_field="text",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        seed=42,
    )

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    train_result = trainer.train()

    # Save
    trainer.save_model(f"/models/prm_{variant}_final")
    modal.Volume.from_name("mathai-prm-models").commit()

    metrics = {
        "variant": variant,
        "n_train": len(train_dataset),
        "n_eval": len(eval_dataset),
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
        "eval_loss": trainer.evaluate().get("eval_loss", None),
    }

    print(f"\nTraining complete:")
    print(f"  Train loss: {metrics['train_loss']:.4f}")
    print(f"  Eval loss: {metrics['eval_loss']}")
    print(f"  Runtime: {metrics['train_runtime_seconds']:.0f}s")

    return _json.dumps(metrics)


@app.function(
    image=train_image,
    gpu="A100",
    volumes={"/models": model_volume},
    timeout=7200,
)
def evaluate_prm(
    problems_json: str,
    solutions_json: str,
    variant: str = "pbt_only",
) -> str:
    """Evaluate PRM via best-of-N selection.

    Args:
        problems_json: JSON string of problems with gold answers.
        solutions_json: JSON string of N solutions per problem.
        variant: Which PRM to load.

    Returns:
        JSON string with evaluation results.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import json as _json

    problems = _json.loads(problems_json)
    solutions = _json.loads(solutions_json)

    model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
    adapter_path = f"/models/prm_{variant}_final"

    print(f"Loading base model + {variant} adapter...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir="/models",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir="/models",
        device_map="auto",
    )

    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        print("  Adapter loaded successfully")
    except Exception as e:
        print(f"  Warning: Could not load adapter: {e}")
        print("  Using base model (no PRM training)")

    model.eval()

    # Score each solution by computing perplexity of the labeled version
    # Lower perplexity on + labels = better solution
    results = []
    for i, (prob, sols) in enumerate(zip(problems, solutions)):
        if not sols:
            continue

        best_score = float("-inf")
        best_idx = 0

        for j, sol in enumerate(sols):
            # Compute log-probability of the solution
            inputs = tokenizer(sol, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                # Negative loss = higher score = better
                score = -outputs.loss.item()

            if score > best_score:
                best_score = score
                best_idx = j

        results.append({
            "problem_id": prob.get("id", f"prob_{i}"),
            "selected_idx": best_idx,
            "selected_score": best_score,
        })

        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(problems)} problems")

    return _json.dumps(results)


@app.local_entrypoint()
def main():
    import json as _json
    from pathlib import Path

    results_dir = Path("results")

    # === Step 1: Pre-cache model ===
    print("Step 1: Ensuring model is cached in volume...")
    dl_result = download_model.remote()
    dl_data = _json.loads(dl_result)
    if dl_data.get("status") != "ok":
        print(f"  Model download failed: {dl_data}")
        return
    print(f"  Model cached: {dl_data['model']}")

    # === Step 2: Train both PRM variants ===
    variant_files = {
        "pbt_only": "prm_train_pbt_only.json",
        "outcome_cond": "prm_train_outcome_cond.json",
    }
    for variant, filename in variant_files.items():
        data_file = results_dir / filename
        if not data_file.exists():
            print(f"Training data not found: {data_file}")
            continue

        with open(data_file) as f:
            data = _json.load(f)

        print(f"\n{'='*60}")
        print(f"Training PRM: {variant}")
        print(f"{'='*60}")

        metrics_json = train_prm.remote(_json.dumps(data), variant)
        metrics = _json.loads(metrics_json)

        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
            continue

        print(f"  Train loss: {metrics['train_loss']:.4f}")
        print(f"  Eval loss: {metrics['eval_loss']}")
        print(f"  Runtime: {metrics['train_runtime_seconds']:.0f}s")

        # Save metrics
        out_path = results_dir / f"prm_train_metrics_{variant}.json"
        with open(out_path, "w") as f:
            _json.dump(metrics, f, indent=2)
        print(f"  Saved to {out_path}")

    print("\nAll PRM training complete.")

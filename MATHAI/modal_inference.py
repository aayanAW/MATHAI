"""Modal GPU inference for ExeVer experiments.

Deploys Qwen2.5-Math-7B-Instruct on an H100 GPU via vLLM.
"""
import modal

app = modal.App("exever-inference")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers==4.49.0",
        "vllm==0.7.3",
        "numpy<2",
    )
)

MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
MODEL_DIR = "/models"
model_volume = modal.Volume.from_name("exever-models", create_if_missing=True)


@app.cls(
    image=vllm_image,
    gpu="H100",
    volumes={MODEL_DIR: model_volume},
    timeout=1800,
    scaledown_window=300,
)
class QwenMathModel:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        self.llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            download_dir=MODEL_DIR,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )
        model_volume.commit()

    @modal.method()
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.95,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    @modal.method()
    def generate_batch(self, prompts: list, temperature: float = 0.0, max_tokens: int = 2048) -> list:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.95,
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


@app.local_entrypoint()
def main():
    model = QwenMathModel()

    test_prompt = """Solve the following math problem step by step.

Format your solution with clear step markers:
## Step 1: [brief title]
[reasoning and computation for this step]

At the end, state your final answer as: The answer is \\boxed{answer}.

Problem: What is the sum of the solutions of x^2 - 5x + 6 = 0?"""

    print("Sending test prompt to Qwen2.5-Math-7B on Modal H100...")
    response = model.generate.remote(test_prompt)
    print(f"\n=== Model Response ===\n{response}\n")

    # Quick batch test
    prompts = [test_prompt, test_prompt.replace("x^2 - 5x + 6", "x^2 - 7x + 12")]
    print("Testing batch (2 problems)...")
    responses = model.generate_batch.remote(prompts)
    for i, r in enumerate(responses):
        print(f"\n--- Problem {i+1} ---")
        print(r[:300] + "..." if len(r) > 300 else r)

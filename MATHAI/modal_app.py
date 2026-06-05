"""Modal deployment for ExeVer GPU inference.

Deploys two vLLM servers (solver + verifier) on separate GPUs.
Uses A10G for cost efficiency ($0.50-1.00/hr per GPU).

Usage:
    modal run modal_app.py  # Deploy and test
    modal serve modal_app.py  # Keep running for batch requests
"""
import modal

# Modal app definition
app = modal.App("exever-math")

# Container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "sympy>=1.12",
        "datasets>=3.0",
        "numpy>=1.24",
        "tqdm",
    )
)

# Model volume for caching downloaded models
model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)

# Solver model (Qwen2.5-Math-7B-Instruct)
SOLVER_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
# Verifier model (cross-model: DeepSeek-R1-Distill-Qwen-7B)
VERIFIER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


@app.cls(
    image=image,
    gpu=modal.gpu.A10G(),
    volumes={"/models": model_cache},
    timeout=600,
    container_idle_timeout=300,
)
class SolverModel:
    """vLLM server for the solver model (Pass 1: CoT generation)."""

    @modal.enter()
    def setup(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=SOLVER_MODEL,
            trust_remote_code=True,
            download_dir="/models",
            max_model_len=4096,
        )
        self.default_params = SamplingParams(
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
        )

    @modal.method()
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=2048,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.95,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    @modal.method()
    def generate_batch(self, prompts: list, temperature: float = 0.0) -> list:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=2048,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.95,
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


@app.cls(
    image=image,
    gpu=modal.gpu.A10G(),
    volumes={"/models": model_cache},
    timeout=600,
    container_idle_timeout=300,
)
class VerifierModel:
    """vLLM server for the verifier model (Pass 2: verification code)."""

    @modal.enter()
    def setup(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=VERIFIER_MODEL,
            trust_remote_code=True,
            download_dir="/models",
            max_model_len=4096,
        )
        self.default_params = SamplingParams(
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
        )

    @modal.method()
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=2048,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.95,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    @modal.method()
    def generate_batch(self, prompts: list, temperature: float = 0.0) -> list:
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=2048,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.95,
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


@app.local_entrypoint()
def main():
    """Test deployment by running a simple math problem."""
    solver = SolverModel()
    verifier = VerifierModel()

    test_problem = "What is the sum of the solutions of x^2 - 5x + 6 = 0?"

    # Test solver
    from src.exever.prompts import format_solve_prompt, format_verify_prompt

    solve_prompt = format_solve_prompt(test_problem)
    print("=== Solver (Pass 1) ===")
    solution = solver.generate.remote(solve_prompt)
    print(solution[:500])

    # Test verifier
    verify_prompt = format_verify_prompt(solution)
    print("\n=== Verifier (Pass 2) ===")
    verification = verifier.generate.remote(verify_prompt)
    print(verification[:500])

    # Test execution locally
    from src.exever.executor import execute_verification_script
    from src.exever.step_parser import extract_python_code

    code = extract_python_code(verification)
    print("\n=== Execution ===")
    result = execute_verification_script(code)
    print(f"Verdict: {result.verdict}")
    if result.answer_extracted:
        print(f"Answer: {result.answer_extracted}")

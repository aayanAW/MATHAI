"""Unified inference interface for LLM generation.

Supports:
- vLLM (for fast batched inference on GPU)
- HuggingFace Transformers (fallback)
- Mock (for testing without GPU)
"""
import os
from typing import Dict, List, Optional


class ModelWrapper:
    """Base class for LLM inference."""

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]


class MockModel(ModelWrapper):
    """Mock model for testing pipeline without GPU.

    Returns predefined responses or empty strings.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        # Check for keyword matches in responses
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return f"[Mock response #{self.call_count}]"


class VLLMModel(ModelWrapper):
    """vLLM-based inference for fast GPU generation."""

    def __init__(
        self,
        model_id: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        from vllm import LLM, SamplingParams
        self.model_id = model_id
        self.llm = LLM(model=model_id, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]


class HFModel(ModelWrapper):
    """HuggingFace Transformers inference (slower, for fallback)."""

    def __init__(
        self,
        model_id: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        device: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 0.01),
                do_sample=self.temperature > 0,
            )
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response


def create_model(
    model_id: str,
    backend: str = "mock",
    **kwargs,
) -> ModelWrapper:
    """Factory function to create a model wrapper.

    Args:
        model_id: HuggingFace model ID.
        backend: One of "vllm", "hf", "mock".
    """
    if backend == "mock":
        return MockModel()
    elif backend == "vllm":
        return VLLMModel(model_id, **kwargs)
    elif backend == "hf":
        return HFModel(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

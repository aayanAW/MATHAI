"""Thin wrappers around external LLM APIs.

Supports:
  - Together AI (Llama-3.3-70B, DeepSeek-V3, gpt-oss-120b, Qwen3-Coder-480B, etc.)
  - OpenAI (GPT-5-mini, gpt-4o, etc.)
  - Anthropic (Claude Sonnet 4.6, Opus 4.7, etc.)
  - Google (Gemini 2.5 Pro)
  - Mistral (Mistral Large)

Each wrapper takes a frozen system + user prompt and returns the raw
string response. Retries, rate limiting, and caching are caller's
responsibility.

API keys are read from environment variables; no keys are baked into the
code. See README.md for setup instructions.
"""
from __future__ import annotations

import os
from typing import Callable


def together_call(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Call Together AI. Requires ``TOGETHER_API_KEY`` env var."""
    try:
        from together import Together
    except ImportError as e:
        raise RuntimeError("pip install together") from e
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("set TOGETHER_API_KEY environment variable")
    client = Together(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    if msg is None or msg.content is None:
        raise RuntimeError("Together AI returned empty content")
    return msg.content


def openai_call(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Call OpenAI. Requires ``OPENAI_API_KEY`` env var.

    Handles both the legacy Chat Completions parameters (``max_tokens``,
    ``temperature``) used by gpt-4 / gpt-4o and the newer reasoning-model
    parameters (``max_completion_tokens``, no ``temperature``) used by
    gpt-5 / o-series. Falls back if the API rejects the legacy params.
    """
    try:
        from openai import BadRequestError, OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY environment variable")
    client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    is_reasoning = (model.startswith("gpt-5") or model.startswith("o1")
                    or model.startswith("o3") or model.startswith("o4"))
    # Try legacy params first.
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except BadRequestError:
        # Newer models reject ``max_tokens`` and/or ``temperature``.
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        except BadRequestError:
            # Reasoning-model fallback: no temperature, low reasoning
            # effort to keep latency tractable for batch runs.
            kwargs: dict = {"max_completion_tokens": max_tokens}
            if is_reasoning:
                kwargs["reasoning_effort"] = "low"
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned empty content")
    return content


def anthropic_call(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Call Anthropic. Requires ``ANTHROPIC_API_KEY`` env var.

    Some 2026 Anthropic models (Opus 4.7) deprecate the ``temperature``
    parameter and reject the legacy call shape. We retry without
    ``temperature`` on a 400 ``invalid_request_error`` indicating the
    deprecation, matching the OpenAI ``gpt-5`` family fallback pattern.
    """
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic") from e
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("set ANTHROPIC_API_KEY environment variable")
    client = anthropic.Anthropic(api_key=api_key)

    def _do_call(include_temperature: bool) -> "anthropic.types.Message":
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if include_temperature:
            kwargs["temperature"] = temperature
        return client.messages.create(**kwargs)

    try:
        resp = _do_call(include_temperature=True)
    except anthropic.BadRequestError as e:
        if "temperature" in str(e).lower():
            resp = _do_call(include_temperature=False)
        else:
            raise

    block = resp.content[0]
    if not hasattr(block, "text"):
        raise RuntimeError(f"Anthropic returned unexpected block: {type(block).__name__}")
    return block.text  # type: ignore[attr-defined]


def google_call(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Call Google Gemini. Requires ``GOOGLE_API_KEY`` env var.

    Gemini 2.5 models internally consume part of max_output_tokens on
    thinking tokens. We disable thinking with thinking_budget=0 so the
    visible response is the full max_tokens.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as e:
        raise RuntimeError("pip install google-genai") from e
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("set GOOGLE_API_KEY environment variable")
    client = genai.Client(api_key=api_key)
    cfg_obj = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=(
            genai_types.ThinkingConfig(thinking_budget=0)
            if ("2.5" in model or "3." in model) else None
        ),
    )
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=cfg_obj,
    )
    return resp.text or ""


# Registry of frontier extractors used in the DAJV paper.
EXTRACTOR_REGISTRY: dict[str, dict] = {
    "E01_llama_3_3_70B": {
        "provider": "together",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "E03_deepseek_v3": {
        "provider": "together",
        "model": "deepseek-ai/DeepSeek-V3",
    },
    "E05_gpt_oss_120B": {
        "provider": "together",
        "model": "openai/gpt-oss-120B-Instruct-Turbo",
    },
    "E06_gpt_5_mini": {
        "provider": "openai",
        "model": "gpt-5-mini",
    },
    "E07_claude_sonnet_4_6": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6-20251101",
    },
    "E09_qwen3_coder_480B": {
        "provider": "together",
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    },
    # Together AI extractors (E02 Llama-4-Maverick, E04 DeepSeek-R1, E10
    # Qwen2.5-72B) are gated on the user adding prepaid credit to the
    # Together account. The key works but the account currently returns
    # HTTP 402. Registered below for resumption when credit is added.
    "E02_llama_4_maverick": {
        "provider": "together",
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    },
    "E04_deepseek_r1_distill_70B": {
        "provider": "together",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    },
    "E10_qwen2_5_72B": {
        "provider": "together",
        "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    },
    # OpenAI scale-up (replaces blocked Together extractors). Adds 3
    # within-OpenAI siblings to the existing E05 + E06, enabling H6
    # within-lab dependency analysis with k_within = 5.
    "E08S_gpt_4o": {
        "provider": "openai",
        "model": "gpt-4o-2024-11-20",
    },
    "E04S_gpt_4_1": {
        "provider": "openai",
        "model": "gpt-4.1-2025-04-14",
    },
    "E10S_gpt_5": {
        "provider": "openai",
        "model": "gpt-5-2025-08-07",
    },
    # Google extractor.
    "E12_gemini_2_5_flash": {
        "provider": "google",
        "model": "gemini-2.5-flash",
    },
    # Within-Anthropic scale-up. Adds three Anthropic siblings to E07
    # (Sonnet 4.6) so that the within-Anthropic pair count goes from 0
    # to C(4, 2) = 6.
    "E01A_claude_opus_4_7": {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
    },
    "E02A_claude_opus_4_6": {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
    },
    "E03A_claude_haiku_4_5": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
    },
    # Meta + Qwen serverless additions (unlocked by user's Together AI
    # credit top-up).  Adds lab diversity for H6 (Meta is a new lab;
    # Qwen3-235B is a within-Qwen sibling to E09 Qwen3-Coder-480B).
    "E13_llama_3_3_70B": {
        "provider": "together",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "E14_qwen_3_235B": {
        "provider": "together",
        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    },
    # E11 (Mistral-Large) requires an additional API key not provided.
}


def make_extractor_call(extractor_id: str) -> Callable[[str], str]:
    """Return a frozen ``str -> str`` extractor call for the given id."""
    cfg = EXTRACTOR_REGISTRY[extractor_id]
    provider = cfg["provider"]
    model = cfg["model"]
    if provider == "together":
        return lambda p: together_call(p, model=model)
    if provider == "openai":
        return lambda p: openai_call(p, model=model)
    if provider == "anthropic":
        return lambda p: anthropic_call(p, model=model)
    if provider == "google":
        return lambda p: google_call(p, model=model)
    raise ValueError(f"unknown provider: {provider}")

"""Parse an extractor's raw response into an ExtractionResult.

The extractor is expected to return either:
  - a fenced Python code block containing ``def verify(answer)``, OR
  - the literal token ``UNVERIFIABLE``.

Anything else is treated as a parsing failure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from verifyensemble.extractors.prompt import EXTRACTION_PROMPT


@dataclass
class ExtractionResult:
    """Result of the extraction step."""
    script: Optional[str]
    unverifiable: bool
    raw_response: str
    error: Optional[str] = None


def extract_verifier(
    problem: str,
    llm_call: Callable[[str], str],
) -> ExtractionResult:
    """Run the extractor LLM on a problem and parse the response.

    Args:
        problem: The math problem statement.
        llm_call: A callable ``str -> str``. Should return the raw LLM
            response. Caller is responsible for retries / temperature
            / max_tokens.
    """
    try:
        raw = llm_call(EXTRACTION_PROMPT.format(problem=problem))
    except Exception as e:
        return ExtractionResult(None, False, "", error=f"llm_call failed: {e}")

    raw = raw.strip()

    if re.search(r"^\s*UNVERIFIABLE\s*$", raw, re.MULTILINE) or raw == "UNVERIFIABLE":
        return ExtractionResult(None, True, raw)

    code_match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
    if not code_match:
        code_match = re.search(r"```\n(.*?)```", raw, re.DOTALL)
    if not code_match:
        if "def verify" in raw:
            return ExtractionResult(raw, False, raw)
        return ExtractionResult(None, False, raw, error="no code block found")

    code = code_match.group(1).strip()
    if "def verify" not in code:
        return ExtractionResult(None, False, raw, error="no verify function defined")

    return ExtractionResult(code, False, raw)

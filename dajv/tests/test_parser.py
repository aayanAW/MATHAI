"""Tests for the extractor response parser."""
from __future__ import annotations

from verifyensemble.extractors.parser import ExtractionResult, extract_verifier


def _make_call(response: str):
    def call(_prompt: str) -> str:
        return response
    return call


def test_parses_valid_python_block():
    raw = "```python\ndef verify(answer):\n    return answer == '42'\n```"
    r = extract_verifier("ignored", _make_call(raw))
    assert r.script is not None
    assert "def verify" in r.script
    assert not r.unverifiable
    assert r.error is None


def test_recognizes_unverifiable_token_standalone():
    r = extract_verifier("ignored", _make_call("UNVERIFIABLE"))
    assert r.unverifiable is True
    assert r.script is None


def test_recognizes_unverifiable_token_with_leading_ws():
    r = extract_verifier("ignored", _make_call("   UNVERIFIABLE   "))
    assert r.unverifiable is True


def test_handles_bare_def_verify_outside_code_block():
    raw = "def verify(answer):\n    return False\n"
    r = extract_verifier("ignored", _make_call(raw))
    assert r.script is not None  # script is captured from raw response
    assert not r.unverifiable


def test_handles_unfenced_python_block_with_default_fence():
    raw = "```\ndef verify(answer):\n    return True\n```"
    r = extract_verifier("ignored", _make_call(raw))
    assert r.script is not None
    assert "verify" in r.script


def test_returns_parse_error_when_no_code_block_and_no_def():
    raw = "Sorry, I cannot construct a verifier for this problem."
    r = extract_verifier("ignored", _make_call(raw))
    assert r.script is None
    assert r.unverifiable is False
    assert r.error is not None


def test_handles_llm_call_exception():
    def call(_prompt: str) -> str:
        raise RuntimeError("network down")
    r = extract_verifier("ignored", call)
    assert r.script is None
    assert r.error is not None
    assert "network down" in r.error


def test_extraction_result_dataclass_fields():
    r = ExtractionResult(script="x", unverifiable=False, raw_response="raw")
    assert r.script == "x"
    assert r.error is None

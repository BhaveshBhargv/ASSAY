"""Calls the LLM and parses its reply into a RecommendationReport.

Models don't always return clean JSON, so parsing tries the whole reply first,
then a fenced code block, then the first balanced {...} in the text.
"""
from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from .ports import ILLMProvider
from .report import RecommendationReport

log = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when the model's output can't be turned into a valid report."""


def generate_report(provider: ILLMProvider, system: str, user: str) -> RecommendationReport:
    raw = provider.complete(system, user)
    data = _extract_json(raw)
    if data is None:
        raise GenerationError(f"no JSON object found in model output:\n{raw[:400]}")
    try:
        return RecommendationReport.model_validate(data)
    except ValidationError as exc:
        raise GenerationError(f"model output failed schema validation: {exc}") from exc


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull a single JSON object out of an LLM reply, or return None."""
    if not text:
        return None

    # the whole reply
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # a ```json ... ``` block
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # the first balanced {...}
    span = _first_balanced_object(text)
    if span:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            return None
    return None


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None

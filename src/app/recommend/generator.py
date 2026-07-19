"""
generator.py — call the LLM and parse its output into a validated report.

Responsibility (single): given a system+user prompt via the ILLMProvider port,
obtain the completion and turn it into a `RecommendationReport`. Local models are
not always perfectly obedient about JSON, so parsing is defensive: try strict
JSON, then extract the first balanced JSON object from the text, then validate
against the Pydantic schema. Prompt-building and grounding live elsewhere; this
module only bridges "prompt -> validated object".
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
    """Raised when the model output cannot be parsed into a valid report."""


def generate_report(provider: ILLMProvider, system: str, user: str) -> RecommendationReport:
    raw = provider.complete(system, user)
    data = _extract_json(raw)
    if data is None:
        raise GenerationError(f"no JSON object found in model output:\n{raw[:400]}")
    try:
        return RecommendationReport.model_validate(data)
    except ValidationError as exc:
        raise GenerationError(f"model output failed schema validation: {exc}") from exc


# --- robust JSON extraction ------------------------------------------------ #

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Best-effort recovery of a single JSON object from an LLM response."""
    if not text:
        return None

    # 1) whole string is JSON
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # 2) fenced ```json ... ``` block
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) first balanced { ... } span
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

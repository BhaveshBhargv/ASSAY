"""The interface every LLM provider implements.

It's plain text in and text out, so providers can be swapped and tests can use
a fake one without a network connection or a model.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ILLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply to a system and user message."""
        ...

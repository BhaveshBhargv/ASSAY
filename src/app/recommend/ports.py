"""
ports.py — the LLM abstraction (Dependency Inversion).

The generator depends ONLY on this tiny string-in/string-out port, never on any
concrete SDK or framework. Real adapters (LangChain ChatOllama / ChatAnthropic)
and the offline FakeProvider all satisfy it, so the provider is swappable and the
core is unit-testable without a network or a running model.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ILLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for a system + user message pair."""
        ...
